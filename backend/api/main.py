"""FastAPI application for SkyVision-DL facade inspections."""

from __future__ import annotations

import os
import zipfile
import csv
import io
import json
from datetime import datetime, timezone
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import shutil
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.api.auth import router as auth_router
from backend.analytics.deduplication import estimate_unique_detections
from backend.auth.dependencies import get_current_user, require_admin
from backend.database.connection import SessionLocal, get_db, init_database
from backend.models.inspection_models import (
    DefectDetection,
    ExtractedFrame,
    InspectionJob,
    JobStatus,
    Report,
    ReportType,
    UploadedVideo,
    User,
    UserRole,
    AuditAction,
)
from backend.reports.pdf_report_generator import generate_pdf_report
from backend.services.audit_logger import record_audit_event
from backend.services.job_processor import process_inspection_job
from backend.services.inspection_pipeline import (
    DEFAULT_MODEL_PATH,
    STORAGE_ROOT,
    create_inspection_id,
)
from backend.yolo.detector import is_ultralytics_model_alias


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_ROOT = PROJECT_ROOT / "frontend"
UPLOAD_ROOT = STORAGE_ROOT / "uploads"
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_VIDEO_MIME_TYPES = {"video/mp4", "video/x-msvideo", "application/octet-stream"}
ALLOWED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "application/octet-stream"}
MAX_VIDEO_UPLOAD_BYTES = int(os.getenv("SKYVISION_MAX_VIDEO_UPLOAD_MB", "500")) * 1024 * 1024
MAX_IMAGE_UPLOAD_BYTES = int(os.getenv("SKYVISION_MAX_IMAGE_UPLOAD_MB", "50")) * 1024 * 1024
LOG_ROOT = STORAGE_ROOT / "logs"
CONFIG_ROOT = STORAGE_ROOT / "config"
BACKUP_ROOT = STORAGE_ROOT / "backups"
SETTINGS_PATH = CONFIG_ROOT / "settings.json"
logger = logging.getLogger(__name__)


class ArchiveRequest(BaseModel):
    archived: bool


class SettingsUpdate(BaseModel):
    default_sample_fps: float | None = None
    default_confidence: float | None = None
    default_device: str | None = None
    default_model_path: str | None = None
    max_video_upload_mb: int | None = None
    max_image_upload_mb: int | None = None


def get_allowed_origins() -> list[str]:
    raw_origins = os.getenv("SKYVISION_ALLOWED_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000")
    origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    return origins or ["http://127.0.0.1:8000"]


def configure_local_logging() -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_file = LOG_ROOT / "skyvision.log"
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if any(isinstance(handler, RotatingFileHandler) and getattr(handler, "baseFilename", "") == str(log_file) for handler in root_logger.handlers):
        return
    handler = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root_logger.addHandler(handler)


def default_local_settings() -> dict[str, Any]:
    return {
        "default_sample_fps": 1.0,
        "default_confidence": 0.35,
        "default_device": "0",
        "default_model_path": os.getenv("SKYVISION_MODEL_PATH", str(DEFAULT_MODEL_PATH)),
        "max_video_upload_mb": int(os.getenv("SKYVISION_MAX_VIDEO_UPLOAD_MB", "500")),
        "max_image_upload_mb": int(os.getenv("SKYVISION_MAX_IMAGE_UPLOAD_MB", "50")),
    }


def load_local_settings() -> dict[str, Any]:
    settings = default_local_settings()
    if not SETTINGS_PATH.exists():
        return settings

    try:
        stored_settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("Unable to read local settings file")
        return settings

    if isinstance(stored_settings, dict):
        settings.update({key: value for key, value in stored_settings.items() if key in settings})
    try:
        return normalize_settings(settings)
    except (TypeError, ValueError, HTTPException):
        logger.exception("Local settings file contains invalid values")
        return default_local_settings()


def normalize_settings(settings: dict[str, Any]) -> dict[str, Any]:
    defaults = default_local_settings()
    normalized = dict(defaults)
    normalized.update(settings)
    normalized["default_sample_fps"] = float(normalized["default_sample_fps"])
    normalized["default_confidence"] = float(normalized["default_confidence"])
    normalized["default_device"] = str(normalized["default_device"]).strip() or defaults["default_device"]
    normalized["default_model_path"] = str(normalized["default_model_path"]).strip() or defaults["default_model_path"]
    normalized["max_video_upload_mb"] = int(normalized["max_video_upload_mb"])
    normalized["max_image_upload_mb"] = int(normalized["max_image_upload_mb"])
    validate_settings(normalized)
    return normalized


def validate_settings(settings: dict[str, Any]) -> None:
    if settings["default_sample_fps"] <= 0:
        raise HTTPException(status_code=400, detail="Default sample FPS must be greater than zero.")
    if settings["default_confidence"] <= 0 or settings["default_confidence"] >= 1:
        raise HTTPException(status_code=400, detail="Default confidence must be between 0 and 1.")
    if settings["max_video_upload_mb"] <= 0:
        raise HTTPException(status_code=400, detail="Maximum video upload size must be greater than zero.")
    if settings["max_image_upload_mb"] <= 0:
        raise HTTPException(status_code=400, detail="Maximum image upload size must be greater than zero.")


def save_local_settings(payload: SettingsUpdate) -> dict[str, Any]:
    settings = load_local_settings()
    updates = (
        payload.model_dump(exclude_unset=True)
        if hasattr(payload, "model_dump")
        else payload.dict(exclude_unset=True)
    )
    settings.update(updates)
    settings = normalize_settings(settings)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return settings


def upload_size_limits_bytes() -> dict[str, int]:
    settings = load_local_settings()
    return {
        "video": int(settings["max_video_upload_mb"]) * 1024 * 1024,
        "image": int(settings["max_image_upload_mb"]) * 1024 * 1024,
    }


configure_local_logging()


app = FastAPI(title="SkyVision-DL", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

app.mount("/artifacts", StaticFiles(directory=str(STORAGE_ROOT)), name="artifacts")
if FRONTEND_ROOT.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_ROOT)), name="frontend")
app.include_router(auth_router)


@app.on_event("startup")
def on_startup() -> None:
    init_database()
    ensure_default_admin_role()


@app.get("/")
def index() -> FileResponse:
    index_path = FRONTEND_ROOT / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(index_path)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config")
def config() -> dict[str, Any]:
    settings = load_local_settings()
    raw_path_str = settings["default_model_path"]
    if is_ultralytics_model_alias(raw_path_str):
        model_path_str = raw_path_str
        model_exists = True
        model_info = {
            "name": raw_path_str,
            "path": raw_path_str,
            "exists": True,
            "version_label": f"{raw_path_str} standard model alias",
        }
    else:
        model_path = Path(raw_path_str).expanduser()
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
        model_exists = model_path.exists()
        model_path_str = str(model_path) if model_exists else "yolov8n.pt"
        model_info = build_model_info(model_path if model_exists else Path(model_path_str), model_exists)

    return {
        "default_model_path": model_path_str,
        "model_exists": model_exists or is_ultralytics_model_alias(model_path_str),
        "model": model_info,
        "settings": settings,
        "supported_video_formats": sorted(ALLOWED_VIDEO_EXTENSIONS),
        "supported_image_formats": sorted(ALLOWED_IMAGE_EXTENSIONS),
        "pipeline": [
            "Upload Video/Image",
            "Extract Frames",
            "YOLO",
            "Show Defects",
            "Generate Report",
        ],
    }


@app.get("/api/system/status")
def system_status(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    database_ok = False
    try:
        db.execute(select(1))
        database_ok = True
    except Exception:
        logger.exception("Database status check failed")

    settings = load_local_settings()
    model_ref = settings["default_model_path"]
    model_path = Path(model_ref).expanduser()
    if not model_path.is_absolute() and not is_ultralytics_model_alias(model_ref):
        model_path = PROJECT_ROOT / model_path
    model_exists = True if is_ultralytics_model_alias(model_ref) else model_path.exists()
    model_version = (
        {
            "name": model_ref,
            "path": model_ref,
            "exists": True,
            "version_label": f"{model_ref} standard model alias",
        }
        if is_ultralytics_model_alias(model_ref)
        else build_model_info(model_path, model_exists)
    )
    usage = shutil.disk_usage(STORAGE_ROOT)

    return {
        "app": {"name": app.title, "version": app.version},
        "user": {"email": current_user.email, "role": current_user.role},
        "database": {"status": "connected" if database_ok else "unavailable"},
        "model": {
            "path": model_ref,
            "exists": model_exists,
            "version": model_version,
        },
        "storage": {
            "root": str(STORAGE_ROOT),
            "exists": STORAGE_ROOT.exists(),
            "free_mb": round(usage.free / (1024 * 1024), 1),
            "total_mb": round(usage.total / (1024 * 1024), 1),
            "logs": str(LOG_ROOT),
        },
        "uploads": {
            "max_video_mb": settings["max_video_upload_mb"],
            "max_image_mb": settings["max_image_upload_mb"],
            "video_formats": sorted(ALLOWED_VIDEO_EXTENSIONS),
            "image_formats": sorted(ALLOWED_IMAGE_EXTENSIONS),
        },
        "security": {"allowed_origins": get_allowed_origins()},
    }


@app.get("/api/settings")
def get_settings(
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    return {
        "settings": load_local_settings(),
        "editable": current_user.role == UserRole.ADMIN.value,
    }


@app.put("/api/settings")
def update_settings(
    payload: SettingsUpdate,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    settings = save_local_settings(payload)
    record_audit_event(
        db,
        user_id=current_user.id,
        action="settings_updated",
        entity_type="settings",
        entity_id="local",
        message="Application settings updated.",
        metadata=settings,
    )
    db.commit()
    logger.info("Application settings updated by user_id=%s", current_user.id)
    return {"settings": settings}


@app.get("/api/system/backup.zip")
def download_system_backup(
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> FileResponse:
    backup_path = generate_local_backup_package()
    record_audit_event(
        db,
        user_id=current_user.id,
        action="system_backup_downloaded",
        entity_type="system_backup",
        entity_id=backup_path.name,
        message="Local system backup package downloaded.",
        metadata={"path": str(backup_path)},
    )
    db.commit()
    return FileResponse(
        path=str(backup_path),
        media_type="application/zip",
        filename=backup_path.name,
    )


@app.post("/api/inspections")
async def create_inspection(
    background_tasks: BackgroundTasks,
    video: UploadFile | None = File(None),
    image: UploadFile | None = File(None),
    sample_fps: float = Form(1.0),
    confidence: float = Form(0.35),
    device: str = Form("0"),
    model_path: str | None = Form(None),
    organization_name: str | None = Form(None),
    project_name: str | None = Form(None),
    report_title: str | None = Form(None),
    inspector_name: str | None = Form(None),
    building_name: str | None = Form(None),
    floor_level: str | None = Form(None),
    facade_side: str | None = Form(None),
    location: str | None = Form(None),
    inspection_date: str | None = Form(None),
    remarks: str | None = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    upload = image or video
    media_type = "image" if image is not None else "video"
    if upload is None:
        logger.info("Inspection request rejected: no upload provided")
        raise HTTPException(status_code=400, detail="Upload either a video or an image.")
    if image is not None and video is not None:
        logger.info("Inspection request rejected: both video and image provided")
        raise HTTPException(status_code=400, detail="Upload only one file type: video or image.")

    extension = Path(upload.filename or "").suffix.lower()
    allowed_extensions = ALLOWED_IMAGE_EXTENSIONS if media_type == "image" else ALLOWED_VIDEO_EXTENSIONS
    allowed_mime_types = ALLOWED_IMAGE_MIME_TYPES if media_type == "image" else ALLOWED_VIDEO_MIME_TYPES
    upload_limits = upload_size_limits_bytes()
    max_upload_bytes = upload_limits["image"] if media_type == "image" else upload_limits["video"]
    if extension not in allowed_extensions:
        supported = ", ".join(sorted(allowed_extensions))
        logger.info("Inspection request rejected: unsupported %s extension %s", media_type, extension)
        raise HTTPException(status_code=400, detail=f"Unsupported {media_type} format. Supported formats: {supported}.")
    if upload.content_type and upload.content_type not in allowed_mime_types:
        supported = ", ".join(sorted(allowed_mime_types - {"application/octet-stream"}))
        logger.info("Inspection request rejected: unsupported MIME type %s", upload.content_type)
        raise HTTPException(status_code=400, detail=f"Unsupported {media_type} content type. Expected: {supported}.")
    if sample_fps <= 0:
        raise HTTPException(status_code=400, detail="sample_fps must be greater than zero.")
    if confidence <= 0 or confidence >= 1:
        raise HTTPException(status_code=400, detail="confidence must be between 0 and 1.")

    inspection_id = create_inspection_id()
    upload_dir = UPLOAD_ROOT / inspection_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = upload_dir / f"input{extension}"

    try:
        uploaded_bytes = save_upload_with_size_limit(upload, upload_path, max_upload_bytes)
    finally:
        await upload.close()

    settings = load_local_settings()
    selected_model_ref = (model_path or "").strip() or settings["default_model_path"]
    if is_ultralytics_model_alias(selected_model_ref):
        selected_model_path = selected_model_ref
    else:
        selected_model_path_obj = Path(selected_model_ref).expanduser()
        if not selected_model_path_obj.is_absolute():
            selected_model_path_obj = PROJECT_ROOT / selected_model_path_obj
        if not selected_model_path_obj.exists():
            selected_model_path = "yolov8n.pt"
        else:
            selected_model_path = str(selected_model_path_obj)


    job = InspectionJob(
        job_id=inspection_id,
        user_id=current_user.id,
        status=JobStatus.QUEUED.value,
        progress_percent=0,
        model_path=selected_model_path,
        confidence_threshold=confidence,
        sample_fps=sample_fps,
        device=device,
        metadata_json={
            "archived": False,
            "inspection_metadata": compact_metadata(
                {
                    "organization_name": organization_name,
                    "project_name": project_name,
                    "report_title": report_title or "Facade Glass Inspection Report",
                    "inspector_name": inspector_name or current_user.full_name or current_user.email,
                    "building_name": building_name,
                    "floor_level": floor_level,
                    "facade_side": facade_side,
                    "location": location,
                    "inspection_date": inspection_date,
                    "remarks": remarks,
                }
            ),
        },
    )
    db.add(job)
    db.flush()

    db.add(
        UploadedVideo(
            inspection_job_id=job.id,
            original_filename=upload.filename or upload_path.name,
            stored_path=str(upload_path),
            content_type=upload.content_type,
            file_size_bytes=uploaded_bytes,
        )
    )
    record_audit_event(
        db,
        user_id=current_user.id,
        action=AuditAction.UPLOAD.value,
        entity_type=media_type,
        entity_id=inspection_id,
        message=f"{media_type.title()} uploaded for inspection.",
        metadata={
            "filename": upload.filename or upload_path.name,
            "content_type": upload.content_type,
            "file_size_bytes": uploaded_bytes,
        },
    )
    record_audit_event(
        db,
        user_id=current_user.id,
        action=AuditAction.INSPECTION_CREATED.value,
        entity_type="inspection_job",
        entity_id=inspection_id,
        message="Inspection job created.",
        metadata={"media_type": media_type, "confidence": confidence, "sample_fps": sample_fps},
    )
    db.commit()

    background_tasks.add_task(
        process_inspection_job,
        job_id=inspection_id,
        video_path=str(upload_path),
        model_path=selected_model_path,
        sample_fps=sample_fps,
        confidence=confidence,
        device=device,
        media_type=media_type,
    )
    logger.info("Queued inspection job %s for user_id=%s media_type=%s", inspection_id, current_user.id, media_type)

    return {
        "job_id": inspection_id,
        "status": JobStatus.QUEUED.value,
        "progress_percent": 0,
        "status_url": f"/api/jobs/{inspection_id}",
        "result_url": f"/api/jobs/{inspection_id}/result",
    }


@app.get("/api/inspections/history")
def get_inspection_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    jobs = db.scalars(
        select(InspectionJob)
        .where(InspectionJob.user_id == current_user.id)
        .order_by(InspectionJob.created_at.desc(), InspectionJob.id.desc())
    ).all()

    history_items = []
    for job in jobs:
        uploaded_video = db.scalar(
            select(UploadedVideo).where(UploadedVideo.inspection_job_id == job.id)
        )
        total_detections = db.scalar(
            select(func.count(DefectDetection.id)).where(DefectDetection.inspection_job_id == job.id)
        )
        estimated_unique_detections = None
        critical_defects = None
        high_defects = None
        report_reliability = None
        if job.status == JobStatus.COMPLETED.value:
            job_summary = build_job_summary(db, job)
            estimated_unique_detections = job_summary["estimated_unique_detections"]
            critical_defects = job_summary["critical_defects"]
            high_defects = job_summary["high_defects"]
            report_reliability = job_summary["report_reliability"]
        reports = db.scalars(select(Report).where(Report.inspection_job_id == job.id)).all()
        report_urls = {
            f"{report.report_type}_url": artifact_url(report.file_path)
            for report in reports
            if Path(report.file_path).exists()
        }

        history_items.append(
            {
                "job_id": job.job_id,
                "video_filename": uploaded_video.original_filename if uploaded_video else "-",
                "status": job.status,
                "archived": bool((job.metadata_json or {}).get("archived")),
                "archived_at": (job.metadata_json or {}).get("archived_at"),
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "total_detections": total_detections or 0,
                "estimated_unique_detections": estimated_unique_detections,
                "defect_areas_needing_attention": estimated_unique_detections,
                "critical_defects": critical_defects,
                "high_defects": high_defects,
                "report_reliability": report_reliability,
                "progress_percent": job.progress_percent,
                "result_url": f"/api/jobs/{job.job_id}/result"
                if job.status == JobStatus.COMPLETED.value
                else None,
                "pdf_report_url": f"/api/jobs/{job.job_id}/report/pdf"
                if job.status == JobStatus.COMPLETED.value
                else None,
                "report": report_urls,
            }
        )

    return {"items": history_items}


@app.get("/api/inspections/history.csv")
def export_inspection_history_csv(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    jobs = db.scalars(
        select(InspectionJob)
        .where(InspectionJob.user_id == current_user.id)
        .order_by(InspectionJob.created_at.desc(), InspectionJob.id.desc())
    ).all()
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "job_id",
            "media_filename",
            "status",
            "archived",
            "created_at",
            "completed_at",
            "total_detections",
            "attention_areas",
            "critical_defects",
            "high_priority_defects",
            "report_reliability",
        ],
    )
    writer.writeheader()

    for job in jobs:
        uploaded_video = db.scalar(select(UploadedVideo).where(UploadedVideo.inspection_job_id == job.id))
        total_detections = db.scalar(
            select(func.count(DefectDetection.id)).where(DefectDetection.inspection_job_id == job.id)
        ) or 0
        summary = build_job_summary(db, job) if job.status == JobStatus.COMPLETED.value else {}
        writer.writerow(
            {
                "job_id": job.job_id,
                "media_filename": uploaded_video.original_filename if uploaded_video else "",
                "status": job.status,
                "archived": bool((job.metadata_json or {}).get("archived")),
                "created_at": job.created_at.isoformat() if job.created_at else "",
                "completed_at": job.completed_at.isoformat() if job.completed_at else "",
                "total_detections": total_detections,
                "attention_areas": summary.get("estimated_unique_detections", ""),
                "critical_defects": summary.get("critical_defects", ""),
                "high_priority_defects": summary.get("high_defects", ""),
                "report_reliability": summary.get("report_reliability", ""),
            }
        )

    record_audit_event(
        db,
        user_id=current_user.id,
        action="history_csv_exported",
        entity_type="inspection_history",
        entity_id=str(current_user.id),
        message="Inspection history CSV exported.",
        metadata={"row_count": len(jobs)},
    )
    db.commit()

    filename = f"skyvision_inspection_history_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.patch("/api/inspections/{job_id}/archive")
def archive_inspection(
    job_id: str,
    payload: ArchiveRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    job = get_user_job_or_404(db, job_id, current_user)
    if job.status not in {JobStatus.COMPLETED.value, JobStatus.FAILED.value}:
        raise HTTPException(status_code=409, detail="Only completed or failed inspections can be archived.")

    metadata = dict(job.metadata_json or {})
    metadata["archived"] = payload.archived
    metadata["archived_at"] = datetime.now(timezone.utc).isoformat() if payload.archived else None
    job.metadata_json = metadata
    record_audit_event(
        db,
        user_id=current_user.id,
        action="inspection_archived" if payload.archived else "inspection_restored",
        entity_type="inspection_job",
        entity_id=job.job_id,
        message="Inspection archived." if payload.archived else "Inspection restored.",
        metadata={"archived": payload.archived},
    )
    db.commit()
    return {"job_id": job.job_id, "archived": payload.archived}


@app.get("/api/jobs/{job_id}")
def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    job = get_user_job_or_404(db, job_id, current_user)
    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress_percent": job.progress_percent,
        "error_message": job.error_message,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }


@app.get("/api/jobs/{job_id}/result")
def get_job_result(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    job = get_user_job_or_404(db, job_id, current_user)
    if job.status != JobStatus.COMPLETED.value:
        raise HTTPException(status_code=409, detail=f"Job is not completed. Current status: {job.status}")

    return to_job_result_payload(db, job)


@app.get("/api/jobs/{job_id}/report/pdf")
def download_job_pdf_report(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> FileResponse:
    job = get_user_job_or_404(db, job_id, current_user)
    if job.status != JobStatus.COMPLETED.value:
        raise HTTPException(status_code=409, detail=f"Job is not completed. Current status: {job.status}")

    report = db.scalar(
        select(Report).where(
            Report.inspection_job_id == job.id,
            Report.report_type == ReportType.PDF.value,
        )
    )

    pdf_path = generate_pdf_from_job_records(db, job)
    if report is None:
        report = Report(
            inspection_job_id=job.id,
            report_type=ReportType.PDF.value,
            file_path=pdf_path,
            summary_json=build_job_summary(db, job),
        )
        db.add(report)
    else:
        report.file_path = pdf_path
        report.summary_json = build_job_summary(db, job)
    db.commit()
    record_audit_event(
        db,
        user_id=current_user.id,
        action=AuditAction.REPORT_DOWNLOAD.value,
        entity_type="inspection_job",
        entity_id=job.job_id,
        message="PDF report downloaded.",
        metadata={"report_type": ReportType.PDF.value},
    )
    db.commit()

    return FileResponse(
        path=report.file_path,
        media_type="application/pdf",
        filename=f"skyvision_{job.job_id}_inspection_report.pdf",
    )


@app.get("/api/jobs/{job_id}/report/package.zip")
def download_job_report_package(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> FileResponse:
    job = get_user_job_or_404(db, job_id, current_user)
    if job.status != JobStatus.COMPLETED.value:
        raise HTTPException(status_code=409, detail=f"Job is not completed. Current status: {job.status}")

    package_path = generate_report_package_from_job_records(db, job)
    record_audit_event(
        db,
        user_id=current_user.id,
        action=AuditAction.REPORT_DOWNLOAD.value,
        entity_type="inspection_job",
        entity_id=job.job_id,
        message="ZIP report package downloaded.",
        metadata={"report_type": "zip"},
    )
    db.commit()
    return FileResponse(
        path=str(package_path),
        media_type="application/zip",
        filename=f"skyvision_{job.job_id}_report_package.zip",
    )


def get_user_job_or_404(db: Session, job_id: str, current_user: User) -> InspectionJob:
    job = db.scalar(
        select(InspectionJob).where(
            InspectionJob.job_id == job_id,
            InspectionJob.user_id == current_user.id,
        )
    )
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


def to_job_result_payload(db: Session, job: InspectionJob) -> dict[str, Any]:
    frames = db.scalars(
        select(ExtractedFrame)
        .where(ExtractedFrame.inspection_job_id == job.id)
        .order_by(ExtractedFrame.frame_index)
    ).all()
    detections = db.scalars(
        select(DefectDetection).where(DefectDetection.inspection_job_id == job.id)
    ).all()
    reports = db.scalars(select(Report).where(Report.inspection_job_id == job.id)).all()

    detections_by_frame: dict[int, list[DefectDetection]] = {}
    for detection in detections:
        detections_by_frame.setdefault(detection.frame_id, []).append(detection)

    detections_by_class: dict[str, int] = {}
    for detection in detections:
        detections_by_class[detection.class_name] = detections_by_class.get(detection.class_name, 0) + 1
    deduplication_summary = estimate_unique_detections(
        [
            {
                "class_name": detection.class_name,
                "confidence": detection.confidence,
                "bbox_xyxy": detection.bbox_xyxy,
                "frame_width": frame.width,
                "frame_height": frame.height,
                "frame_index": frame.frame_index,
                "frame_sequence_index": sequence_index,
                "timestamp_seconds": frame.timestamp_seconds,
                "image_path": frame.image_path,
                "annotated_image_path": frame.annotated_image_path,
            }
            for sequence_index, frame in enumerate(frames)
            for detection in detections_by_frame.get(frame.id, [])
        ]
    )

    report_urls = {
        f"{report.report_type}_url": artifact_url(report.file_path)
        for report in reports
        if Path(report.file_path).exists()
    }

    return {
        "inspection_id": job.job_id,
        "job_id": job.job_id,
        "status": job.status,
        "summary": {
            "total_frames_sampled": len(frames),
            "total_detections": len(detections),
            **deduplication_summary,
            "detections_by_class": detections_by_class,
            "video_metadata": job.metadata_json or {},
        },
        "frames": [
            {
                "image_url": artifact_url(frame.image_path),
                "annotated_image_url": artifact_url(frame.annotated_image_path or frame.image_path),
                "heatmap_image_url": artifact_url(heatmap_path_for_frame(frame)),
                "detections": [
                    {
                        "class_id": detection.class_id,
                        "class_name": detection.class_name,
                        "confidence": detection.confidence,
                        "bbox_xyxy": detection.bbox_xyxy,
                    }
                    for detection in detections_by_frame.get(frame.id, [])
                ],
            }
            for frame in frames
        ],
        "report": report_urls,
    }


def generate_pdf_from_job_records(db: Session, job: InspectionJob) -> str:
    reports_dir = STORAGE_ROOT / "inspections" / job.job_id / "reports"
    frames = db.scalars(
        select(ExtractedFrame)
        .where(ExtractedFrame.inspection_job_id == job.id)
        .order_by(ExtractedFrame.frame_index)
    ).all()
    detections = db.scalars(
        select(DefectDetection).where(DefectDetection.inspection_job_id == job.id)
    ).all()

    detections_by_frame: dict[int, list[DefectDetection]] = {}
    detections_by_class: dict[str, int] = {}
    for detection in detections:
        detections_by_frame.setdefault(detection.frame_id, []).append(detection)
        detections_by_class[detection.class_name] = detections_by_class.get(detection.class_name, 0) + 1

    frame_payload = [
        {
            "image_path": frame.image_path,
            "annotated_image_path": frame.annotated_image_path,
            "heatmap_image_path": str(heatmap_path_for_frame(frame, as_path=True)),
            "frame_index": frame.frame_index,
            "frame_sequence_index": sequence_index,
            "timestamp_seconds": frame.timestamp_seconds,
            "frame_width": frame.width,
            "frame_height": frame.height,
            "detections": [
                {
                    "class_id": detection.class_id,
                    "class_name": detection.class_name,
                    "confidence": detection.confidence,
                    "bbox_xyxy": detection.bbox_xyxy,
                }
                for detection in detections_by_frame.get(frame.id, [])
            ],
        }
        for sequence_index, frame in enumerate(frames)
    ]

    return generate_pdf_report(
        job_id=job.job_id,
        video_metadata=job.metadata_json or {},
        frames=frame_payload,
        detections_by_class=detections_by_class,
        output_dir=reports_dir,
    )


def generate_report_package_from_job_records(db: Session, job: InspectionJob) -> Path:
    reports_dir = STORAGE_ROOT / "inspections" / job.job_id / "reports"
    annotated_dir = STORAGE_ROOT / "inspections" / job.job_id / "annotated"
    reports_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = Path(generate_pdf_from_job_records(db, job))
    package_path = reports_dir / f"skyvision_{job.job_id}_report_package.zip"

    reports = db.scalars(select(Report).where(Report.inspection_job_id == job.id)).all()
    candidate_files = {pdf_path}
    for report in reports:
        report_path = Path(report.file_path)
        if report_path.exists():
            candidate_files.add(report_path)

    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(candidate_files):
            if file_path.exists():
                archive.write(file_path, arcname=f"reports/{file_path.name}")

        if annotated_dir.exists():
            for artifact_path in sorted(annotated_dir.glob("*")):
                if artifact_path.is_file() and artifact_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    archive.write(artifact_path, arcname=f"annotated/{artifact_path.name}")

    logger.info("Generated report package for job %s at %s", job.job_id, package_path)
    return package_path


def generate_local_backup_package() -> Path:
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    backup_path = BACKUP_ROOT / f"skyvision_backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.zip"
    include_files: list[Path] = [
        PROJECT_ROOT / "database" / "skyvision.db",
        PROJECT_ROOT / "database" / "schema.sql",
        PROJECT_ROOT / ".env.example",
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "PROJECT_STATUS.md",
        SETTINGS_PATH,
    ]
    include_files.extend(sorted((PROJECT_ROOT / "database" / "schemas").glob("*.yaml")))
    include_files.extend(sorted((PROJECT_ROOT / "docs").glob("*.md")))
    include_files.extend(sorted(LOG_ROOT.glob("*.log*")))

    with zipfile.ZipFile(backup_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in include_files:
            if file_path.exists() and file_path.is_file():
                archive.write(file_path, arcname=file_path.relative_to(PROJECT_ROOT).as_posix())

        inspections_root = STORAGE_ROOT / "inspections"
        if inspections_root.exists():
            for report_path in sorted(inspections_root.glob("*/reports/*")):
                if report_path.is_file():
                    archive.write(report_path, arcname=report_path.relative_to(PROJECT_ROOT).as_posix())

    logger.info("Generated local backup package at %s", backup_path)
    return backup_path


def build_job_summary(db: Session, job: InspectionJob) -> dict[str, Any]:
    frames_count = db.scalar(
        select(func.count(ExtractedFrame.id)).where(ExtractedFrame.inspection_job_id == job.id)
    )
    detections = db.scalars(select(DefectDetection).where(DefectDetection.inspection_job_id == job.id)).all()
    frames = db.scalars(select(ExtractedFrame).where(ExtractedFrame.inspection_job_id == job.id)).all()
    detections_by_frame: dict[int, list[DefectDetection]] = {}
    for detection in detections:
        detections_by_frame.setdefault(detection.frame_id, []).append(detection)
    detections_by_class: dict[str, int] = {}
    for detection in detections:
        detections_by_class[detection.class_name] = detections_by_class.get(detection.class_name, 0) + 1
    deduplication_summary = estimate_unique_detections(
        [
            {
                "class_name": detection.class_name,
                "confidence": detection.confidence,
                "bbox_xyxy": detection.bbox_xyxy,
                "frame_width": frame.width,
                "frame_height": frame.height,
                "frame_index": frame.frame_index,
                "frame_sequence_index": sequence_index,
                "timestamp_seconds": frame.timestamp_seconds,
                "image_path": frame.image_path,
                "annotated_image_path": frame.annotated_image_path,
            }
            for sequence_index, frame in enumerate(frames)
            for detection in detections_by_frame.get(frame.id, [])
        ]
    )

    return {
        "total_frames_sampled": frames_count or 0,
        "total_detections": len(detections),
        **deduplication_summary,
        "detections_by_class": detections_by_class,
        "video_metadata": job.metadata_json or {},
    }


def heatmap_path_for_frame(frame: ExtractedFrame, as_path: bool = False) -> str | Path:
    source_path = Path(frame.annotated_image_path or frame.image_path)
    heatmap_path = source_path.with_name(f"heatmap_{source_path.name}")
    if heatmap_path.exists():
        return heatmap_path if as_path else str(heatmap_path)
    fallback = Path(frame.annotated_image_path or frame.image_path)
    return fallback if as_path else str(fallback)


def to_api_payload(result: dict[str, Any]) -> dict[str, Any]:
    report = result["report"]
    frames = report["frames"]
    return {
        "inspection_id": result["inspection_id"],
        "summary": {
            "total_frames_sampled": report["total_frames_sampled"],
            "total_detections": report["total_detections"],
            "detections_by_class": report["detections_by_class"],
            "video_metadata": report["video_metadata"],
        },
        "frames": [
            {
                "image_url": artifact_url(frame["image_path"]),
                "annotated_image_url": artifact_url(frame["annotated_image_path"]),
                "heatmap_image_url": artifact_url(frame.get("heatmap_image_path") or frame["annotated_image_path"]),
                "detections": frame["detections"],
            }
            for frame in frames
        ],
        "report": {
            "json_url": artifact_url(report["json_report_path"]),
            "html_url": artifact_url(report["html_report_path"]),
        },
    }


def artifact_url(path_value: str) -> str:
    path = Path(path_value).expanduser().resolve()
    relative_path = path.relative_to(STORAGE_ROOT.resolve()).as_posix()
    return f"/artifacts/{relative_path}"


def build_model_info(model_path: Path, exists: bool) -> dict[str, Any]:
    if not exists:
        return {
            "name": model_path.name,
            "path": str(model_path),
            "exists": False,
            "version_label": "yolov8n.pt fallback",
        }

    stat = model_path.stat()
    updated_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    size_mb = stat.st_size / (1024 * 1024)
    return {
        "name": model_path.name,
        "path": str(model_path),
        "exists": True,
        "size_bytes": stat.st_size,
        "updated_at": updated_at.isoformat(),
        "version_label": f"{model_path.name} | {size_mb:.1f} MB | {updated_at.strftime('%Y-%m-%d %H:%M UTC')}",
    }


def compact_metadata(values: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.strip() if isinstance(value, str) else value
        for key, value in values.items()
        if value is not None and (not isinstance(value, str) or value.strip())
    }


def save_upload_with_size_limit(upload: UploadFile, destination: Path, max_bytes: int) -> int:
    uploaded_bytes = 0
    with destination.open("wb") as output_file:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            uploaded_bytes += len(chunk)
            if uploaded_bytes > max_bytes:
                output_file.close()
                destination.unlink(missing_ok=True)
                max_mb = max_bytes / (1024 * 1024)
                raise HTTPException(status_code=413, detail=f"Upload is too large. Maximum allowed size is {max_mb:.0f} MB.")
            output_file.write(chunk)
    return uploaded_bytes


def ensure_default_admin_role() -> None:
    db = SessionLocal()
    try:
        admin_exists = db.scalar(select(func.count(User.id)).where(User.role == UserRole.ADMIN.value)) or 0
        if admin_exists:
            return
        first_user = db.scalar(select(User).order_by(User.id.asc()))
        if first_user is None:
            return
        first_user.role = UserRole.ADMIN.value
        record_audit_event(
            db,
            user_id=first_user.id,
            action=AuditAction.USER_ROLE_UPDATED.value,
            entity_type="user",
            entity_id=str(first_user.id),
            message="First local user promoted to administrator.",
            metadata={"email": first_user.email, "role": first_user.role},
        )
        db.commit()
        logger.info("Promoted first user to admin for local administration: %s", first_user.email)
    finally:
        db.close()
