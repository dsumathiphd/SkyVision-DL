"""FastAPI MVP for SkyVision-DL video inspections."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.services.inspection_pipeline import (
    DEFAULT_MODEL_PATH,
    STORAGE_ROOT,
    create_inspection_id,
    run_video_inspection,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_ROOT = PROJECT_ROOT / "frontend"
UPLOAD_ROOT = STORAGE_ROOT / "uploads"
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi"}


app = FastAPI(title="SkyVision-DL MVP", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

app.mount("/artifacts", StaticFiles(directory=str(STORAGE_ROOT)), name="artifacts")
if FRONTEND_ROOT.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_ROOT)), name="frontend")


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
    model_path = Path(os.environ.get("SKYVISION_MODEL_PATH", str(DEFAULT_MODEL_PATH))).expanduser()
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    return {
        "default_model_path": str(model_path),
        "model_exists": model_path.exists(),
        "supported_video_formats": sorted(ALLOWED_VIDEO_EXTENSIONS),
        "pipeline": [
            "Upload Video",
            "Extract Frames",
            "YOLO",
            "Show Defects",
            "Generate Report",
        ],
    }


@app.post("/api/inspections")
async def create_inspection(
    video: UploadFile = File(...),
    sample_fps: float = Form(1.0),
    confidence: float = Form(0.25),
    device: str = Form("0"),
    model_path: str | None = Form(None),
) -> dict[str, Any]:
    extension = Path(video.filename or "").suffix.lower()
    if extension not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only MP4 and AVI uploads are supported.")
    if sample_fps <= 0:
        raise HTTPException(status_code=400, detail="sample_fps must be greater than zero.")
    if confidence <= 0 or confidence >= 1:
        raise HTTPException(status_code=400, detail="confidence must be between 0 and 1.")

    inspection_id = create_inspection_id()
    upload_dir = UPLOAD_ROOT / inspection_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = upload_dir / f"input{extension}"

    try:
        with upload_path.open("wb") as output_file:
            shutil.copyfileobj(video.file, output_file)
    finally:
        await video.close()

    selected_model_path = Path(
        model_path
        or os.environ.get("SKYVISION_MODEL_PATH", str(DEFAULT_MODEL_PATH))
    ).expanduser()
    if not selected_model_path.is_absolute():
        selected_model_path = PROJECT_ROOT / selected_model_path

    try:
        result = run_video_inspection(
            video_path=upload_path,
            model_path=selected_model_path,
            sample_fps=sample_fps,
            confidence=confidence,
            device=device,
            inspection_id=inspection_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return to_api_payload(result.to_dict())


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
