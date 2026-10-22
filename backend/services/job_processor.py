"""Background inspection job processing for SkyVision-DL."""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
from sqlalchemy import select

from backend.database.connection import SessionLocal
from backend.models.inspection_models import (
    DefectDetection,
    ExtractedFrame,
    InspectionJob,
    JobStatus,
    Report,
    ReportType,
    UploadedVideo,
)
from backend.reports.report_generator import generate_report
from backend.reports.pdf_report_generator import generate_pdf_report
from backend.video.frame_extractor import ExtractedFrame as ExtractedFrameData
from backend.video.frame_extractor import VideoMetadata, extract_frames
from backend.yolo.detector import YOLOv8Detector, is_ultralytics_model_alias


logger = logging.getLogger(__name__)


def process_inspection_job(
    job_id: str,
    video_path: str,
    model_path: str,
    sample_fps: float,
    confidence: float,
    device: str,
    media_type: str = "video",
) -> None:
    """Run frame extraction, YOLO inference, report generation, and persistence."""

    db = SessionLocal()
    try:
        job = db.scalar(select(InspectionJob).where(InspectionJob.job_id == job_id))
        if job is None:
            return

        job.status = JobStatus.PROCESSING.value
        job.progress_percent = 10
        job.started_at = datetime.now(timezone.utc)
        db.commit()
        logger.info("Started inspection job %s with media_type=%s", job_id, media_type)

        inspection_root = Path(video_path).resolve().parents[1] / ".." / "inspections" / job_id
        inspection_root = inspection_root.resolve()
        frames_dir = inspection_root / "frames"
        annotated_dir = inspection_root / "annotated"
        reports_dir = inspection_root / "reports"

        detector = YOLOv8Detector(model_path=model_path, confidence=confidence, device=device)
        try:
            detector.load()
        except FileNotFoundError as exc:
            if not is_ultralytics_model_alias(model_path):
                fallback_model = "yolov8n.pt"
                detector = YOLOv8Detector(model_path=fallback_model, confidence=confidence, device=device)
                detector.load()
                job.model_path = fallback_model
                metadata_dict = dict(job.metadata_json or {})
                metadata_dict["warning"] = (
                    f"Requested model '{model_path}' not found. "
                    f"Fell back to standard alias '{fallback_model}'."
                )
                job.metadata_json = metadata_dict
                db.commit()
            else:
                raise exc


        if media_type == "image":
            metadata, extracted_frames = prepare_single_image_frame(
                image_path=video_path,
                output_dir=frames_dir,
            )
        else:
            metadata, extracted_frames = extract_frames(
                video_path=video_path,
                output_dir=frames_dir,
                sample_fps=sample_fps,
                image_extension=".jpg",
            )

        uploaded_video = db.scalar(select(UploadedVideo).where(UploadedVideo.inspection_job_id == job.id))
        if uploaded_video is not None:
            uploaded_video.fps = metadata.fps
            uploaded_video.frame_count = metadata.frame_count
            uploaded_video.width = metadata.width
            uploaded_video.height = metadata.height
            uploaded_video.duration_seconds = metadata.duration_seconds

        existing_metadata = dict(job.metadata_json or {})
        job.metadata_json = {
            **metadata.to_dict(),
            "archived": existing_metadata.get("archived", False),
            "archived_at": existing_metadata.get("archived_at"),
            "inspection_metadata": existing_metadata.get("inspection_metadata", {}),
            "warning": existing_metadata.get("warning"),
        }
        job.progress_percent = 35
        db.commit()

        frame_rows: dict[int, ExtractedFrame] = {}
        for frame in extracted_frames:
            image = cv2.imread(frame.output_path)
            height, width = image.shape[:2] if image is not None else (None, None)
            frame_row = ExtractedFrame(
                inspection_job_id=job.id,
                frame_index=frame.frame_index,
                timestamp_seconds=frame.timestamp_seconds,
                image_path=frame.output_path,
                width=width,
                height=height,
            )
            db.add(frame_row)
            frame_rows[frame.frame_index] = frame_row

        db.flush()

        frame_results = []
        total_frames = max(len(extracted_frames), 1)
        for index, frame in enumerate(extracted_frames, start=1):
            frame_result = detector.predict_frame(frame.output_path, annotated_output_dir=annotated_dir)
            frame_results.append(frame_result)

            frame_row = frame_rows[frame.frame_index]
            frame_row.annotated_image_path = frame_result.annotated_image_path

            for detection in frame_result.detections:
                db.add(
                    DefectDetection(
                        inspection_job_id=job.id,
                        frame_id=frame_row.id,
                        class_id=detection.class_id,
                        class_name=detection.class_name,
                        confidence=detection.confidence,
                        bbox_xyxy=list(detection.bbox_xyxy),
                    )
                )

            job.progress_percent = min(90, 35 + round((index / total_frames) * 55))
            db.commit()

        report = generate_report(
            inspection_id=job_id,
            video_metadata=metadata,
            frame_results=frame_results,
            output_dir=reports_dir,
            extra_metadata=job.metadata_json or {},
        )
        pdf_path = generate_pdf_report(
            job_id=job_id,
            video_metadata=job.metadata_json or metadata.to_dict(),
            frames=report.frames,
            detections_by_class=report.detections_by_class,
            output_dir=reports_dir,
        )

        db.add(
            Report(
                inspection_job_id=job.id,
                report_type=ReportType.JSON.value,
                file_path=report.json_report_path,
                summary_json=_report_summary(report.to_dict()),
            )
        )
        db.add(
            Report(
                inspection_job_id=job.id,
                report_type=ReportType.HTML.value,
                file_path=report.html_report_path,
                summary_json=_report_summary(report.to_dict()),
            )
        )
        db.add(
            Report(
                inspection_job_id=job.id,
                report_type=ReportType.PDF.value,
                file_path=pdf_path,
                summary_json=_report_summary(report.to_dict()),
            )
        )

        job.status = JobStatus.COMPLETED.value
        job.progress_percent = 100
        job.completed_at = datetime.now(timezone.utc)
        db.commit()
        logger.info("Completed inspection job %s", job_id)
    except Exception as exc:
        logger.exception("Inspection job %s failed", job_id)
        db.rollback()
        failed_job = db.scalar(select(InspectionJob).where(InspectionJob.job_id == job_id))
        if failed_job is not None:
            failed_job.status = JobStatus.FAILED.value
            failed_job.error_message = str(exc)
            failed_job.completed_at = datetime.now(timezone.utc)
            db.commit()
    finally:
        db.close()


def _report_summary(report: dict[str, Any]) -> dict[str, Any]:
    detections_by_class = report.get("detections_by_class", {})
    if not isinstance(detections_by_class, dict):
        detections_by_class = dict(Counter())

    return {
        "total_frames_sampled": report.get("total_frames_sampled", 0),
        "total_detections": report.get("total_detections", 0),
        "defect_areas_needing_attention": report.get("defect_areas_needing_attention", 0),
        "critical_defects": report.get("critical_defects", 0),
        "high_defects": report.get("high_defects", 0),
        "medium_defects": report.get("medium_defects", 0),
        "low_defects": report.get("low_defects", 0),
        "report_reliability": report.get("report_reliability"),
        "maintenance_priority_list": report.get("maintenance_priority_list", []),
        "raw_detection_count": report.get("raw_detection_count", report.get("total_detections", 0)),
        "repeated_frame_detections": report.get("repeated_frame_detections", 0),
        "detections_by_class": detections_by_class,
        "video_metadata": report.get("video_metadata", {}),
    }


def prepare_single_image_frame(
    image_path: str | Path,
    output_dir: str | Path,
) -> tuple[VideoMetadata, list[ExtractedFrameData]]:
    """Normalize an uploaded image into a single-frame inspection input."""

    source = Path(image_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Image file not found: {source}")

    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f"Unable to read uploaded image file: {source}")

    height, width = image.shape[:2]
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    frame_path = destination / f"{source.stem}_frame_00000000.jpg"
    if not cv2.imwrite(str(frame_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise ValueError(f"Failed to write normalized image frame: {frame_path}")

    metadata = VideoMetadata(
        path=str(source),
        file_name=source.name,
        extension=source.suffix.lower(),
        fps=0.0,
        frame_count=1,
        width=width,
        height=height,
        duration_seconds=0.0,
        media_type="image",
    )
    return metadata, [
        ExtractedFrameData(
            frame_index=0,
            timestamp_seconds=0.0,
            output_path=str(frame_path),
        )
    ]
