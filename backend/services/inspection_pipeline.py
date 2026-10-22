"""Inspection pipeline: upload media -> frames -> YOLO -> report."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

from backend.reports.report_generator import InspectionReport, generate_report
from backend.video.frame_extractor import extract_frames
from backend.yolo.detector import YOLOv8Detector


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STORAGE_ROOT = PROJECT_ROOT / "backend" / "storage"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "weights" / "best.pt"


@dataclass(frozen=True)
class InspectionResult:
    inspection_id: str
    upload_path: str
    frames_dir: str
    annotated_dir: str
    report: InspectionReport

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["report"] = self.report.to_dict()
        return payload


def create_inspection_id() -> str:
    return f"insp_{uuid4().hex[:12]}"


def run_video_inspection(
    video_path: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    sample_fps: float = 1.0,
    confidence: float = 0.15,
    device: str = "0",
    inspection_id: str | None = None,
) -> InspectionResult:
    inspection_id = inspection_id or create_inspection_id()
    inspection_root = STORAGE_ROOT / "inspections" / inspection_id
    frames_dir = inspection_root / "frames"
    annotated_dir = inspection_root / "annotated"
    reports_dir = inspection_root / "reports"
    detector = YOLOv8Detector(model_path=model_path, confidence=confidence, device=device)
    detector.load()

    metadata, frames = extract_frames(
        video_path=video_path,
        output_dir=frames_dir,
        sample_fps=sample_fps,
        image_extension=".jpg",
    )

    frame_results = [
        detector.predict_frame(frame.output_path, annotated_output_dir=annotated_dir)
        for frame in frames
    ]

    report = generate_report(
        inspection_id=inspection_id,
        video_metadata=metadata,
        frame_results=frame_results,
        output_dir=reports_dir,
    )

    return InspectionResult(
        inspection_id=inspection_id,
        upload_path=str(Path(video_path).expanduser().resolve()),
        frames_dir=str(frames_dir),
        annotated_dir=str(annotated_dir),
        report=report,
    )
