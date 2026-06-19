"""YOLOv8 inference wrapper for SkyVision-DL."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any

import cv2


DEFECT_CLASSES = {
    0: "Broken Glass",
    1: "Crack",
    2: "Fracture",
    3: "Water Leakage",
    4: "Surface Stain",
    5: "Delamination",
    6: "Repair Patch",
}


class DetectorError(RuntimeError):
    """Raised when YOLO inference cannot be completed."""


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrameDetections:
    image_path: str
    annotated_image_path: str
    detections: list[Detection]

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "annotated_image_path": self.annotated_image_path,
            "detections": [detection.to_dict() for detection in self.detections],
        }


class YOLOv8Detector:
    """Small production wrapper around Ultralytics YOLOv8."""

    def __init__(self, model_path: str | Path, confidence: float = 0.25, device: str = "0") -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.confidence = confidence
        self.device = device
        self._model = None

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLO model weights not found: {self.model_path}. "
                "Train a model or provide --model-path / SKYVISION_MODEL_PATH."
            )

        try:
            os.environ.setdefault(
                "YOLO_CONFIG_DIR",
                str(Path(__file__).resolve().parents[2] / "backend" / "storage" / "ultralytics"),
            )
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorError("Ultralytics is not installed. Install it with: pip install ultralytics") from exc

        self._model = YOLO(str(self.model_path))

    def predict_frame(self, image_path: str | Path, annotated_output_dir: str | Path) -> FrameDetections:
        if self._model is None:
            self.load()

        path = Path(image_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Frame image not found: {path}")

        output_dir = Path(annotated_output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            results = self._model.predict(
                source=str(path),
                conf=self.confidence,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            raise DetectorError(f"YOLO inference failed for frame: {path}") from exc

        if not results:
            raise DetectorError(f"YOLO returned no result object for frame: {path}")

        result = results[0]
        detections = self._parse_detections(result)
        annotated_image_path = output_dir / path.name
        annotated_image = result.plot()

        if not cv2.imwrite(str(annotated_image_path), annotated_image):
            raise DetectorError(f"Failed to write annotated frame: {annotated_image_path}")

        return FrameDetections(
            image_path=str(path),
            annotated_image_path=str(annotated_image_path),
            detections=detections,
        )

    def _parse_detections(self, result: Any) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return []

        names = getattr(result, "names", {}) or DEFECT_CLASSES
        detections: list[Detection] = []
        xyxy_values = boxes.xyxy.cpu().tolist()
        confidence_values = boxes.conf.cpu().tolist()
        class_values = boxes.cls.cpu().tolist()

        for bbox, confidence, class_id_value in zip(xyxy_values, confidence_values, class_values):
            class_id = int(class_id_value)
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=str(names.get(class_id, DEFECT_CLASSES.get(class_id, f"class_{class_id}"))),
                    confidence=float(confidence),
                    bbox_xyxy=tuple(float(value) for value in bbox),
                )
            )

        return detections
