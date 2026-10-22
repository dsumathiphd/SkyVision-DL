"""YOLOv8 inference wrapper for SkyVision-DL."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ULTRALYTICS_MODEL_ALIASES = {
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
}


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
    heatmap_image_path: str
    detections: list[Detection]

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "annotated_image_path": self.annotated_image_path,
            "heatmap_image_path": self.heatmap_image_path,
            "detections": [detection.to_dict() for detection in self.detections],
        }


class YOLOv8Detector:
    """Small production wrapper around Ultralytics YOLOv8."""

    def __init__(self, model_path: str | Path, confidence: float = 0.15, device: str = "0") -> None:
        self.model_ref = str(model_path)
        self.model_path = Path(model_path).expanduser().resolve()
        self.confidence = confidence
        self.device = device
        self._model = None

    def load(self) -> None:
        if is_ultralytics_model_alias(self.model_ref):
            model_source = Path(self.model_ref).name.lower()
        elif self.model_path.exists():
            model_source = str(self.model_path)
        else:
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

        self._model = YOLO(model_source)

    def predict_frame(self, image_path: str | Path, annotated_output_dir: str | Path) -> FrameDetections:
        if self._model is None:
            self.load()

        path = Path(image_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Frame image not found: {path}")

        output_dir = Path(annotated_output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        import torch
        target_device = self.device
        if target_device != "cpu" and not torch.cuda.is_available():
            target_device = "cpu"

        try:
            results = self._model.predict(
                source=str(path),
                conf=self.confidence,
                device=target_device,
                verbose=False,
            )
        except Exception as exc:
            raise DetectorError(
                f"YOLO inference could not process frame '{path.name}'. "
                "Confirm the model path is valid and use device 'cpu' if GPU is unavailable. "
                f"Details: {exc}"
            ) from exc

        if not results:
            raise DetectorError(f"YOLO returned no result object for frame: {path}")

        result = results[0]
        source_image = cv2.imread(str(path))
        if source_image is None:
            raise DetectorError(f"Failed to read frame image for annotation: {path}")

        detections = _parse_detections_with_active_image(self, result, source_image)
        if not detections:
            detections = _predict_tiled_detections(
                detector=self,
                source_image=source_image,
                image_shape=source_image.shape,
            )
        annotated_image_path = output_dir / path.name
        heatmap_image_path = output_dir / f"heatmap_{path.name}"
        annotated_image = _draw_detections(source_image, detections)
        heatmap_image = _draw_detection_heatmap(source_image, detections)

        if not cv2.imwrite(str(annotated_image_path), annotated_image):
            raise DetectorError(f"Failed to write annotated frame: {annotated_image_path}")
        if not cv2.imwrite(str(heatmap_image_path), heatmap_image):
            raise DetectorError(f"Failed to write heatmap frame: {heatmap_image_path}")

        return FrameDetections(
            image_path=str(path),
            annotated_image_path=str(annotated_image_path),
            heatmap_image_path=str(heatmap_image_path),
            detections=detections,
        )

    def _parse_detections(self, result: Any, image_shape: tuple[int, ...]) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return []

        names = getattr(result, "names", {}) or DEFECT_CLASSES
        detections: list[Detection] = []
        xyxy_values = boxes.xyxy.cpu().tolist()
        confidence_values = boxes.conf.cpu().tolist()
        class_values = boxes.cls.cpu().tolist()

        for bbox, confidence, class_id_value in zip(xyxy_values, confidence_values, class_values):
            detections.extend(
                _build_detections_from_box(
                    names=names,
                    image_shape=image_shape,
                    bbox=bbox,
                    confidence=float(confidence),
                    class_id=int(class_id_value),
                )
            )

        return _nms_detections(detections, iou_threshold=0.35)


def _build_detections_from_box(
    names: dict[int, str],
    image_shape: tuple[int, ...],
    bbox: list[float] | tuple[float, float, float, float],
    confidence: float,
    class_id: int,
) -> list[Detection]:
    class_name = str(names.get(class_id, DEFECT_CLASSES.get(class_id, f"class_{class_id}")))
    refined_boxes = _refine_panel_boxes(image_shape, list(bbox)) if _is_broken_glass(class_name) else [bbox]
    if _is_broken_glass(class_name):
        filtered_boxes = _filter_window_opening_boxes(refined_boxes, image_shape)
        if not filtered_boxes and confidence >= 0.05:
            filtered_boxes = [
                candidate
                for candidate in refined_boxes
                if _looks_like_glass_defect_region(candidate, image_shape)
            ]
        refined_boxes = filtered_boxes

    adjusted_confidence = float(confidence)
    if len(refined_boxes) > 1:
        adjusted_confidence = max(adjusted_confidence - 0.01, 0.01)

    return [
        Detection(
            class_id=class_id,
            class_name=class_name,
            confidence=adjusted_confidence,
            bbox_xyxy=tuple(float(value) for value in refined_bbox),
        )
        for refined_bbox in refined_boxes
    ]


def _predict_tiled_detections(
    detector: YOLOv8Detector,
    source_image: np.ndarray,
    image_shape: tuple[int, ...],
) -> list[Detection]:
    frame_height, frame_width = image_shape[:2]
    if min(frame_width, frame_height) < 900:
        return []

    import torch

    target_device = detector.device
    if target_device != "cpu" and not torch.cuda.is_available():
        target_device = "cpu"

    tile_size = 640
    stride = 448
    x_starts = _tile_starts(frame_width, tile_size, stride)
    y_starts = _tile_starts(frame_height, tile_size, stride)
    detections: list[Detection] = []

    _set_active_image(source_image)
    try:
        for y1 in y_starts:
            for x1 in x_starts:
                x2 = min(x1 + tile_size, frame_width)
                y2 = min(y1 + tile_size, frame_height)
                tile = source_image[y1:y2, x1:x2]
                if tile.shape[0] < 160 or tile.shape[1] < 160:
                    continue

                results = detector._model.predict(
                    source=tile,
                    conf=detector.confidence,
                    device=target_device,
                    verbose=False,
                )
                if not results:
                    continue

                result = results[0]
                boxes = getattr(result, "boxes", None)
                if boxes is None or boxes.xyxy is None:
                    continue

                names = getattr(result, "names", {}) or DEFECT_CLASSES
                xyxy_values = boxes.xyxy.cpu().tolist()
                confidence_values = boxes.conf.cpu().tolist()
                class_values = boxes.cls.cpu().tolist()
                for bbox, confidence, class_id_value in zip(xyxy_values, confidence_values, class_values):
                    global_bbox = (
                        float(bbox[0]) + x1,
                        float(bbox[1]) + y1,
                        float(bbox[2]) + x1,
                        float(bbox[3]) + y1,
                    )
                    detections.extend(
                        _build_detections_from_box(
                            names=names,
                            image_shape=image_shape,
                            bbox=global_bbox,
                            confidence=float(confidence),
                            class_id=int(class_id_value),
                        )
                    )
    finally:
        _clear_active_image()

    return _nms_detections(detections, iou_threshold=0.25)


def _tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]

    starts = list(range(0, max(length - tile_size + 1, 1), stride))
    last = max(length - tile_size, 0)
    if starts[-1] != last:
        starts.append(last)
    return starts


def _is_broken_glass(class_name: str) -> bool:
    return class_name.lower().replace("_", " ") == "broken glass"


def _refine_panel_boxes(image_shape: tuple[int, ...], bbox: list[float]) -> list[tuple[float, float, float, float]]:
    frame_height, frame_width = image_shape[:2]
    x1, y1, x2, y2 = _clip_bbox(bbox, frame_width, frame_height)
    box_area_ratio = ((x2 - x1) * (y2 - y1)) / max(frame_width * frame_height, 1)
    if box_area_ratio < 0.08:
        return [(x1, y1, x2, y2)]

    contour_boxes = _panel_boxes_from_edges(image_shape, (x1, y1, x2, y2))
    if len(contour_boxes) >= 2:
        return contour_boxes[:12]

    return _panel_boxes_from_grid(image_shape, (x1, y1, x2, y2))[:8]


def _panel_boxes_from_edges(
    image_shape: tuple[int, ...],
    bbox: tuple[float, float, float, float],
) -> list[tuple[float, float, float, float]]:
    frame_height, frame_width = image_shape[:2]
    x1, y1, x2, y2 = [int(value) for value in bbox]
    image = _current_image()
    if image is None:
        return []

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return []

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 140)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[float, float, float, float]] = []
    frame_area = max(frame_width * frame_height, 1)
    for contour in contours:
        rx, ry, rw, rh = cv2.boundingRect(contour)
        area_ratio = (rw * rh) / frame_area
        aspect_ratio = rw / max(rh, 1)
        if rw < 35 or rh < 35:
            continue
        if area_ratio < 0.004 or area_ratio > 0.18:
            continue
        if aspect_ratio < 0.35 or aspect_ratio > 3.4:
            continue
        candidates.append((x1 + rx, y1 + ry, x1 + rx + rw, y1 + ry + rh))

    return _nms(candidates, iou_threshold=0.35)


_ACTIVE_IMAGE: np.ndarray | None = None


def _current_image() -> np.ndarray | None:
    return _ACTIVE_IMAGE


def _panel_boxes_from_grid(
    image_shape: tuple[int, ...],
    bbox: tuple[float, float, float, float],
) -> list[tuple[float, float, float, float]]:
    frame_height, frame_width = image_shape[:2]
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    cols = 3 if width / max(frame_width, 1) > 0.45 else 2
    rows = 2 if height / max(frame_height, 1) > 0.45 else 1
    boxes = []
    for row in range(rows):
        for col in range(cols):
            px1 = x1 + (width * col / cols)
            py1 = y1 + (height * row / rows)
            px2 = x1 + (width * (col + 1) / cols)
            py2 = y1 + (height * (row + 1) / rows)
            boxes.append(_pad_inward((px1, py1, px2, py2), 0.08))
    return boxes


def _filter_window_opening_boxes(
    boxes: list[tuple[float, float, float, float]],
    image_shape: tuple[int, ...],
) -> list[tuple[float, float, float, float]]:
    tightened = []
    for box in boxes:
        candidate = _tighten_to_window_opening(box, image_shape)
        if candidate is not None:
            tightened.append(candidate)
    return _nms(tightened, iou_threshold=0.25)


def _looks_like_glass_defect_region(
    bbox: tuple[float, float, float, float],
    image_shape: tuple[int, ...],
) -> bool:
    image = _current_image()
    if image is None:
        return True

    frame_height, frame_width = image_shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in _clip_bbox(list(bbox), frame_width, frame_height)]
    width = x2 - x1
    height = y2 - y1
    if width < 28 or height < 28:
        return False

    area_ratio = (width * height) / max(frame_width * frame_height, 1)
    aspect_ratio = width / max(height, 1)
    if area_ratio < 0.0015 or area_ratio > 0.16:
        return False
    if aspect_ratio < 0.25 or aspect_ratio > 4.2:
        return False

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 35, 130)

    edge_density = float(np.mean(edges > 0))
    dark_ratio = float(np.mean(gray < 92))
    green_ratio = float(np.mean((hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 90) & (hsv[:, :, 1] > 40)))
    bright_wall_ratio = float(np.mean((gray > 130) & (hsv[:, :, 1] < 65)))

    if green_ratio > 0.24:
        return False
    if bright_wall_ratio > 0.78 and dark_ratio < 0.08:
        return False
    return edge_density >= 0.012 and dark_ratio >= 0.025


def _tighten_to_window_opening(
    bbox: tuple[float, float, float, float],
    image_shape: tuple[int, ...],
) -> tuple[float, float, float, float] | None:
    image = _current_image()
    if image is None:
        return bbox

    frame_height, frame_width = image_shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in _clip_bbox(list(bbox), frame_width, frame_height)]
    if x2 <= x1 or y2 <= y1:
        return None

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    green_ratio = float(np.mean((hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 90) & (hsv[:, :, 1] > 40)))
    dark_mask = (gray < 76).astype(np.uint8) * 255
    dark_mask = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    edges = cv2.Canny(gray, 45, 140)
    edge_density = float(np.mean(edges > 0))
    dark_ratio = float(np.mean(dark_mask > 0))
    bright_wall_ratio = float(np.mean((gray > 125) & (hsv[:, :, 1] < 70)))

    if dark_ratio < 0.035 or edge_density < 0.018:
        return None
    if green_ratio > 0.20:
        return None
    if bright_wall_ratio >= 0.68 and dark_ratio < 0.12:
        return None

    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_candidates = []
    roi_area = max(roi.shape[0] * roi.shape[1], 1)
    for contour in contours:
        rx, ry, rw, rh = cv2.boundingRect(contour)
        contour_area = cv2.contourArea(contour)
        component_area_ratio = contour_area / roi_area
        if rw < 18 or rh < 24 or component_area_ratio < 0.004:
            continue
        contour_candidates.append((rx, ry, rw, rh, contour_area))

    if not contour_candidates:
        return None

    largest_area = max(item[4] for item in contour_candidates)
    meaningful_contours = [
        (rx, ry, rw, rh)
        for rx, ry, rw, rh, contour_area in contour_candidates
        if contour_area >= largest_area * 0.18
    ]
    left = min(item[0] for item in meaningful_contours)
    top = min(item[1] for item in meaningful_contours)
    right = max(item[0] + item[2] for item in meaningful_contours)
    bottom = max(item[1] + item[3] for item in meaningful_contours)
    margin_x = max(int((right - left) * 0.16), 8)
    margin_y = max(int((bottom - top) * 0.18), 8)
    tx1 = max(x1 + left - margin_x, x1)
    ty1 = max(y1 + top - margin_y, y1)
    tx2 = min(x1 + right + margin_x, x2)
    ty2 = min(y1 + bottom + margin_y, y2)

    width = tx2 - tx1
    height = ty2 - ty1
    tightened_roi = image[ty1:ty2, tx1:tx2]
    if tightened_roi.size == 0:
        return None
    tightened_gray = cv2.cvtColor(tightened_roi, cv2.COLOR_BGR2GRAY)
    tightened_hsv = cv2.cvtColor(tightened_roi, cv2.COLOR_BGR2HSV)
    tightened_green_ratio = float(
        np.mean((tightened_hsv[:, :, 0] > 35) & (tightened_hsv[:, :, 0] < 90) & (tightened_hsv[:, :, 1] > 40))
    )
    area_ratio = (width * height) / max(frame_width * frame_height, 1)
    aspect_ratio = width / max(height, 1)
    tightened_dark_ratio = float(np.mean(tightened_gray < 76))
    if width < 45 or height < 45:
        return None
    if tightened_green_ratio > 0.18:
        return None
    if tightened_dark_ratio < 0.10:
        return None
    if width / max(frame_width, 1) > 0.34 or height / max(frame_height, 1) > 0.46:
        return None
    if area_ratio < 0.0025 or area_ratio > 0.10:
        return None
    if aspect_ratio < 0.28 or aspect_ratio > 3.8:
        return None
    return (float(tx1), float(ty1), float(tx2), float(ty2))


def _clip_bbox(bbox: list[float], frame_width: int, frame_height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return (
        min(max(x1, 0.0), float(frame_width - 1)),
        min(max(y1, 0.0), float(frame_height - 1)),
        min(max(x2, 0.0), float(frame_width - 1)),
        min(max(y2, 0.0), float(frame_height - 1)),
    )


def _pad_inward(bbox: tuple[float, float, float, float], ratio: float) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    dx = (x2 - x1) * ratio
    dy = (y2 - y1) * ratio
    return (x1 + dx, y1 + dy, x2 - dx, y2 - dy)


def _draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    annotated = image.copy()
    ordered_detections = sorted(
        detections,
        key=lambda item: 1 if _is_surface_stain_like_region(image, item) else 0,
    )
    for detection in ordered_detections:
        x1, y1, x2, y2 = [int(round(value)) for value in detection.bbox_xyxy]
        color = (35, 35, 220) if _is_broken_glass(detection.class_name) else (30, 130, 230)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def _draw_detection_heatmap(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Create a class-aware visual heatmap from detection boxes and confidences."""

    if not detections:
        return image.copy()

    height, width = image.shape[:2]
    overlay = np.zeros_like(image, dtype=np.float32)
    alpha = np.zeros((height, width), dtype=np.float32)
    for detection in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in detection.bbox_xyxy]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        is_stain_like = _is_surface_stain_like_region(image, detection)
        color = np.array(_heatmap_color("surface_stain") if is_stain_like else _heatmap_color(detection.class_name), dtype=np.float32)
        confidence_weight = min(max(float(detection.confidence), 0.08), 1.0)
        class_weight = _heatmap_severity_weight(detection.class_name)
        detection_alpha = min(0.24 + confidence_weight * class_weight * 0.50, 0.78)
        if is_stain_like:
            detection_alpha = max(detection_alpha, 0.62)

        roi_alpha = alpha[y1:y2, x1:x2]
        update_mask = detection_alpha >= (roi_alpha * 0.85)
        overlay_roi = overlay[y1:y2, x1:x2]
        overlay_roi[update_mask] = color
        roi_alpha[update_mask] = detection_alpha

    if not np.any(alpha):
        return image.copy()

    sigma = max(width, height) * 0.012
    overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=sigma)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=sigma)
    alpha = np.clip(alpha, 0.0, 0.72)
    blended = image.astype(np.float32) * (1.0 - alpha[..., None]) + overlay * alpha[..., None]
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    legend_height = 34
    cv2.rectangle(blended, (0, 0), (610, legend_height), (12, 22, 32), -1)
    cv2.putText(
        blended,
        "Heatmap colors: red broken glass, orange fracture, blue surface stain",
        (12, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return blended


def _heatmap_color(class_name: str) -> tuple[int, int, int]:
    normalized = class_name.lower().replace(" ", "_")
    if normalized == "broken_glass":
        return (35, 35, 230)  # red in BGR
    if normalized == "fracture":
        return (0, 145, 255)  # orange in BGR
    if normalized == "surface_stain":
        return (230, 120, 25)  # blue in BGR
    if normalized == "crack":
        return (0, 210, 255)  # yellow-orange in BGR
    if normalized in {"water_leakage", "delamination"}:
        return (90, 60, 210)  # magenta-red in BGR
    if normalized == "repair_patch":
        return (85, 190, 80)  # green in BGR
    return (40, 170, 220)


def _is_surface_stain_like_region(image: np.ndarray, detection: Detection) -> bool:
    normalized = detection.class_name.lower().replace(" ", "_")
    if normalized == "surface_stain":
        return True
    if normalized != "broken_glass" or detection.confidence >= 0.45:
        return False

    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in detection.bbox_xyxy]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 0 or box_height <= 0:
        return False

    area_ratio = (box_width * box_height) / max(width * height, 1)
    aspect_ratio = box_height / max(box_width, 1)
    if area_ratio < 0.004 or aspect_ratio < 1.05:
        return False

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 45, 135)
    vertical_kernel_height = max(int(box_height * 0.18), 18)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, vertical_kernel_height))
    vertical_streaks = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
    vertical_ratio = float(np.mean(vertical_streaks > 0))
    dark_ratio = float(np.mean(gray < 95))
    bright_ratio = float(np.mean(gray > 155))
    low_saturation_ratio = float(np.mean(hsv[:, :, 1] < 75))

    if low_saturation_ratio >= 0.85 and dark_ratio >= 0.08 and bright_ratio <= 0.08:
        return True
    return vertical_ratio >= 0.002 or (dark_ratio >= 0.10 and bright_ratio >= 0.08)


def _heatmap_severity_weight(class_name: str) -> float:
    normalized = class_name.lower().replace(" ", "_")
    if normalized in {"broken_glass", "fracture"}:
        return 1.0
    if normalized in {"crack", "water_leakage", "delamination"}:
        return 0.75
    if normalized in {"surface_stain", "repair_patch"}:
        return 0.45
    return 0.6


def _nms(
    boxes: list[tuple[float, float, float, float]],
    iou_threshold: float,
) -> list[tuple[float, float, float, float]]:
    selected: list[tuple[float, float, float, float]] = []
    for box in sorted(boxes, key=_area, reverse=True):
        if all(_iou(box, kept) < iou_threshold for kept in selected):
            selected.append(box)
    return selected


def _nms_detections(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    selected: list[Detection] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if all(
            detection.class_name != kept.class_name or _iou(detection.bbox_xyxy, kept.bbox_xyxy) < iou_threshold
            for kept in selected
        ):
            selected.append(detection)
    return selected


def _area(box: tuple[float, float, float, float]) -> float:
    return max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)


def _iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    union = _area(first) + _area(second) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _set_active_image(image: np.ndarray) -> None:
    global _ACTIVE_IMAGE
    _ACTIVE_IMAGE = image


def _clear_active_image() -> None:
    global _ACTIVE_IMAGE
    _ACTIVE_IMAGE = None


def _parse_detections_with_active_image(
    detector: YOLOv8Detector,
    result: Any,
    source_image: np.ndarray,
) -> list[Detection]:
    _set_active_image(source_image)
    try:
        return detector._parse_detections(result, source_image.shape)
    finally:
        _clear_active_image()


def is_ultralytics_model_alias(model_ref: str | Path) -> bool:
    ref_str = str(model_ref).strip().lower()
    return ref_str in ULTRALYTICS_MODEL_ALIASES or Path(ref_str).name in ULTRALYTICS_MODEL_ALIASES
