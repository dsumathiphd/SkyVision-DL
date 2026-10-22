"""Maintenance-oriented defect grouping for sampled video frames."""

from __future__ import annotations

from collections import Counter
import json
from typing import Any


DEFAULT_IOU_THRESHOLD = 0.38
TEMPORAL_FRAME_WINDOW = 8
TEMPORAL_CENTER_DISTANCE = 0.36
TEMPORAL_LARGE_BOX_CENTER_DISTANCE = 0.34


def estimate_unique_detections(
    detections: list[dict[str, Any]],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict[str, Any]:
    """Group repeated frame detections into maintenance action areas."""

    normalized = [_normalize_detection(item) for item in detections]
    normalized = [item for item in normalized if item is not None]
    normalized.sort(key=lambda item: (item["sequence_index"], -item["confidence"]))

    clusters: list[dict[str, Any]] = []
    for detection in normalized:
        match = _find_matching_cluster(clusters, detection, iou_threshold)
        if match is None:
            clusters.append(
                {
                    "class_name": detection["class_name"],
                    "confidence": detection["confidence"],
                    "bbox": detection["bbox"],
                    "bbox_xyxy": detection["bbox_xyxy"],
                    "frame_index": detection["frame_index"],
                    "sequence_index": detection["sequence_index"],
                    "timestamp_seconds": detection["timestamp_seconds"],
                    "frame_width": detection["frame_width"],
                    "frame_height": detection["frame_height"],
                    "image_path": detection["image_path"],
                    "annotated_image_path": detection["annotated_image_path"],
                    "centers": [detection["center"]],
                    "areas": [detection["area"]],
                    "aspect_ratios": [detection["aspect_ratio"]],
                    "frame_indices": [detection["sequence_index"]],
                    "members": 1,
                }
            )
            continue

        match["members"] += 1
        if detection["confidence"] > match["confidence"]:
            match["confidence"] = detection["confidence"]
            match["bbox"] = detection["bbox"]
            match["bbox_xyxy"] = detection["bbox_xyxy"]
            match["frame_index"] = detection["frame_index"]
            match["sequence_index"] = detection["sequence_index"]
            match["timestamp_seconds"] = detection["timestamp_seconds"]
            match["frame_width"] = detection["frame_width"]
            match["frame_height"] = detection["frame_height"]
            match["image_path"] = detection["image_path"]
            match["annotated_image_path"] = detection["annotated_image_path"]
        match["centers"].append(detection["center"])
        match["areas"].append(detection["area"])
        match["aspect_ratios"].append(detection["aspect_ratio"])
        match["frame_indices"].append(detection["sequence_index"])

    priority_items = [_build_priority_item(index, cluster) for index, cluster in enumerate(clusters, start=1)]
    priority_items.sort(key=lambda item: (item["severity_rank"], item["priority_score"]), reverse=True)
    for index, item in enumerate(priority_items, start=1):
        item["priority"] = index
        item.pop("severity_rank", None)

    severity_counts = Counter(item["severity"] for item in priority_items)
    unique_by_class = Counter(str(cluster["class_name"]) for cluster in clusters)
    raw_count = len(normalized)
    unique_count = len(clusters)
    return {
        "raw_detection_count": raw_count,
        "estimated_unique_detections": unique_count,
        "defect_areas_needing_attention": unique_count,
        "repeated_frame_detections": max(raw_count - unique_count, 0),
        "unique_detections_by_class": dict(sorted(unique_by_class.items())),
        "maintenance_priority_list": priority_items,
        "critical_defects": severity_counts.get("Critical", 0),
        "high_defects": severity_counts.get("High", 0),
        "medium_defects": severity_counts.get("Medium", 0),
        "low_defects": severity_counts.get("Low", 0),
        "report_reliability": _overall_reliability(priority_items),
        "deduplication_method": (
            f"same-class normalized bbox IoU >= {iou_threshold:.2f} plus "
            f"temporal overlap grouping within {TEMPORAL_FRAME_WINDOW} sampled frames"
        ),
    }


def _normalize_detection(detection: dict[str, Any]) -> dict[str, Any] | None:
    bbox = detection.get("bbox_xyxy")
    if isinstance(bbox, str):
        try:
            bbox = json.loads(bbox)
        except json.JSONDecodeError:
            return None
    width = detection.get("frame_width")
    height = detection.get("frame_height")
    if not isinstance(bbox, list | tuple) or len(bbox) != 4 or not width or not height:
        return None

    try:
        x1, y1, x2, y2 = [float(value) for value in bbox]
        frame_width = float(width)
        frame_height = float(height)
        confidence = float(detection.get("confidence", 0.0))
    except (TypeError, ValueError):
        return None

    if frame_width <= 0 or frame_height <= 0:
        return None

    nx1 = min(max(x1 / frame_width, 0.0), 1.0)
    ny1 = min(max(y1 / frame_height, 0.0), 1.0)
    nx2 = min(max(x2 / frame_width, 0.0), 1.0)
    ny2 = min(max(y2 / frame_height, 0.0), 1.0)
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    area = max((nx2 - nx1) * (ny2 - ny1), 0.0)
    aspect_ratio = (nx2 - nx1) / max(ny2 - ny1, 1e-6)
    center = ((nx1 + nx2) / 2, (ny1 + ny2) / 2)

    return {
        "class_name": str(detection.get("class_name", "unknown")),
        "confidence": confidence,
        "bbox": (nx1, ny1, nx2, ny2),
        "center": center,
        "area": area,
        "aspect_ratio": aspect_ratio,
        "bbox_xyxy": [x1, y1, x2, y2],
        "frame_index": detection.get("frame_index"),
        "sequence_index": _as_int(detection.get("frame_sequence_index", detection.get("frame_index"))) or 0,
        "timestamp_seconds": detection.get("timestamp_seconds"),
        "frame_width": frame_width,
        "frame_height": frame_height,
        "image_path": detection.get("image_path"),
        "annotated_image_path": detection.get("annotated_image_path"),
    }


def _build_priority_item(index: int, cluster: dict[str, Any]) -> dict[str, Any]:
    bbox = cluster["bbox"]
    class_name = str(cluster["class_name"])
    confidence = float(cluster["confidence"])
    evidence_count = int(cluster["members"])
    area_ratio = max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 0.0)
    severity = _severity(class_name, confidence, evidence_count, area_ratio)
    reliability = _item_reliability(confidence, evidence_count)
    priority_score = _priority_score(severity, confidence, evidence_count, area_ratio)

    return {
        "priority": index,
        "class_name": class_name,
        "severity": severity,
        "severity_rank": _severity_rank(severity),
        "priority_score": round(priority_score, 3),
        "confidence": round(confidence, 3),
        "evidence_frame_count": evidence_count,
        "approximate_location": _approximate_location(bbox),
        "recommended_action": _recommended_action(class_name, severity),
        "reliability": reliability,
        "area_ratio": round(area_ratio, 3),
        "frame_index": cluster.get("frame_index"),
        "timestamp_seconds": cluster.get("timestamp_seconds"),
        "bbox_xyxy": cluster.get("bbox_xyxy"),
        "image_path": cluster.get("image_path"),
        "annotated_image_path": cluster.get("annotated_image_path"),
    }


def _severity(class_name: str, confidence: float, evidence_count: int, area_ratio: float) -> str:
    name = class_name.lower().replace(" ", "_")
    if name == "broken_glass":
        if area_ratio >= 0.30 or evidence_count >= 6 or confidence >= 0.70:
            return "Critical"
        if confidence >= 0.03 or evidence_count >= 1:
            return "High"
        return "Medium"
    if name in {"crack", "fracture"}:
        if name == "fracture" and (confidence >= 0.35 or evidence_count >= 2 or area_ratio >= 0.08):
            return "High"
        if confidence >= 0.55 or evidence_count >= 3 or area_ratio >= 0.12:
            return "High"
        return "Medium"
    if name in {"water_leakage", "delamination"}:
        if area_ratio >= 0.12 or evidence_count >= 2 or confidence >= 0.45:
            return "High"
        return "Medium"
    if name in {"surface_stain", "repair_patch"}:
        if area_ratio >= 0.25 or evidence_count >= 5:
            return "Medium"
        return "Low"
    if area_ratio >= 0.25 or evidence_count >= 5:
        return "High"
    return "Medium"


def _recommended_action(class_name: str, severity: str) -> str:
    name = class_name.lower().replace("_", " ")
    if "broken" in name:
        if severity == "Critical":
            return "Restrict nearby access and schedule immediate glass replacement inspection."
        return "Plan glass replacement assessment in the next maintenance cycle."
    if "crack" in name or "fracture" in name:
        return "Inspect crack propagation risk and seal or replace the affected panel."
    if "water" in name or "leakage" in name:
        return "Check gasket, sealant, and drainage path for water ingress."
    if "delamination" in name:
        return "Inspect laminated layer separation and plan panel replacement if spreading."
    if "stain" in name:
        return "Clean and re-inspect to confirm whether staining hides a deeper defect."
    if "repair" in name or "patch" in name:
        return "Review prior repair quality and monitor during the next inspection."
    return "Review the flagged facade area during maintenance inspection."


def _approximate_location(bbox: tuple[float, float, float, float]) -> str:
    x_center = (bbox[0] + bbox[2]) / 2
    y_center = (bbox[1] + bbox[3]) / 2
    horizontal = "left" if x_center < 0.33 else "right" if x_center > 0.67 else "center"
    vertical = "upper" if y_center < 0.33 else "lower" if y_center > 0.67 else "middle"
    if horizontal == "center":
        return f"{vertical} facade region"
    return f"{vertical}-{horizontal} facade region"


def _item_reliability(confidence: float, evidence_count: int) -> str:
    if evidence_count >= 5 and confidence >= 0.5:
        return "High"
    if evidence_count >= 2 or confidence >= 0.4:
        return "Medium"
    return "Low"


def _overall_reliability(priority_items: list[dict[str, Any]]) -> str:
    if not priority_items:
        return "No defects detected"
    high_count = sum(1 for item in priority_items if item["reliability"] == "High")
    medium_or_high = sum(1 for item in priority_items if item["reliability"] in {"Medium", "High"})
    if high_count / len(priority_items) >= 0.7:
        return "High"
    if medium_or_high:
        return "Medium"
    return "Low"


def _priority_score(severity: str, confidence: float, evidence_count: int, area_ratio: float) -> float:
    evidence_component = min(evidence_count / 10, 1.0)
    area_component = min(area_ratio / 0.5, 1.0)
    return _severity_rank(severity) + confidence + evidence_component + area_component


def _severity_rank(severity: str) -> int:
    return {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}.get(severity, 0)


def _find_matching_cluster(
    clusters: list[dict[str, Any]],
    detection: dict[str, Any],
    iou_threshold: float,
) -> dict[str, Any] | None:
    for cluster in clusters:
        if cluster["class_name"] != detection["class_name"]:
            continue
        if _iou(cluster["bbox"], detection["bbox"]) >= iou_threshold:
            return cluster
        if _temporal_overlap_match(cluster, detection):
            return cluster
    return None


def _temporal_overlap_match(cluster: dict[str, Any], detection: dict[str, Any]) -> bool:
    detection_frame = _as_int(detection.get("sequence_index"))
    if detection_frame is None:
        return False

    cluster_frames = [_as_int(value) for value in cluster.get("frame_indices", [])]
    cluster_frames = [value for value in cluster_frames if value is not None]
    if not cluster_frames:
        return False

    min_frame_gap = min(abs(detection_frame - frame_index) for frame_index in cluster_frames)
    if min_frame_gap == 0 or min_frame_gap > TEMPORAL_FRAME_WINDOW:
        return False

    detection_area = float(detection.get("area", 0.0))
    cluster_areas = [float(value) for value in cluster.get("areas", []) if float(value) > 0]
    if not cluster_areas or detection_area <= 0:
        return False

    closest_area_ratio = min(
        max(detection_area, cluster_area) / max(min(detection_area, cluster_area), 1e-6)
        for cluster_area in cluster_areas
    )
    if closest_area_ratio > 4.0:
        return False

    detection_aspect = float(detection.get("aspect_ratio", 0.0))
    cluster_aspects = [float(value) for value in cluster.get("aspect_ratios", []) if float(value) > 0]
    if detection_aspect <= 0 or not cluster_aspects:
        return False
    closest_aspect_ratio = min(
        max(detection_aspect, cluster_aspect) / max(min(detection_aspect, cluster_aspect), 1e-6)
        for cluster_aspect in cluster_aspects
    )
    if closest_aspect_ratio > 4.0:
        return False

    center = detection.get("center")
    if not isinstance(center, tuple | list) or len(center) != 2:
        return False

    center_distances = [
        _center_distance(tuple(center), tuple(cluster_center))
        for cluster_center in cluster.get("centers", [])
        if isinstance(cluster_center, tuple | list) and len(cluster_center) == 2
    ]
    if not center_distances:
        return False

    distance_threshold = (
        TEMPORAL_LARGE_BOX_CENTER_DISTANCE
        if detection_area >= 0.05 or max(cluster_areas) >= 0.05
        else TEMPORAL_CENTER_DISTANCE
    )
    return min(center_distances) <= distance_threshold


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _center_distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return ((float(first[0]) - float(second[0])) ** 2 + (float(first[1]) - float(second[1])) ** 2) ** 0.5


def _iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter_width = max(ix2 - ix1, 0.0)
    inter_height = max(iy2 - iy1, 0.0)
    intersection = inter_width * inter_height

    first_area = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    second_area = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = first_area + second_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union
