"""Generate inspection reports for SkyVision-DL."""

from __future__ import annotations

import html
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.analytics.deduplication import estimate_unique_detections
from backend.video.frame_extractor import VideoMetadata
from backend.yolo.detector import FrameDetections


@dataclass(frozen=True)
class InspectionReport:
    inspection_id: str
    created_at: str
    video_metadata: dict[str, Any]
    total_frames_sampled: int
    total_detections: int
    detections_by_class: dict[str, int]
    defect_areas_needing_attention: int
    critical_defects: int
    high_defects: int
    medium_defects: int
    low_defects: int
    report_reliability: str
    maintenance_priority_list: list[dict[str, Any]]
    raw_detection_count: int
    repeated_frame_detections: int
    frames: list[dict[str, Any]]
    json_report_path: str
    html_report_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_report(
    inspection_id: str,
    video_metadata: VideoMetadata,
    frame_results: list[FrameDetections],
    output_dir: str | Path,
    extra_metadata: dict[str, Any] | None = None,
) -> InspectionReport:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    video_metadata_payload = video_metadata.to_dict()
    if extra_metadata:
        video_metadata_payload.update(extra_metadata)

    frame_payload = [
        {
            **frame.to_dict(),
            **_frame_context(frame, sequence_index, video_metadata),
        }
        for sequence_index, frame in enumerate(frame_results)
    ]
    class_counter: Counter[str] = Counter()
    for frame in frame_results:
        for detection in frame.detections:
            class_counter[detection.class_name] += 1
    maintenance_summary = estimate_unique_detections(
        [
            {
                **detection,
                "frame_width": video_metadata.width,
                "frame_height": video_metadata.height,
                "frame_index": frame.get("frame_index"),
                "frame_sequence_index": frame.get("frame_sequence_index"),
                "timestamp_seconds": frame.get("timestamp_seconds"),
                "image_path": frame.get("image_path"),
                "annotated_image_path": frame.get("annotated_image_path"),
            }
            for frame in frame_payload
            for detection in frame.get("detections", [])
        ]
    )

    json_path = destination / "inspection_report.json"
    html_path = destination / "inspection_report.html"

    report = InspectionReport(
        inspection_id=inspection_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        video_metadata=video_metadata_payload,
        total_frames_sampled=len(frame_results),
        total_detections=sum(class_counter.values()),
        detections_by_class=dict(sorted(class_counter.items())),
        defect_areas_needing_attention=maintenance_summary["defect_areas_needing_attention"],
        critical_defects=maintenance_summary["critical_defects"],
        high_defects=maintenance_summary["high_defects"],
        medium_defects=maintenance_summary["medium_defects"],
        low_defects=maintenance_summary["low_defects"],
        report_reliability=maintenance_summary["report_reliability"],
        maintenance_priority_list=maintenance_summary["maintenance_priority_list"],
        raw_detection_count=maintenance_summary["raw_detection_count"],
        repeated_frame_detections=maintenance_summary["repeated_frame_detections"],
        frames=frame_payload,
        json_report_path=str(json_path),
        html_report_path=str(html_path),
    )

    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    html_path.write_text(render_html_report(report), encoding="utf-8")
    return report


def _frame_context(
    frame: FrameDetections,
    sequence_index: int,
    video_metadata: VideoMetadata,
) -> dict[str, int | float | None]:
    frame_index = _parse_frame_index(frame.image_path)
    timestamp = frame_index / video_metadata.fps if frame_index is not None and video_metadata.fps else None
    return {
        "frame_index": frame_index if frame_index is not None else sequence_index,
        "frame_sequence_index": sequence_index,
        "timestamp_seconds": timestamp,
        "frame_width": video_metadata.width,
        "frame_height": video_metadata.height,
    }


def _parse_frame_index(image_path: str) -> int | None:
    match = re.search(r"(?:frame|input_frame)_(\d+)\.[A-Za-z0-9]+$", Path(image_path).name)
    if not match:
        return None
    return int(match.group(1))


def render_html_report(report: InspectionReport) -> str:
    media_type = str(report.video_metadata.get("media_type", "video")).title()
    inspection_metadata = report.video_metadata.get("inspection_metadata") or {}
    report_title = inspection_metadata.get("report_title") or "SkyVision-DL Inspection Report"
    duration_label = "Video duration seconds" if media_type.lower() == "video" else "Uploaded image reviewed"
    duration_value = report.video_metadata.get("duration_seconds") if media_type.lower() == "video" else "Single image"
    rows = "\n".join(
        f"<tr><td>{html.escape(name)}</td><td>{count}</td></tr>"
        for name, count in report.detections_by_class.items()
    )
    if not rows:
        rows = "<tr><td colspan=\"2\">No defects detected</td></tr>"

    priority_rows = "\n".join(
        "<tr>"
        f"<td>{item.get('priority')}</td>"
        f"<td>{html.escape(str(item.get('class_name', '')).replace('_', ' ').title())}</td>"
        f"<td>{html.escape(str(item.get('severity', '')))}</td>"
        f"<td>{html.escape(str(item.get('approximate_location', '')).title())}</td>"
        f"<td>{html.escape(str(item.get('evidence_frame_count', 0)))} frame(s), "
        f"{float(item.get('confidence', 0.0)):.2f} confidence</td>"
        f"<td>{html.escape(str(item.get('recommended_action', '')))}</td>"
        "</tr>"
        for item in report.maintenance_priority_list
    )
    if not priority_rows:
        priority_rows = "<tr><td colspan=\"6\">No maintenance action required from this run.</td></tr>"

    frame_sections = "\n".join(render_frame_section(frame) for frame in report.frames)
    metadata_rows = render_inspection_metadata_rows(inspection_metadata)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(str(report_title))}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #17202a; }}
    h1, h2 {{ margin-bottom: 8px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border: 1px solid #d8dee4; padding: 8px 10px; text-align: left; }}
    th {{ background: #f4f6f8; }}
    .frame {{ border-top: 1px solid #d8dee4; padding: 18px 0; }}
    .frame img {{ max-width: 100%; border: 1px solid #d8dee4; }}
    .muted {{ color: #687280; }}
  </style>
</head>
<body>
  <h1>{html.escape(str(report_title))}</h1>
  <p class="muted">Inspection ID: {html.escape(report.inspection_id)}</p>
  <p class="muted">Created at: {html.escape(report.created_at)}</p>

  <h2>Inspection Details</h2>
  <table>
    {metadata_rows}
  </table>

  <h2>Summary</h2>
  <table>
    <tr><th>Actual glass panels needing attention</th><td>{report.defect_areas_needing_attention}</td></tr>
    <tr><th>Critical defects</th><td>{report.critical_defects}</td></tr>
    <tr><th>High-priority defects</th><td>{report.high_defects}</td></tr>
    <tr><th>Report reliability</th><td>{html.escape(report.report_reliability)}</td></tr>
    <tr><th>Total sampled frames reviewed</th><td>{report.total_frames_sampled}</td></tr>
    <tr><th>Raw AI frame observations</th><td>{report.total_detections}</td></tr>
    <tr><th>Repeated observations grouped</th><td>{report.repeated_frame_detections}</td></tr>
    <tr><th>Media type</th><td>{html.escape(media_type)}</td></tr>
    <tr><th>{html.escape(duration_label)}</th><td>{html.escape(str(duration_value))}</td></tr>
  </table>

  <h2>Priority Glass Panels</h2>
  <table>
    <tr><th>Priority</th><th>Defect</th><th>Severity</th><th>Location</th><th>Evidence</th><th>Recommended Action</th></tr>
    {priority_rows}
  </table>

  <h2>Detections by Class</h2>
  <table>
    <tr><th>Class</th><th>Count</th></tr>
    {rows}
  </table>

  <h2>Defect Frames</h2>
  {frame_sections}
</body>
</html>
"""


def render_inspection_metadata_rows(metadata: dict[str, Any]) -> str:
    labels = {
        "organization_name": "Organization",
        "project_name": "Project",
        "inspector_name": "Inspector",
        "building_name": "Building",
        "floor_level": "Floor / Level",
        "facade_side": "Facade Side",
        "location": "Location",
        "inspection_date": "Inspection Date",
        "remarks": "Remarks",
    }
    rows = [
        f"<tr><th>{html.escape(label)}</th><td>{html.escape(str(metadata.get(key)))}</td></tr>"
        for key, label in labels.items()
        if metadata.get(key)
    ]
    return "\n".join(rows) if rows else "<tr><td>No inspection metadata provided.</td></tr>"


def render_frame_section(frame: dict[str, Any]) -> str:
    detections = frame.get("detections", [])
    detection_items = "\n".join(
        "<li>"
        f"{html.escape(str(item.get('class_name')))} "
        f"({float(item.get('confidence', 0.0)):.2f})"
        "</li>"
        for item in detections
    )
    if not detection_items:
        detection_items = "<li>No defects detected</li>"

    annotated_path = Path(str(frame.get("annotated_image_path", "")))
    heatmap_path = Path(str(frame.get("heatmap_image_path") or ""))
    report_image_path = f"../annotated/{annotated_path.name}"
    heatmap_image_path = f"../annotated/{heatmap_path.name}" if heatmap_path.name else report_image_path

    return f"""<section class="frame">
  <p><strong>{html.escape(Path(str(frame.get("image_path", ""))).name)}</strong></p>
  <p class="muted">Annotated defect view</p>
  <img src="{html.escape(report_image_path)}" alt="Annotated inspection frame">
  <p class="muted">Defect attention heatmap</p>
  <img src="{html.escape(heatmap_image_path)}" alt="Defect attention heatmap">
  <ul>{detection_items}</ul>
</section>"""
