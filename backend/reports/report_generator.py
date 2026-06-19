"""Generate MVP inspection reports for SkyVision-DL."""

from __future__ import annotations

import html
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
) -> InspectionReport:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    frame_payload = [frame.to_dict() for frame in frame_results]
    class_counter: Counter[str] = Counter()
    for frame in frame_results:
        for detection in frame.detections:
            class_counter[detection.class_name] += 1

    json_path = destination / "inspection_report.json"
    html_path = destination / "inspection_report.html"

    report = InspectionReport(
        inspection_id=inspection_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        video_metadata=video_metadata.to_dict(),
        total_frames_sampled=len(frame_results),
        total_detections=sum(class_counter.values()),
        detections_by_class=dict(sorted(class_counter.items())),
        frames=frame_payload,
        json_report_path=str(json_path),
        html_report_path=str(html_path),
    )

    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    html_path.write_text(render_html_report(report), encoding="utf-8")
    return report


def render_html_report(report: InspectionReport) -> str:
    rows = "\n".join(
        f"<tr><td>{html.escape(name)}</td><td>{count}</td></tr>"
        for name, count in report.detections_by_class.items()
    )
    if not rows:
        rows = "<tr><td colspan=\"2\">No defects detected</td></tr>"

    frame_sections = "\n".join(render_frame_section(frame) for frame in report.frames)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SkyVision-DL Inspection Report</title>
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
  <h1>SkyVision-DL Inspection Report</h1>
  <p class="muted">Inspection ID: {html.escape(report.inspection_id)}</p>
  <p class="muted">Created at: {html.escape(report.created_at)}</p>

  <h2>Summary</h2>
  <table>
    <tr><th>Total sampled frames</th><td>{report.total_frames_sampled}</td></tr>
    <tr><th>Total detections</th><td>{report.total_detections}</td></tr>
    <tr><th>Video FPS</th><td>{report.video_metadata.get("fps")}</td></tr>
    <tr><th>Video duration seconds</th><td>{report.video_metadata.get("duration_seconds")}</td></tr>
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
    report_image_path = f"../annotated/{annotated_path.name}"

    return f"""<section class="frame">
  <p><strong>{html.escape(Path(str(frame.get("image_path", ""))).name)}</strong></p>
  <img src="{html.escape(report_image_path)}" alt="Annotated inspection frame">
  <ul>{detection_items}</ul>
</section>"""
