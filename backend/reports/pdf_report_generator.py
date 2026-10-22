"""PDF inspection report generation for SkyVision-DL."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from backend.analytics.deduplication import estimate_unique_detections


def generate_pdf_report(
    job_id: str,
    video_metadata: dict[str, Any],
    frames: list[dict[str, Any]],
    detections_by_class: dict[str, int],
    output_dir: str | Path,
) -> str:
    """Generate a client-ready PDF report and return its filesystem path."""

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    pdf_path = destination / "inspection_report.pdf"

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
        title="SkyVision-DL Inspection Report",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#123f54"),
        alignment=TA_LEFT,
        spaceAfter=8,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#52616f"),
        spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#123f54"),
        spaceBefore=8,
        spaceAfter=8,
    )
    normal_style = styles["BodyText"]
    small_style = ParagraphStyle(
        "Small",
        parent=normal_style,
        fontSize=8,
        leading=10,
        alignment=TA_LEFT,
    )
    badge_style = ParagraphStyle(
        "Badge",
        parent=normal_style,
        fontSize=9,
        leading=11,
        alignment=TA_CENTER,
        textColor=colors.white,
    )

    total_detections = sum(len(frame.get("detections", [])) for frame in frames)
    deduplication_summary = estimate_unique_detections(
        [
            {
                **detection,
                "frame_width": frame.get("frame_width") or video_metadata.get("width"),
                "frame_height": frame.get("frame_height") or video_metadata.get("height"),
                "frame_index": frame.get("frame_index"),
                "frame_sequence_index": frame.get("frame_sequence_index"),
                "timestamp_seconds": frame.get("timestamp_seconds"),
                "image_path": frame.get("image_path"),
                "annotated_image_path": frame.get("annotated_image_path"),
            }
            for frame in frames
            for detection in frame.get("detections", [])
        ]
    )
    priority_items = deduplication_summary["maintenance_priority_list"]
    media_type = str(video_metadata.get("media_type", "video")).title()
    inspection_metadata = video_metadata.get("inspection_metadata") or {}
    report_title = str(inspection_metadata.get("report_title") or "SkyVision-DL Maintenance Action Report")
    duration_label = "Video duration seconds" if media_type.lower() == "video" else "Uploaded image reviewed"
    duration_value = _format_value(video_metadata.get("duration_seconds")) if media_type.lower() == "video" else "Single image"
    generated_at = datetime.now(timezone.utc)

    story: list[Any] = [
        _cover_band(job_id, media_type, generated_at, deduplication_summary, badge_style),
        Paragraph(report_title, title_style),
        Paragraph(
            "Panel-level defect findings are grouped for maintenance planning. Raw frame observations are retained as supporting evidence.",
            subtitle_style,
        ),
        Spacer(1, 0.08 * inch),
    ]

    metadata_rows = _inspection_metadata_rows(inspection_metadata)
    if metadata_rows:
        story.extend([Paragraph("Inspection Details", heading_style), _table(metadata_rows), Spacer(1, 0.14 * inch)])

    summary_rows = [
        ["Maintenance Decision", "Result"],
        ["Actual glass panels needing attention", str(deduplication_summary["defect_areas_needing_attention"])],
        ["Immediate safety review", str(deduplication_summary["critical_defects"])],
        ["Repair-priority areas", str(deduplication_summary["high_defects"])],
        ["Reliability for maintenance planning", str(deduplication_summary["report_reliability"])],
    ]
    story.extend([Paragraph("Executive Summary", heading_style), _table(summary_rows), Spacer(1, 0.14 * inch)])

    evidence_rows = [
        ["Evidence Item", "Value"],
        ["Frames reviewed", str(len(frames))],
        ["Raw AI frame observations", str(total_detections)],
        ["Repeated observations grouped", str(deduplication_summary["repeated_frame_detections"])],
        ["Media type", media_type],
        [duration_label, duration_value],
        ["Resolution", _resolution(video_metadata)],
    ]
    story.extend([Paragraph("Inspection Evidence", heading_style), _table(evidence_rows), Spacer(1, 0.14 * inch)])

    priority_rows: list[list[Any]] = [["Priority", "Defect", "Severity", "Location", "Evidence", "Recommended Action"]]
    if priority_items:
        for item in priority_items:
            priority_rows.append(
                [
                    str(item["priority"]),
                    str(item["class_name"]).replace("_", " ").title(),
                    str(item["severity"]),
                    str(item["approximate_location"]).title(),
                    f"{item['evidence_frame_count']} frame(s), {float(item['confidence']):.2f} confidence",
                    Paragraph(str(item["recommended_action"]), small_style),
                ]
            )
    else:
        priority_rows.append(["-", "No defects detected", "-", "-", "-", "No maintenance action required from this run."])
    story.extend(
        [
            Paragraph("Repair Priority Queue", heading_style),
            _table(priority_rows, font_size=8),
            Spacer(1, 0.18 * inch),
        ]
    )

    class_rows = [["Defect Class", "Count"]]
    if detections_by_class:
        class_rows.extend([[name, str(count)] for name, count in sorted(detections_by_class.items())])
    else:
        class_rows.append(["No defects detected", "0"])
    story.extend([Paragraph("Defect Class Summary", heading_style), _table(class_rows), Spacer(1, 0.18 * inch)])

    story.append(Paragraph("Frame Evidence", heading_style))
    for index, frame in enumerate(frames, start=1):
        story.extend(_frame_section(index, frame, small_style))
        if index < len(frames):
            story.append(Spacer(1, 0.12 * inch))

    doc.build(story)
    return str(pdf_path)


def _frame_section(index: int, frame: dict[str, Any], small_style: ParagraphStyle) -> list[Any]:
    elements: list[Any] = [Paragraph(f"Frame {index}", small_style)]
    image_path = _optional_path(frame.get("annotated_image_path") or frame.get("image_path"))
    heatmap_path = _optional_path(frame.get("heatmap_image_path"))

    if image_path and image_path.exists():
        elements.append(Paragraph("Annotated defect view", small_style))
        elements.append(_image(str(image_path)))
    else:
        elements.append(Paragraph("Annotated image not available.", small_style))

    if heatmap_path and heatmap_path.exists():
        elements.append(Paragraph("Defect attention heatmap", small_style))
        elements.append(_image(str(heatmap_path)))

    detections = frame.get("detections", [])
    if detections:
        rows = [["Class", "Confidence", "Bounding Box"]]
        for detection in detections:
            rows.append(
                [
                    str(detection.get("class_name", "")),
                    f"{float(detection.get('confidence', 0.0)):.2f}",
                    _format_bbox(detection.get("bbox_xyxy", [])),
                ]
            )
        elements.append(_table(rows, font_size=8))
    else:
        elements.append(Paragraph("No defects detected in this frame.", small_style))

    if index % 3 == 0:
        elements.append(PageBreak())
    return elements


def _table(rows: list[list[Any]], font_size: int = 9) -> Table:
    table = Table(rows, hAlign="LEFT", repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f1f2")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17202a")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b8c5d3")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _inspection_metadata_rows(metadata: dict[str, Any]) -> list[list[Any]]:
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
    rows = [["Inspection Field", "Value"]]
    rows.extend([[label, str(metadata.get(key))] for key, label in labels.items() if metadata.get(key)])
    return rows if len(rows) > 1 else []


def _cover_band(
    job_id: str,
    media_type: str,
    generated_at: datetime,
    summary: dict[str, Any],
    badge_style: ParagraphStyle,
) -> Table:
    reliability = str(summary.get("report_reliability") or "Review")
    severity_label = "Immediate Review" if summary.get("critical_defects") else "Repair Planning" if summary.get("high_defects") else "Routine Review"
    rows = [
        [
            Paragraph("<b>SkyVision-DL</b><br/>Facade Glass Inspection", badge_style),
            f"Job ID: {job_id}",
            f"Media: {media_type}",
            f"Reliability: {reliability}",
            severity_label,
        ],
        [
            "",
            f"Generated: {generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
            f"Attention areas: {summary.get('defect_areas_needing_attention', 0)}",
            f"Critical: {summary.get('critical_defects', 0)}",
            f"High: {summary.get('high_defects', 0)}",
        ],
    ]
    table = Table(rows, colWidths=[1.55 * inch, 1.55 * inch, 1.25 * inch, 1.25 * inch, 1.25 * inch])
    table.setStyle(
        TableStyle(
            [
                ("SPAN", (0, 0), (0, 1)),
                ("BACKGROUND", (0, 0), (0, 1), colors.HexColor("#123f54")),
                ("BACKGROUND", (1, 0), (-1, -1), colors.HexColor("#e8f1f2")),
                ("TEXTCOLOR", (1, 0), (-1, -1), colors.HexColor("#17202a")),
                ("FONTNAME", (1, 0), (-1, -1), "Helvetica-Bold"),
                ("FONTSIZE", (1, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b8c5d3")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def _image(path: str) -> Image:
    image = Image(path)
    max_width = 6.8 * inch
    max_height = 3.4 * inch
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight, 1.0)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    return image


def _optional_path(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if str(path) in {"", "."}:
        return None
    return path


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _resolution(video_metadata: dict[str, Any]) -> str:
    width = video_metadata.get("width")
    height = video_metadata.get("height")
    if width and height:
        return f"{width} x {height}"
    return "-"


def _format_bbox(bbox: Any) -> str:
    if not isinstance(bbox, list | tuple):
        return "-"
    return ", ".join(f"{float(value):.1f}" for value in bbox)
