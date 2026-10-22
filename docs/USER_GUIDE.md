# SkyVision-DL User Guide

## Purpose

SkyVision-DL helps maintenance teams inspect facade glass panels from uploaded videos or images, identify visible defect areas, and generate inspection reports for repair planning.

## Login

1. Open `http://127.0.0.1:8000`.
2. Login with your registered account.
3. Register a new account if access has not been created yet.

## Run An Inspection

1. Choose `Video Analysis` for MP4/AVI inspection footage or `Image Analysis` for JPG/PNG facade images.
2. Enter the inspection metadata:
   - Organization
   - Project
   - Report title
   - Inspector
   - Building
   - Floor or level
   - Facade side
   - Location
   - Inspection date
   - Remarks
3. Upload the media file.
4. Review the run settings:
   - Sample FPS controls video frame sampling.
   - Confidence controls detection sensitivity.
   - Device selects CPU/GPU execution.
   - Model Path selects the YOLO weight file.
5. Select `Run Inspection`.

## Review Results

After processing finishes, the dashboard shows:

- Panels needing attention
- Immediate safety review count
- Repair priority count
- Planning reliability
- Most observed defect
- Frame-level annotated images and heatmaps

Priority labels are intended for maintenance planning:

- Critical: immediate safety review
- High: repair planning required
- Medium: inspect and monitor
- Low: cosmetic or routine review

## Download Reports

Completed inspections support:

- JSON Report
- HTML Report
- PDF Report
- ZIP Package

Use the PDF for maintenance review and the ZIP package when sharing the full report bundle.

## Inspection History

The history table shows previous inspections for the logged-in user. Available actions:

- `Open`: reload completed results in the dashboard.
- `PDF`: download the PDF report.
- `ZIP`: download the report package.
- `Archive`: hide completed or failed inspections from the active list.
- `Restore`: return archived inspections to the active list.
- `Export CSV`: download the inspection history as a spreadsheet-friendly file.

