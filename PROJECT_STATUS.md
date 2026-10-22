# SkyVision-DL Project Status

Date: October 13, 2026

## Current Scope

SkyVision-DL is a professional facade glass inspection application focused on maintenance-oriented defect review.

```text
Authentication
-> Video/Image Upload
-> Background Inspection Job
-> Frame Extraction
-> YOLOv8 Detection
-> Result Display
-> Inspection History
-> JSON / HTML / PDF / ZIP Reports
```

## Product Readiness Checklist

- [x] SQLite local development database
- [x] PostgreSQL-ready SQLAlchemy schema
- [x] User registration
- [x] User login
- [x] JWT-protected APIs
- [x] Video upload support for MP4 and AVI
- [x] Image upload support for JPG, JPEG, and PNG
- [x] Background inspection jobs
- [x] Job status polling
- [x] YOLOv8 inference integration
- [x] CPU fallback when GPU is unavailable
- [x] Annotated frame output
- [x] JSON report generation
- [x] HTML report generation
- [x] PDF report generation
- [x] Inspection history table
- [x] Open completed historical inspection results
- [x] Download PDF reports from history
- [x] Download ZIP report package from completed jobs
- [x] Loading states and duplicate-click protection
- [x] Clearer user-facing error messages
- [x] Run instructions in README
- [x] Deployment notes and environment configuration example
- [x] Maintenance-focused dashboard
- [x] Searchable inspection history
- [x] Model version metadata display
- [x] Priority severity rules for broken glass, crack, fracture, leakage, delamination, stains, and repair patches
- [x] Configurable CORS origins
- [x] Upload size and MIME validation
- [x] Local audit logging
- [x] Role foundation for admin, inspector, maintenance manager, and viewer
- [x] Admin user management panel
- [x] Archive and restore inspections from history
- [x] Report branding and inspection metadata
- [x] Structured local logging
- [x] System status panel
- [x] Local application settings panel
- [x] Inspection history CSV export
- [x] Local backup ZIP package
- [x] Backend API contract checks
- [x] User guide
- [x] Admin guide
- [x] Release checklist

## Recommended Run Settings

```text
Sample FPS: 1.0
Confidence: 0.35
Device: cpu
Model Path: models/weights/best.pt
```

## Current Model Notes

- `models/weights/best.pt` is the trained SkyVision-DL facade defect model.
- The current trained model is strongest for broken-glass detection.
- Additional labeled data is still needed for robust multi-class defect detection.
- Old failed jobs may remain visible in inspection history because they are valid historical records.

## Deferred Enterprise Extensions

- Billing
- Digital twin
- Enterprise admin dashboard
- Multi-tenant organization management
- Production cloud storage
- Advanced analytics
