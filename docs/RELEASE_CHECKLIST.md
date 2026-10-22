# SkyVision-DL Release Checklist

## Application

- [ ] Authentication verified for login and registration
- [ ] Admin role assigned to at least one user
- [ ] User roles reviewed
- [ ] Upload limits reviewed
- [ ] CORS origins configured
- [ ] Local settings saved
- [ ] System status panel healthy

## Model

- [ ] `models/weights/best.pt` present
- [ ] Model path confirmed in dashboard
- [ ] Confidence threshold reviewed
- [ ] Sample FPS reviewed for video runs
- [ ] Test image inspection completed
- [ ] Test video inspection completed

## Reports

- [ ] JSON report downloads correctly
- [ ] HTML report downloads correctly
- [ ] PDF report downloads correctly
- [ ] ZIP report package downloads correctly
- [ ] Report metadata appears correctly
- [ ] Heatmaps and annotated frames render correctly

## Data And Operations

- [ ] Inspection history refresh works
- [ ] Inspection history CSV export works
- [ ] Archive and restore actions work
- [ ] Local backup ZIP downloads correctly
- [ ] Logs are written under `backend/storage/logs`
- [ ] Database file is backed up

## Verification

- [ ] `python -m py_compile backend\api\main.py`
- [ ] `python -m py_compile tests\test_api_contract.py`
- [ ] `node --check frontend\app.js`
- [ ] `python -m pytest tests`

