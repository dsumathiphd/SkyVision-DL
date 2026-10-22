# SkyVision-DL Deployment Guide

## Production Readiness Checklist

- Use PostgreSQL for deployed environments.
- Set a strong `SKYVISION_JWT_SECRET`.
- Store trained YOLO weights at `models/weights/best.pt` or set `SKYVISION_MODEL_PATH`.
- Keep `backend/storage` on persistent storage.
- Run the app behind HTTPS using a reverse proxy.
- Restrict CORS before public deployment.
- Back up the database and generated inspection reports.
- Review `backend/storage/logs/skyvision.log` during local validation and configure centralized logging for hosted deployments.

## Environment Variables

```text
DATABASE_URL=postgresql+psycopg://user:password@host:5432/skyvision
SKYVISION_MODEL_PATH=models/weights/best.pt
SKYVISION_JWT_SECRET=replace-with-production-secret
SKYVISION_JWT_ALGORITHM=HS256
SKYVISION_ACCESS_TOKEN_EXPIRE_MINUTES=120
SKYVISION_ALLOWED_ORIGINS=https://your-domain.example
SKYVISION_MAX_VIDEO_UPLOAD_MB=500
SKYVISION_MAX_IMAGE_UPLOAD_MB=50
```

## Local Run

```powershell
cd C:\Users\dsuma\OneDrive\Documents\SkyVision-DL
python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

## Deployment Notes

The current application is a single FastAPI service serving both backend APIs and the static frontend. Background inspection processing uses FastAPI background tasks and local storage. For higher traffic deployments, move background processing to a worker queue and object storage while keeping the same API contract.
