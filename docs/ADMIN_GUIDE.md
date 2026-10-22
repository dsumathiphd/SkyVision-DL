# SkyVision-DL Admin Guide

## Administrator Role

Administrators can manage users, review system status, update local application defaults, and download local backup packages.

## User Management

The User Management panel supports:

- Role changes for registered users
- Activating or deactivating users
- Refreshing the local user list

Available roles:

- Admin
- Inspector
- Maintenance Manager
- Viewer

## Application Settings

The Application Settings panel controls local defaults for new inspections:

- Default Sample FPS
- Default Confidence
- Default Device
- Default Model Path
- Max Video Upload MB
- Max Image Upload MB

Saved settings are stored locally under `backend/storage/config/settings.json`.

## System Status

The System Status panel reports:

- Application version
- Logged-in user role
- Database connectivity
- Model file availability
- Storage capacity
- Upload limits
- Configured CORS origins

## Local Backup

Use `Download Backup` to create a ZIP package containing:

- Local SQLite database
- Report files
- Logs
- Local settings
- Project documentation
- Database schema files

The backup intentionally excludes uploaded videos, model weights, and training datasets to keep packages manageable.

## Production Notes

Before external deployment:

- Replace the JWT secret.
- Move from local SQLite to PostgreSQL.
- Configure HTTPS through a reverse proxy.
- Store media/report artifacts in managed object storage.
- Run API tests before release.
- Back up the database and report files.

