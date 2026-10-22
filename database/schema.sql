-- SkyVision-DL database schema.
-- Designed for SQLite development and PostgreSQL production with minor type adaptation.

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    full_name VARCHAR(255),
    password_hash VARCHAR(255),
    role VARCHAR(32) NOT NULL DEFAULT 'inspector',
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (role IN ('admin', 'inspector', 'viewer', 'maintenance_manager')),
    CHECK (is_active IN (0, 1))
);

CREATE INDEX IF NOT EXISTS ix_users_email ON users (email);

CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(64) NOT NULL,
    entity_type VARCHAR(64),
    entity_id VARCHAR(128),
    message TEXT,
    metadata_json JSON DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_audit_logs_action ON audit_logs (action);

CREATE TABLE IF NOT EXISTS inspection_jobs (
    id INTEGER PRIMARY KEY,
    job_id VARCHAR(64) NOT NULL UNIQUE,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'queued',
    progress_percent INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    model_path TEXT,
    confidence_threshold FLOAT NOT NULL DEFAULT 0.15,
    sample_fps FLOAT NOT NULL DEFAULT 1.0,
    device VARCHAR(32) NOT NULL DEFAULT '0',
    metadata_json JSON DEFAULT '{}',
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (status IN ('queued', 'processing', 'completed', 'failed')),
    CHECK (progress_percent >= 0 AND progress_percent <= 100)
);

CREATE INDEX IF NOT EXISTS ix_inspection_jobs_job_id ON inspection_jobs (job_id);
CREATE INDEX IF NOT EXISTS ix_inspection_jobs_status ON inspection_jobs (status);

CREATE TABLE IF NOT EXISTS uploaded_videos (
    id INTEGER PRIMARY KEY,
    inspection_job_id INTEGER NOT NULL UNIQUE REFERENCES inspection_jobs(id) ON DELETE CASCADE,
    original_filename VARCHAR(255) NOT NULL,
    stored_path TEXT NOT NULL,
    content_type VARCHAR(128),
    file_size_bytes INTEGER,
    fps FLOAT,
    frame_count INTEGER,
    width INTEGER,
    height INTEGER,
    duration_seconds FLOAT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS extracted_frames (
    id INTEGER PRIMARY KEY,
    inspection_job_id INTEGER NOT NULL REFERENCES inspection_jobs(id) ON DELETE CASCADE,
    frame_index INTEGER NOT NULL,
    timestamp_seconds FLOAT NOT NULL,
    image_path TEXT NOT NULL,
    annotated_image_path TEXT,
    width INTEGER,
    height INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_extracted_frames_job_frame_index UNIQUE (inspection_job_id, frame_index)
);

CREATE TABLE IF NOT EXISTS defect_detections (
    id INTEGER PRIMARY KEY,
    inspection_job_id INTEGER NOT NULL REFERENCES inspection_jobs(id) ON DELETE CASCADE,
    frame_id INTEGER NOT NULL REFERENCES extracted_frames(id) ON DELETE CASCADE,
    class_id INTEGER NOT NULL,
    class_name VARCHAR(128) NOT NULL,
    confidence FLOAT NOT NULL,
    bbox_xyxy JSON NOT NULL DEFAULT '[]',
    severity VARCHAR(32),
    metadata_json JSON DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_defect_detections_class_name ON defect_detections (class_name);

CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY,
    inspection_job_id INTEGER NOT NULL REFERENCES inspection_jobs(id) ON DELETE CASCADE,
    report_type VARCHAR(32) NOT NULL,
    file_path TEXT NOT NULL,
    summary_json JSON DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_reports_job_report_type UNIQUE (inspection_job_id, report_type),
    CHECK (report_type IN ('json', 'html', 'pdf'))
);
