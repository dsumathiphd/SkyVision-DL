"""Database models for SkyVision-DL inspection jobs."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.mutable import MutableDict, MutableList
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from backend.database.connection import Base


def json_column(default: Any = None) -> Mapped[Any]:
    return mapped_column(MutableDict.as_mutable(JSON().with_variant(JSONB, "postgresql")), default=default)


def json_list_column(default: Any = None) -> Mapped[Any]:
    return mapped_column(MutableList.as_mutable(JSON().with_variant(JSONB, "postgresql")), default=default)


class UserRole(StrEnum):
    ADMIN = "admin"
    INSPECTOR = "inspector"
    VIEWER = "viewer"
    MAINTENANCE_MANAGER = "maintenance_manager"


class JobStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ReportType(StrEnum):
    JSON = "json"
    HTML = "html"
    PDF = "pdf"


class AuditAction(StrEnum):
    LOGIN = "login"
    REGISTER = "register"
    UPLOAD = "upload"
    INSPECTION_CREATED = "inspection_created"
    REPORT_DOWNLOAD = "report_download"
    USER_ROLE_UPDATED = "user_role_updated"
    USER_STATUS_UPDATED = "user_status_updated"


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    password_hash: Mapped[str | None] = mapped_column(String(255), nullable=True)
    role: Mapped[str] = mapped_column(String(32), default=UserRole.INSPECTOR.value, nullable=False)
    is_active: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    inspection_jobs: Mapped[list[InspectionJob]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )
    audit_logs: Mapped[list[AuditLog]] = relationship(back_populates="user")


class AuditLog(Base, TimestampMixin):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    action: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    entity_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    entity_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict[str, Any] | None] = json_column(default=dict)

    user: Mapped[User | None] = relationship(back_populates="audit_logs")


class InspectionJob(Base, TimestampMixin):
    __tablename__ = "inspection_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default=JobStatus.QUEUED.value, index=True, nullable=False)
    progress_percent: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence_threshold: Mapped[float] = mapped_column(Float, default=0.15, nullable=False)
    sample_fps: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    device: Mapped[str] = mapped_column(String(32), default="0", nullable=False)
    metadata_json: Mapped[dict[str, Any] | None] = json_column(default=dict)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped[User | None] = relationship(back_populates="inspection_jobs")
    uploaded_video: Mapped[UploadedVideo | None] = relationship(
        back_populates="inspection_job",
        cascade="all, delete-orphan",
        uselist=False,
    )
    extracted_frames: Mapped[list[ExtractedFrame]] = relationship(
        back_populates="inspection_job",
        cascade="all, delete-orphan",
    )
    defect_detections: Mapped[list[DefectDetection]] = relationship(
        back_populates="inspection_job",
        cascade="all, delete-orphan",
    )
    reports: Mapped[list[Report]] = relationship(
        back_populates="inspection_job",
        cascade="all, delete-orphan",
    )


class UploadedVideo(Base, TimestampMixin):
    __tablename__ = "uploaded_videos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    inspection_job_id: Mapped[int] = mapped_column(
        ForeignKey("inspection_jobs.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_path: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    file_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    frame_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    inspection_job: Mapped[InspectionJob] = relationship(back_populates="uploaded_video")


class ExtractedFrame(Base, TimestampMixin):
    __tablename__ = "extracted_frames"
    __table_args__ = (
        UniqueConstraint("inspection_job_id", "frame_index", name="uq_extracted_frames_job_frame_index"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    inspection_job_id: Mapped[int] = mapped_column(ForeignKey("inspection_jobs.id", ondelete="CASCADE"), nullable=False)
    frame_index: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    image_path: Mapped[str] = mapped_column(Text, nullable=False)
    annotated_image_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)

    inspection_job: Mapped[InspectionJob] = relationship(back_populates="extracted_frames")
    defect_detections: Mapped[list[DefectDetection]] = relationship(
        back_populates="frame",
        cascade="all, delete-orphan",
    )


class DefectDetection(Base, TimestampMixin):
    __tablename__ = "defect_detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    inspection_job_id: Mapped[int] = mapped_column(ForeignKey("inspection_jobs.id", ondelete="CASCADE"), nullable=False)
    frame_id: Mapped[int] = mapped_column(ForeignKey("extracted_frames.id", ondelete="CASCADE"), nullable=False)
    class_id: Mapped[int] = mapped_column(Integer, nullable=False)
    class_name: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_xyxy: Mapped[list[float]] = json_list_column(default=list)
    severity: Mapped[str | None] = mapped_column(String(32), nullable=True)
    metadata_json: Mapped[dict[str, Any] | None] = json_column(default=dict)

    inspection_job: Mapped[InspectionJob] = relationship(back_populates="defect_detections")
    frame: Mapped[ExtractedFrame] = relationship(back_populates="defect_detections")


class Report(Base, TimestampMixin):
    __tablename__ = "reports"
    __table_args__ = (
        UniqueConstraint("inspection_job_id", "report_type", name="uq_reports_job_report_type"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    inspection_job_id: Mapped[int] = mapped_column(ForeignKey("inspection_jobs.id", ondelete="CASCADE"), nullable=False)
    report_type: Mapped[str] = mapped_column(String(32), nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    summary_json: Mapped[dict[str, Any] | None] = json_column(default=dict)

    inspection_job: Mapped[InspectionJob] = relationship(back_populates="reports")
