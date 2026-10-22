"""SQLAlchemy database connection utilities for SkyVision-DL."""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SQLITE_PATH = PROJECT_ROOT / "database" / "skyvision.db"
DEFAULT_DATABASE_URL = f"sqlite:///{DEFAULT_SQLITE_PATH.as_posix()}"


class Base(DeclarativeBase):
    """Base class for all ORM models."""


def get_database_url() -> str:
    """Return the configured database URL.

    Defaults to a local SQLite database for development. Set DATABASE_URL to a
    PostgreSQL URL such as postgresql+psycopg://user:pass@host:5432/skyvision
    for production-style deployments.
    """

    return os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


def build_engine(database_url: str | None = None) -> Engine:
    url = database_url or get_database_url()

    if url.startswith("sqlite"):
        DEFAULT_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
        return create_engine(
            url,
            connect_args={"check_same_thread": False},
            pool_pre_ping=True,
            future=True,
        )

    return create_engine(url, pool_pre_ping=True, future=True)


engine = build_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_database() -> None:
    """Create database tables for local development usage."""

    import backend.models.inspection_models  # noqa: F401

    Base.metadata.create_all(bind=engine)

