"""Local audit logging utilities for SkyVision-DL."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session

from backend.models.inspection_models import AuditLog


logger = logging.getLogger(__name__)


def record_audit_event(
    db: Session,
    *,
    user_id: int | None,
    action: str,
    entity_type: str | None = None,
    entity_id: str | None = None,
    message: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist an audit event without interrupting the main user workflow."""

    try:
        db.add(
            AuditLog(
                user_id=user_id,
                action=action,
                entity_type=entity_type,
                entity_id=entity_id,
                message=message,
                metadata_json=metadata or {},
            )
        )
    except Exception:
        logger.exception("Failed to stage audit event action=%s entity=%s", action, entity_id)
