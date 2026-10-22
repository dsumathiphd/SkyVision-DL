"""Password hashing and JWT utilities for SkyVision-DL."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from pwdlib import PasswordHash


JWT_ALGORITHM = os.getenv("SKYVISION_JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("SKYVISION_ACCESS_TOKEN_EXPIRE_MINUTES", "120"))
JWT_SECRET = os.getenv("SKYVISION_JWT_SECRET", "skyvision-local-development-secret-change-me")

password_hash = PasswordHash.recommended()


class AuthenticationError(RuntimeError):
    """Raised when authentication data is invalid."""


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(password: str, password_hash_value: str | None) -> bool:
    if not password_hash_value:
        return False
    return password_hash.verify(password, password_hash_value)


def create_access_token(subject: str, claims: dict[str, Any] | None = None) -> tuple[str, int]:
    expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    expires_at = datetime.now(timezone.utc) + expires_delta
    payload: dict[str, Any] = {
        "sub": subject,
        "exp": expires_at,
        "iat": datetime.now(timezone.utc),
    }
    if claims:
        payload.update(claims)

    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token, int(expires_delta.total_seconds())


def decode_access_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError as exc:
        raise AuthenticationError("Invalid or expired access token.") from exc

