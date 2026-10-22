"""Authentication API routes for SkyVision-DL."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.auth.dependencies import get_current_user, require_admin
from backend.auth.schemas import TokenResponse, UserAdminUpdate, UserCreate, UserLogin, UserRead
from backend.auth.security import create_access_token, hash_password, verify_password
from backend.database.connection import get_db
from backend.models.inspection_models import AuditAction, User, UserRole
from backend.services.audit_logger import record_audit_event


router = APIRouter(prefix="/api/auth", tags=["authentication"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register_user(payload: UserCreate, db: Session = Depends(get_db)) -> TokenResponse:
    existing_user = db.scalar(select(User).where(User.email == payload.email.lower()))
    if existing_user is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email is already registered.")

    user_count = db.scalar(select(func.count(User.id))) or 0
    user = User(
        email=payload.email.lower(),
        full_name=payload.full_name,
        password_hash=hash_password(payload.password),
        role=UserRole.ADMIN.value if user_count == 0 else UserRole.INSPECTOR.value,
        is_active=1,
    )
    db.add(user)
    db.flush()
    record_audit_event(
        db,
        user_id=user.id,
        action=AuditAction.REGISTER.value,
        entity_type="user",
        entity_id=str(user.id),
        message="User account registered.",
        metadata={"email": user.email, "role": user.role},
    )
    db.commit()
    db.refresh(user)

    access_token, expires_in = create_access_token(
        subject=str(user.id),
        claims={"email": user.email, "role": user.role},
    )
    return TokenResponse(access_token=access_token, expires_in=expires_in, user=to_user_read(user))


@router.post("/login", response_model=TokenResponse)
def login_user(payload: UserLogin, db: Session = Depends(get_db)) -> TokenResponse:
    user = db.scalar(select(User).where(User.email == payload.email.lower()))
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password.")
    if not bool(user.is_active):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User account is inactive.")

    record_audit_event(
        db,
        user_id=user.id,
        action=AuditAction.LOGIN.value,
        entity_type="user",
        entity_id=str(user.id),
        message="User logged in.",
        metadata={"email": user.email, "role": user.role},
    )
    db.commit()

    access_token, expires_in = create_access_token(
        subject=str(user.id),
        claims={"email": user.email, "role": user.role},
    )
    return TokenResponse(access_token=access_token, expires_in=expires_in, user=to_user_read(user))


@router.get("/me", response_model=UserRead)
def read_current_user(current_user: User = Depends(get_current_user)) -> UserRead:
    return to_user_read(current_user)


@router.get("/admin/users", response_model=list[UserRead])
def list_users(
    _: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> list[UserRead]:
    users = db.scalars(select(User).order_by(User.created_at.desc(), User.id.desc())).all()
    return [to_user_read(user) for user in users]


@router.patch("/admin/users/{user_id}", response_model=UserRead)
def update_user_admin(
    user_id: int,
    payload: UserAdminUpdate,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> UserRead:
    user = db.scalar(select(User).where(User.id == user_id))
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    if payload.role is not None:
        allowed_roles = {role.value for role in UserRole}
        if payload.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid role. Allowed roles: {', '.join(sorted(allowed_roles))}.",
            )
        user.role = payload.role
        record_audit_event(
            db,
            user_id=admin_user.id,
            action=AuditAction.USER_ROLE_UPDATED.value,
            entity_type="user",
            entity_id=str(user.id),
            message="User role updated.",
            metadata={"target_email": user.email, "role": user.role},
        )

    if payload.is_active is not None:
        user.is_active = 1 if payload.is_active else 0
        record_audit_event(
            db,
            user_id=admin_user.id,
            action=AuditAction.USER_STATUS_UPDATED.value,
            entity_type="user",
            entity_id=str(user.id),
            message="User active status updated.",
            metadata={"target_email": user.email, "is_active": bool(user.is_active)},
        )

    db.commit()
    db.refresh(user)
    return to_user_read(user)


def to_user_read(user: User) -> UserRead:
    return UserRead(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        is_active=bool(user.is_active),
    )

