"""Contract checks for the SkyVision-DL API surface."""

from __future__ import annotations

from backend.api.auth import router as auth_router
from backend.api.main import app, compact_metadata, default_local_settings


def route_paths() -> set[str]:
    return {getattr(route, "path", "") for route in app.routes}


def auth_route_paths() -> set[str]:
    return {getattr(route, "path", "") for route in auth_router.routes}


def test_core_inspection_routes_are_registered() -> None:
    paths = route_paths()
    assert "/api/inspections" in paths
    assert "/api/inspections/history" in paths
    assert "/api/inspections/history.csv" in paths
    assert "/api/jobs/{job_id}" in paths
    assert "/api/jobs/{job_id}/result" in paths
    assert "/api/jobs/{job_id}/report/pdf" in paths
    assert "/api/jobs/{job_id}/report/package.zip" in paths


def test_enterprise_control_routes_are_registered() -> None:
    paths = route_paths()
    assert "/api/settings" in paths
    assert "/api/system/status" in paths
    assert "/api/system/backup.zip" in paths


def test_auth_routes_are_available() -> None:
    paths = auth_route_paths()
    assert "/api/auth/register" in paths
    assert "/api/auth/login" in paths
    assert "/api/auth/me" in paths


def test_default_settings_contract() -> None:
    settings = default_local_settings()
    assert settings["default_sample_fps"] > 0
    assert 0 < settings["default_confidence"] < 1
    assert settings["default_device"]
    assert settings["default_model_path"]
    assert settings["max_video_upload_mb"] > 0
    assert settings["max_image_upload_mb"] > 0


def test_compact_metadata_removes_empty_values() -> None:
    assert compact_metadata({"site": " Tower A ", "floor": "", "notes": None}) == {"site": "Tower A"}
