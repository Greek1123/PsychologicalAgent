from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, Settings


VALID_LLM_PROVIDERS = {"mock", "openai_compatible", "local_checkpoint"}
VALID_STT_PROVIDERS = {"disabled", "mock", "openai_compatible"}


@dataclass(slots=True)
class ReadinessCheck:
    name: str
    status: str
    message: str
    details: dict[str, Any]


def build_deployment_readiness(settings: Settings) -> dict[str, Any]:
    checks = [
        _provider_check("llm_provider", settings.llm_provider, VALID_LLM_PROVIDERS),
        _provider_check("stt_provider", settings.stt_provider, VALID_STT_PROVIDERS),
        _path_parent_check("database_path", settings.database_path),
        _path_parent_check("log_file_path", settings.log_file_path),
        _file_exists_check("campus_kb_path", settings.campus_kb_path),
        _frontend_origins_check(settings),
        _admin_api_key_check(settings),
        _local_checkpoint_check(settings),
        _python_runtime_check(),
    ]
    status_rank = {"pass": 0, "warn": 1, "fail": 2}
    overall = max(checks, key=lambda item: status_rank[item.status]).status
    return {
        "status": "ready" if overall == "pass" else "degraded" if overall == "warn" else "blocked",
        "app_env": settings.app_env,
        "app_port": settings.app_port,
        "project_root": str(PROJECT_ROOT),
        "checks": [asdict(check) for check in checks],
        "summary": _summarize_checks(checks),
    }


def _provider_check(name: str, value: str, allowed: set[str]) -> ReadinessCheck:
    normalized = value.strip().lower()
    if normalized in allowed:
        return ReadinessCheck(
            name=name,
            status="pass",
            message=f"{name} is supported.",
            details={"value": value, "allowed": sorted(allowed)},
        )
    return ReadinessCheck(
        name=name,
        status="fail",
        message=f"{name} is not supported.",
        details={"value": value, "allowed": sorted(allowed)},
    )


def _path_parent_check(name: str, value: str) -> ReadinessCheck:
    path = _resolve_project_path(value)
    parent = path.parent
    if parent.exists() and parent.is_dir():
        return ReadinessCheck(
            name=name,
            status="pass",
            message=f"{name} parent directory exists.",
            details={"path": str(path), "parent": str(parent)},
        )
    return ReadinessCheck(
        name=name,
        status="fail",
        message=f"{name} parent directory does not exist.",
        details={"path": str(path), "parent": str(parent)},
    )


def _file_exists_check(name: str, value: str) -> ReadinessCheck:
    path = _resolve_project_path(value)
    if path.exists() and path.is_file():
        return ReadinessCheck(
            name=name,
            status="pass",
            message=f"{name} exists.",
            details={"path": str(path)},
        )
    return ReadinessCheck(
        name=name,
        status="fail",
        message=f"{name} is missing.",
        details={"path": str(path)},
    )


def _local_checkpoint_check(settings: Settings) -> ReadinessCheck:
    if settings.llm_provider.strip().lower() != "local_checkpoint":
        return ReadinessCheck(
            name="local_checkpoint",
            status="pass",
            message="Local checkpoint is not required for the current LLM provider.",
            details={"llm_provider": settings.llm_provider},
        )

    checkpoint_path = _resolve_project_path(settings.local_checkpoint_path)
    base_model_path = _resolve_project_path(settings.local_base_model_path)
    missing = []
    if not checkpoint_path.exists():
        missing.append("LOCAL_CHECKPOINT_PATH")
    if not base_model_path.exists():
        missing.append("LOCAL_BASE_MODEL_PATH")
    status = "pass" if not missing else "fail"
    return ReadinessCheck(
        name="local_checkpoint",
        status=status,
        message="Local checkpoint paths are available." if not missing else "Local checkpoint paths are missing.",
        details={
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_exists": checkpoint_path.exists(),
            "base_model_path": str(base_model_path),
            "base_model_exists": base_model_path.exists(),
            "missing": missing,
        },
    )


def _frontend_origins_check(settings: Settings) -> ReadinessCheck:
    origins = [origin.strip() for origin in settings.frontend_allowed_origins if origin.strip()]
    if not origins:
        return ReadinessCheck(
            name="frontend_allowed_origins",
            status="fail",
            message="No frontend origins are configured.",
            details={"env": "FRONTEND_ALLOWED_ORIGINS", "origins": origins},
        )
    has_wildcard = "*" in origins
    production_like = settings.app_env.strip().lower() in {"prod", "production"}
    if has_wildcard and production_like:
        return ReadinessCheck(
            name="frontend_allowed_origins",
            status="warn",
            message="Wildcard CORS origin should not be used in production.",
            details={"env": "FRONTEND_ALLOWED_ORIGINS", "origins": origins},
        )
    return ReadinessCheck(
        name="frontend_allowed_origins",
        status="pass",
        message="Frontend CORS origins are configured.",
        details={"env": "FRONTEND_ALLOWED_ORIGINS", "origins": origins},
    )


def _admin_api_key_check(settings: Settings) -> ReadinessCheck:
    key = settings.admin_api_key.strip()
    production_like = settings.app_env.strip().lower() in {"prod", "production"}
    required = production_like or settings.require_admin_api_key or bool(key)
    if not required:
        return ReadinessCheck(
            name="admin_api_key",
            status="pass",
            message="Admin API key is optional in local development.",
            details={
                "env": "ADMIN_API_KEY",
                "required": False,
                "configured": False,
            },
        )
    if not key:
        return ReadinessCheck(
            name="admin_api_key",
            status="fail",
            message="Admin API key is required but not configured.",
            details={
                "env": "ADMIN_API_KEY",
                "required": True,
                "configured": False,
            },
        )
    if len(key) < 16:
        return ReadinessCheck(
            name="admin_api_key",
            status="warn",
            message="Admin API key is configured but too short for deployment.",
            details={
                "env": "ADMIN_API_KEY",
                "required": True,
                "configured": True,
                "minimum_length": 16,
            },
        )
    return ReadinessCheck(
        name="admin_api_key",
        status="pass",
        message="Admin API key is configured for protected endpoints.",
        details={
            "env": "ADMIN_API_KEY",
            "required": True,
            "configured": True,
            "minimum_length": 16,
        },
    )


def _python_runtime_check() -> ReadinessCheck:
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    status = "pass" if sys.version_info >= (3, 10) else "fail"
    return ReadinessCheck(
        name="python_runtime",
        status=status,
        message="Python runtime is supported." if status == "pass" else "Python 3.10+ is required.",
        details={"version": version, "executable": sys.executable},
    )


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _summarize_checks(checks: list[ReadinessCheck]) -> dict[str, int]:
    summary = {"pass": 0, "warn": 0, "fail": 0}
    for check in checks:
        summary[check.status] += 1
    return summary
