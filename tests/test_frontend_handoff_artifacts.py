from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_frontend_handoff_client_covers_core_endpoints() -> None:
    client = (ROOT / "frontend_handoff" / "campusSupportApi.ts").read_text(encoding="utf-8")

    for endpoint in [
        "/health",
        "/api/v1/frontend/contract",
        "/api/v1/ops/readiness",
        "/api/v1/support/text",
        "/api/v1/support/audio",
        "/api/v1/analytics/care-queue",
        "/api/v1/sessions/",
    ]:
        assert endpoint in client

    for field in [
        "reply_text",
        "risk",
        "entropy",
        "entropy_reduction",
        "safety",
        "campus_resources",
        "system_flags",
    ]:
        assert field in client


def test_frontend_integration_guide_mentions_client_artifact() -> None:
    guide = (ROOT / "docs" / "frontend_integration_guide.md").read_text(encoding="utf-8")

    assert "frontend_handoff/campusSupportApi.ts" in guide
