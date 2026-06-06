from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "docs" / "backend_ops_runbook.md"


def test_backend_ops_runbook_documents_current_ops_commands() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")

    required_fragments = [
        "scripts\\check_deployment_readiness.py",
        "uvicorn campus_support_agent.main:app",
        "/api/v1/ops/data-governance",
        "/api/v1/ops/database-integrity",
        "scripts\\maintain_sqlite_database.py",
        "scripts\\export_redacted_audit_package.py",
        "scripts\\export_care_queue_snapshot.py",
        "/api/v1/analytics/care-queue/actions/batch",
        "/api/v1/ops/audit-events",
        "data/care_queue_exports/",
    ]
    for fragment in required_fragments:
        assert fragment in text
