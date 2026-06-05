from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from cleanup_expired_data import cleanup_expired_data


def test_cleanup_expired_data_dry_run_does_not_delete(tmp_path: Path) -> None:
    db_path = tmp_path / "cleanup.db"
    _seed_cleanup_db(db_path)

    result = cleanup_expired_data(
        db_path=db_path,
        session_data_retention_days=180,
        audit_log_retention_days=365,
        apply=False,
        now=datetime(2026, 6, 5, tzinfo=timezone.utc),
    )

    assert result["deleted_total"] == 0
    assert result["tables"]["support_responses"]["matched"] == 1
    assert _count_rows(db_path, "support_responses") == 2
    assert _count_rows(db_path, "human_interventions") == 2


def test_cleanup_expired_data_apply_deletes_only_expired_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "cleanup.db"
    _seed_cleanup_db(db_path)

    result = cleanup_expired_data(
        db_path=db_path,
        session_data_retention_days=180,
        audit_log_retention_days=365,
        apply=True,
        now=datetime(2026, 6, 5, tzinfo=timezone.utc),
    )

    assert result["deleted_total"] == 2
    assert result["tables"]["support_responses"]["deleted"] == 1
    assert result["tables"]["human_interventions"]["deleted"] == 1
    assert _count_rows(db_path, "support_responses") == 1
    assert _count_rows(db_path, "human_interventions") == 1


def _seed_cleanup_db(db_path: Path) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE support_responses (id INTEGER PRIMARY KEY, created_at TEXT NOT NULL)")
        connection.execute("CREATE TABLE human_interventions (id INTEGER PRIMARY KEY, created_at TEXT NOT NULL)")
        connection.executemany(
            "INSERT INTO support_responses (created_at) VALUES (?)",
            [("2025-01-01 00:00:00",), ("2026-05-01 00:00:00",)],
        )
        connection.executemany(
            "INSERT INTO human_interventions (created_at) VALUES (?)",
            [("2025-01-01 00:00:00",), ("2026-05-01 00:00:00",)],
        )


def _count_rows(db_path: Path, table: str) -> int:
    with sqlite3.connect(db_path) as connection:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
