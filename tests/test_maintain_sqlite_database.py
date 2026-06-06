from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from maintain_sqlite_database import maintain_sqlite_database


NOW = datetime(2026, 6, 6, 12, 0, 0, tzinfo=timezone.utc)


def test_maintenance_dry_run_checks_integrity_and_does_not_delete_or_backup(tmp_path: Path) -> None:
    db_path = tmp_path / "agent.db"
    backup_dir = tmp_path / "backups"
    _seed_maintenance_db(db_path, orphan=False)

    result = maintain_sqlite_database(
        db_path=db_path,
        backup_dir=backup_dir,
        session_data_retention_days=180,
        audit_log_retention_days=365,
        apply=False,
        now=NOW,
    )

    assert result["integrity"]["status"] == "ok"
    assert result["backup"] == {"created": False, "reason": "dry_run"}
    assert result["cleanup"]["tables"]["support_responses"]["matched"] == 1
    assert result["cleanup"]["tables"]["human_interventions"]["matched"] == 1
    assert _count_rows(db_path, "support_responses") == 2
    assert _count_rows(db_path, "human_interventions") == 2
    assert not backup_dir.exists()


def test_maintenance_apply_creates_backup_before_cleanup(tmp_path: Path) -> None:
    db_path = tmp_path / "agent.db"
    backup_dir = tmp_path / "backups"
    _seed_maintenance_db(db_path, orphan=False)

    result = maintain_sqlite_database(
        db_path=db_path,
        backup_dir=backup_dir,
        session_data_retention_days=180,
        audit_log_retention_days=365,
        apply=True,
        label="before-cleanup",
        now=NOW,
    )

    assert result["integrity"]["status"] == "ok"
    assert result["backup"]["created"] is True
    backup_path = Path(result["backup"]["metadata"]["backup_db"])
    assert backup_path.exists()
    assert _count_rows(backup_path, "support_responses") == 2
    assert result["cleanup"]["deleted_total"] == 2
    assert _count_rows(db_path, "support_responses") == 1
    assert _count_rows(db_path, "human_interventions") == 1


def test_maintenance_apply_rejects_watch_integrity_without_override(tmp_path: Path) -> None:
    db_path = tmp_path / "agent.db"
    _seed_maintenance_db(db_path, orphan=True)

    try:
        maintain_sqlite_database(
            db_path=db_path,
            backup_dir=tmp_path / "backups",
            session_data_retention_days=180,
            audit_log_retention_days=365,
            apply=True,
            now=NOW,
        )
    except RuntimeError as exc:
        assert "watch" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for watch integrity status.")

    assert _count_rows(db_path, "support_responses") == 2
    assert not (tmp_path / "backups").exists()


def test_maintenance_apply_allows_watch_when_explicit_and_preserves_backup(tmp_path: Path) -> None:
    db_path = tmp_path / "agent.db"
    backup_dir = tmp_path / "backups"
    _seed_maintenance_db(db_path, orphan=True)

    result = maintain_sqlite_database(
        db_path=db_path,
        backup_dir=backup_dir,
        session_data_retention_days=180,
        audit_log_retention_days=365,
        apply=True,
        allow_watch=True,
        now=NOW,
    )

    assert result["integrity"]["status"] == "watch"
    assert result["backup"]["metadata"]["source_integrity_status"] == "watch"
    assert Path(result["backup"]["metadata"]["backup_db"]).exists()
    assert result["cleanup"]["deleted_total"] == 2


def _seed_maintenance_db(db_path: Path, *, orphan: bool) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE conversation_messages (id INTEGER PRIMARY KEY, created_at TEXT);
            CREATE TABLE entropy_trace (id INTEGER PRIMARY KEY, response_id TEXT, created_at TEXT);
            CREATE TABLE support_responses (id INTEGER PRIMARY KEY, response_id TEXT UNIQUE, created_at TEXT);
            CREATE TABLE referral_events (id INTEGER PRIMARY KEY, response_id TEXT, created_at TEXT);
            CREATE TABLE intervention_feedback (id INTEGER PRIMARY KEY, response_id TEXT, created_at TEXT);
            CREATE TABLE human_interventions (id INTEGER PRIMARY KEY, response_id TEXT, created_at TEXT);
            """
        )
        connection.executemany(
            "INSERT INTO support_responses (response_id, created_at) VALUES (?, ?)",
            [
                ("resp-old", "2025-01-01 00:00:00"),
                ("resp-fresh", "2026-05-01 00:00:00"),
            ],
        )
        referral_response_id = "resp-missing" if orphan else "resp-fresh"
        connection.execute(
            "INSERT INTO referral_events (response_id, created_at) VALUES (?, '2026-05-01 00:00:00')",
            (referral_response_id,),
        )
        connection.executemany(
            "INSERT INTO human_interventions (response_id, created_at) VALUES (?, ?)",
            [
                ("resp-fresh", "2025-01-01 00:00:00"),
                ("resp-fresh", "2026-05-01 00:00:00"),
            ],
        )


def _count_rows(db_path: Path, table: str) -> int:
    with sqlite3.connect(db_path) as connection:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
