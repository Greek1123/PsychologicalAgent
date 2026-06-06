from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from backup_sqlite_database import backup_sqlite_database


def test_backup_sqlite_database_creates_verified_backup_and_metadata(tmp_path: Path) -> None:
    db_path = tmp_path / "agent.db"
    backup_dir = tmp_path / "backups"
    _seed_backup_db(db_path, orphan=False)

    metadata = backup_sqlite_database(
        db_path=db_path,
        backup_dir=backup_dir,
        label="before-cleanup",
        timestamp=datetime(2026, 6, 6, 12, 0, 0, tzinfo=timezone.utc),
    )

    backup_path = Path(metadata["backup_db"])
    metadata_path = Path(metadata["metadata_path"])
    assert backup_path.exists()
    assert metadata_path.exists()
    assert metadata["source_integrity_status"] == "ok"
    assert metadata["backup_integrity_status"] == "ok"
    assert metadata["table_counts"]["support_responses"] == 1
    assert _count_rows(backup_path, "support_responses") == 1


def test_backup_sqlite_database_rejects_watch_integrity_by_default(tmp_path: Path) -> None:
    db_path = tmp_path / "agent.db"
    _seed_backup_db(db_path, orphan=True)

    try:
        backup_sqlite_database(db_path=db_path, backup_dir=tmp_path / "backups")
    except RuntimeError as exc:
        assert "watch" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for watch integrity status.")


def test_backup_sqlite_database_can_allow_watch_integrity(tmp_path: Path) -> None:
    db_path = tmp_path / "agent.db"
    _seed_backup_db(db_path, orphan=True)

    metadata = backup_sqlite_database(
        db_path=db_path,
        backup_dir=tmp_path / "backups",
        require_integrity_ok=False,
        timestamp=datetime(2026, 6, 6, 12, 0, 0, tzinfo=timezone.utc),
    )

    assert Path(metadata["backup_db"]).exists()
    assert metadata["source_integrity_status"] == "watch"
    assert metadata["backup_integrity_status"] == "watch"


def _seed_backup_db(db_path: Path, *, orphan: bool) -> None:
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
        connection.execute("INSERT INTO support_responses (response_id, created_at) VALUES ('resp-ok', '2026-06-06')")
        referral_response_id = "resp-missing" if orphan else "resp-ok"
        connection.execute(
            "INSERT INTO referral_events (response_id, created_at) VALUES (?, '2026-06-06')",
            (referral_response_id,),
        )


def _count_rows(db_path: Path, table: str) -> int:
    with sqlite3.connect(db_path) as connection:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
