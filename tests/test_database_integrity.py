from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.database_integrity import build_database_integrity_report


def test_database_integrity_blocks_missing_database(tmp_path: Path) -> None:
    report = build_database_integrity_report(tmp_path / "missing.db")

    assert report["status"] == "blocked"
    assert report["exists"] is False
    assert "database_missing" in report["blocking_issues"]


def test_database_integrity_passes_initialized_database(tmp_path: Path) -> None:
    db_path = tmp_path / "ok.db"
    _seed_integrity_db(db_path, orphan=False)

    report = build_database_integrity_report(db_path)

    assert report["status"] == "ok"
    assert report["quick_check"] == "ok"
    assert report["missing_tables"] == []
    assert report["table_counts"]["support_responses"] == 1
    assert report["orphan_references"]["referral_events"]["orphan_count"] == 0


def test_database_integrity_warns_on_orphan_response_references(tmp_path: Path) -> None:
    db_path = tmp_path / "orphan.db"
    _seed_integrity_db(db_path, orphan=True)

    report = build_database_integrity_report(db_path)

    assert report["status"] == "watch"
    assert "orphan_response_references" in report["watch_items"]
    assert report["orphan_references"]["referral_events"]["orphan_count"] == 1


def _seed_integrity_db(db_path: Path, *, orphan: bool) -> None:
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
