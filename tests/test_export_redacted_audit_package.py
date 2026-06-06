from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from export_redacted_audit_package import export_redacted_audit_package


NOW = datetime(2026, 6, 6, 13, 0, 0, tzinfo=timezone.utc)


def test_export_redacted_audit_package_redacts_text_and_pseudonymizes_ids(tmp_path: Path) -> None:
    db_path = tmp_path / "agent.db"
    _seed_audit_db(db_path, orphan=False)

    manifest = export_redacted_audit_package(
        db_path=db_path,
        out_dir=tmp_path / "exports",
        label="handoff",
        salt="fixed-salt",
        timestamp=NOW,
    )

    package_dir = Path(manifest["package_dir"])
    assert package_dir.exists()
    assert Path(manifest["manifest_path"]).exists()
    assert manifest["integrity_status"] == "ok"
    assert manifest["tables"]["support_responses"]["rows"] == 1
    assert manifest["privacy"]["redaction_summary"]["total_redactions"] >= 3

    support_row = _read_first_jsonl(package_dir / "support_responses.jsonl")
    message_row = _read_first_jsonl(package_dir / "conversation_messages.jsonl")
    human_row = _read_first_jsonl(package_dir / "human_interventions.jsonl")

    assert support_row["session_id"].startswith("session_")
    assert support_row["response_id"].startswith("response_")
    assert support_row["input_text"] != "我的手机号是13812345678，邮箱me@example.com"
    assert "13812345678" not in json.dumps(support_row, ensure_ascii=False)
    assert "me@example.com" not in json.dumps(support_row, ensure_ascii=False)
    assert message_row["session_id"] == support_row["session_id"]
    assert human_row["handler_id"].startswith("handler_")
    assert "11010519491231002X" not in json.dumps(human_row, ensure_ascii=False)


def test_export_redacted_audit_package_rejects_watch_without_override(tmp_path: Path) -> None:
    db_path = tmp_path / "agent.db"
    _seed_audit_db(db_path, orphan=True)

    try:
        export_redacted_audit_package(
            db_path=db_path,
            out_dir=tmp_path / "exports",
            timestamp=NOW,
        )
    except RuntimeError as exc:
        assert "watch" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for watch integrity status.")


def test_export_redacted_audit_package_allows_watch_when_explicit(tmp_path: Path) -> None:
    db_path = tmp_path / "agent.db"
    _seed_audit_db(db_path, orphan=True)

    manifest = export_redacted_audit_package(
        db_path=db_path,
        out_dir=tmp_path / "exports",
        allow_watch=True,
        timestamp=NOW,
    )

    assert manifest["integrity_status"] == "watch"
    assert "orphan_response_references" in manifest["integrity_watch_items"]


def _seed_audit_db(db_path: Path, *, orphan: bool) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE conversation_messages (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT
            );
            CREATE TABLE entropy_trace (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                response_id TEXT NOT NULL,
                score INTEGER,
                level INTEGER,
                balance_state TEXT,
                dominant_drivers_json TEXT,
                created_at TEXT
            );
            CREATE TABLE support_responses (
                id INTEGER PRIMARY KEY,
                session_id TEXT,
                response_id TEXT UNIQUE,
                source TEXT,
                input_text TEXT,
                transcript TEXT,
                student_context_json TEXT,
                conversation_history_json TEXT,
                response_json TEXT,
                created_at TEXT
            );
            CREATE TABLE referral_events (
                id INTEGER PRIMARY KEY,
                session_id TEXT,
                response_id TEXT,
                urgency TEXT,
                reasons_json TEXT,
                policy_name TEXT,
                risk_level TEXT,
                entropy_score INTEGER,
                manual_referral_recommended INTEGER,
                created_at TEXT
            );
            CREATE TABLE intervention_feedback (
                id INTEGER PRIMARY KEY,
                session_id TEXT,
                response_id TEXT,
                helpful_score INTEGER,
                mood_after INTEGER,
                user_note TEXT,
                tags_json TEXT,
                created_at TEXT
            );
            CREATE TABLE human_interventions (
                id INTEGER PRIMARY KEY,
                session_id TEXT,
                response_id TEXT,
                status TEXT,
                handler_id TEXT,
                note TEXT,
                next_action TEXT,
                tags_json TEXT,
                created_at TEXT
            );
            """
        )
        response_id = "resp-ok"
        connection.execute(
            "INSERT INTO conversation_messages VALUES (1, 'session-a', 'user', '我的手机号是13812345678', '2026-06-06')"
        )
        connection.execute(
            "INSERT INTO entropy_trace VALUES (1, 'session-a', 'resp-ok', 20, 2, 'fragile', '[]', '2026-06-06')"
        )
        connection.execute(
            """
            INSERT INTO support_responses VALUES (
                1, 'session-a', 'resp-ok', 'text',
                '我的手机号是13812345678，邮箱me@example.com',
                NULL,
                '{"student_id":"学号: 2024123456"}',
                '[{"role":"user","content":"QQ:1234567"}]',
                '{"reply_text":"先联系辅导员，微信 wxhelper01"}',
                '2026-06-06'
            )
            """
        )
        referral_response_id = "resp-missing" if orphan else response_id
        connection.execute(
            "INSERT INTO referral_events VALUES (1, 'session-a', ?, 'urgent', '[]', 'safety', 'critical', 30, 1, '2026-06-06')",
            (referral_response_id,),
        )
        connection.execute(
            "INSERT INTO intervention_feedback VALUES (1, 'session-a', 'resp-ok', 4, 3, '身份证11010519491231002X', '[]', '2026-06-06')"
        )
        connection.execute(
            """
            INSERT INTO human_interventions VALUES (
                1, 'session-a', 'resp-ok', 'acknowledged', 'teacher-001',
                '学生电话13812345678，身份证11010519491231002X',
                '明天联系me@example.com',
                '[]',
                '2026-06-06'
            )
            """
        )


def _read_first_jsonl(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8").splitlines()[0])
