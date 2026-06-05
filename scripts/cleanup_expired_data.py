from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.config import Settings  # noqa: E402


SESSION_DATA_TABLES = (
    "conversation_messages",
    "entropy_trace",
    "support_responses",
    "referral_events",
    "intervention_feedback",
)
AUDIT_LOG_TABLES = ("human_interventions",)


def cleanup_expired_data(
    *,
    db_path: str | Path,
    session_data_retention_days: int,
    audit_log_retention_days: int,
    apply: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    if session_data_retention_days <= 0:
        raise ValueError("session_data_retention_days must be positive.")
    if audit_log_retention_days <= 0:
        raise ValueError("audit_log_retention_days must be positive.")

    current_time = now or datetime.now(timezone.utc)
    session_cutoff = _sqlite_timestamp(current_time - timedelta(days=session_data_retention_days))
    audit_cutoff = _sqlite_timestamp(current_time - timedelta(days=audit_log_retention_days))
    path = Path(db_path)
    if not path.exists():
        return {
            "db_path": str(path),
            "apply": apply,
            "exists": False,
            "deleted_total": 0,
            "tables": {},
        }

    table_results: dict[str, Any] = {}
    deleted_total = 0
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        for table in SESSION_DATA_TABLES:
            result = _cleanup_table(connection, table=table, cutoff=session_cutoff, apply=apply)
            table_results[table] = result
            deleted_total += result["deleted"]
        for table in AUDIT_LOG_TABLES:
            result = _cleanup_table(connection, table=table, cutoff=audit_cutoff, apply=apply)
            table_results[table] = result
            deleted_total += result["deleted"]
        if not apply:
            connection.rollback()

    return {
        "db_path": str(path),
        "apply": apply,
        "exists": True,
        "session_data_retention_days": session_data_retention_days,
        "audit_log_retention_days": audit_log_retention_days,
        "session_cutoff": session_cutoff,
        "audit_cutoff": audit_cutoff,
        "deleted_total": deleted_total,
        "tables": table_results,
    }


def _cleanup_table(connection: sqlite3.Connection, *, table: str, cutoff: str, apply: bool) -> dict[str, Any]:
    if not _table_exists(connection, table):
        return {"exists": False, "matched": 0, "deleted": 0, "cutoff": cutoff}
    if not _table_has_created_at(connection, table):
        return {"exists": True, "matched": 0, "deleted": 0, "cutoff": cutoff, "skipped": "missing_created_at"}

    matched = int(
        connection.execute(
            f"SELECT COUNT(*) AS total FROM {table} WHERE created_at < ?",
            (cutoff,),
        ).fetchone()["total"]
    )
    deleted = 0
    if apply and matched:
        cursor = connection.execute(f"DELETE FROM {table} WHERE created_at < ?", (cutoff,))
        deleted = int(cursor.rowcount or 0)
    return {"exists": True, "matched": matched, "deleted": deleted, "cutoff": cutoff}


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _table_has_created_at(connection: sqlite3.Connection, table: str) -> bool:
    rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row["name"] == "created_at" for row in rows)


def _sqlite_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def parse_args() -> argparse.Namespace:
    settings = Settings()
    parser = argparse.ArgumentParser(description="Dry-run or apply retention cleanup for the local SQLite database.")
    parser.add_argument("--db", default=settings.database_path, help="SQLite database path.")
    parser.add_argument("--session-days", type=int, default=settings.session_data_retention_days)
    parser.add_argument("--audit-days", type=int, default=settings.audit_log_retention_days)
    parser.add_argument("--apply", action="store_true", help="Actually delete expired rows. Omit for dry-run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = cleanup_expired_data(
        db_path=args.db,
        session_data_retention_days=args.session_days,
        audit_log_retention_days=args.audit_days,
        apply=args.apply,
    )
    print(result)


if __name__ == "__main__":
    main()
