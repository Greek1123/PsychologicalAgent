from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


REQUIRED_TABLES = {
    "conversation_messages",
    "entropy_trace",
    "support_responses",
    "referral_events",
    "intervention_feedback",
    "human_interventions",
}

RESPONSE_REFERENCE_TABLES = {
    "referral_events": "response_id",
    "intervention_feedback": "response_id",
    "human_interventions": "response_id",
}


def build_database_integrity_report(db_path: str | Path) -> dict[str, Any]:
    path = Path(db_path)
    if not path.exists():
        return {
            "status": "blocked",
            "db_path": str(path),
            "exists": False,
            "quick_check": None,
            "missing_tables": sorted(REQUIRED_TABLES),
            "table_counts": {},
            "orphan_references": {},
            "blocking_issues": ["database_missing"],
            "watch_items": [],
        }

    blocking_issues: list[str] = []
    watch_items: list[str] = []
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            quick_check = _quick_check(connection)
            tables = _list_tables(connection)
            missing_tables = sorted(REQUIRED_TABLES - tables)
            if quick_check != "ok":
                blocking_issues.append("sqlite_quick_check_failed")
            if missing_tables:
                blocking_issues.append("required_tables_missing")

            table_counts = {
                table: _count_rows(connection, table)
                for table in sorted(REQUIRED_TABLES & tables)
            }
            orphan_references = _build_orphan_reference_report(connection, tables)
            if any(item["orphan_count"] > 0 for item in orphan_references.values()):
                watch_items.append("orphan_response_references")
    except sqlite3.Error as exc:
        return {
            "status": "blocked",
            "db_path": str(path),
            "exists": True,
            "quick_check": None,
            "missing_tables": [],
            "table_counts": {},
            "orphan_references": {},
            "blocking_issues": ["database_unreadable"],
            "watch_items": [],
            "error": str(exc),
        }

    status = "blocked" if blocking_issues else "watch" if watch_items else "ok"
    return {
        "status": status,
        "db_path": str(path),
        "exists": True,
        "quick_check": quick_check,
        "missing_tables": missing_tables,
        "table_counts": table_counts,
        "orphan_references": orphan_references,
        "blocking_issues": blocking_issues,
        "watch_items": watch_items,
    }


def _quick_check(connection: sqlite3.Connection) -> str:
    row = connection.execute("PRAGMA quick_check").fetchone()
    return str(row[0]) if row else "missing_result"


def _list_tables(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {str(row["name"]) for row in rows}


def _count_rows(connection: sqlite3.Connection, table: str) -> int:
    return int(connection.execute(f"SELECT COUNT(*) AS total FROM {table}").fetchone()["total"])


def _build_orphan_reference_report(connection: sqlite3.Connection, tables: set[str]) -> dict[str, dict[str, Any]]:
    if "support_responses" not in tables:
        return {}

    report: dict[str, dict[str, Any]] = {}
    for table, column in RESPONSE_REFERENCE_TABLES.items():
        if table not in tables:
            continue
        row = connection.execute(
            f"""
            SELECT COUNT(*) AS total
            FROM {table} AS child
            LEFT JOIN support_responses AS parent
              ON child.{column} = parent.response_id
            WHERE child.{column} IS NOT NULL
              AND child.{column} != ''
              AND parent.response_id IS NULL
            """
        ).fetchone()
        report[table] = {
            "reference_column": column,
            "orphan_count": int(row["total"] if row else 0),
        }
    return report
