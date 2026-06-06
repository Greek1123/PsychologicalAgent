from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.config import Settings  # noqa: E402
from campus_support_agent.database_integrity import build_database_integrity_report  # noqa: E402
from campus_support_agent.privacy_redaction import (  # noqa: E402
    PrivacyRedactionSummary,
    redact_private_identifiers_in_value,
)


EXPORT_TABLES = (
    "conversation_messages",
    "entropy_trace",
    "support_responses",
    "referral_events",
    "intervention_feedback",
    "human_interventions",
)

PSEUDONYM_FIELDS = {
    "session_id": "session",
    "response_id": "response",
    "handler_id": "handler",
}


def export_redacted_audit_package(
    *,
    db_path: str | Path,
    out_dir: str | Path,
    label: str = "audit",
    salt: str | None = None,
    limit_per_table: int | None = None,
    allow_watch: bool = False,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    source = Path(db_path)
    if not source.exists():
        raise FileNotFoundError(f"Database not found: {source}")

    integrity = build_database_integrity_report(source)
    if integrity["status"] == "blocked":
        raise RuntimeError("Database integrity is blocked; refusing audit export.")
    if integrity["status"] == "watch" and not allow_watch:
        raise RuntimeError("Database integrity is watch; rerun with --allow-watch to export with warnings.")

    created_at = timestamp or datetime.now(timezone.utc)
    export_salt = salt or f"audit-{created_at.isoformat()}"
    safe_label = _safe_label(label)
    package_dir = Path(out_dir) / f"{source.stem}_{created_at.strftime('%Y%m%d_%H%M%S')}_{safe_label}"
    package_dir.mkdir(parents=True, exist_ok=True)

    redaction_summary = PrivacyRedactionSummary()
    table_reports: dict[str, Any] = {}
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        available_tables = _list_tables(connection)
        for table in EXPORT_TABLES:
            if table not in available_tables:
                table_reports[table] = {"exists": False, "rows": 0, "file": None}
                continue
            rows = _read_rows(connection, table, limit=limit_per_table)
            output_path = package_dir / f"{table}.jsonl"
            table_redactions = PrivacyRedactionSummary()
            with output_path.open("w", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    exported, row_summary = _project_row_for_export(dict(row), salt=export_salt)
                    table_redactions.merge(row_summary)
                    handle.write(json.dumps(exported, ensure_ascii=False, sort_keys=True) + "\n")
            redaction_summary.merge(table_redactions)
            table_reports[table] = {
                "exists": True,
                "rows": len(rows),
                "file": str(output_path),
                "redactions": table_redactions.as_dict(),
            }

    manifest = {
        "package_dir": str(package_dir),
        "source_db": str(source),
        "label": safe_label,
        "created_at": created_at.isoformat(),
        "integrity_status": integrity["status"],
        "integrity_watch_items": integrity.get("watch_items") or [],
        "tables": table_reports,
        "privacy": {
            "direct_identifiers_redacted": True,
            "stable_pseudonyms": sorted(PSEUDONYM_FIELDS),
            "salt_sha256": hashlib.sha256(export_salt.encode("utf-8")).hexdigest(),
            "redaction_summary": redaction_summary.as_dict(),
        },
    }
    manifest_path = package_dir / "manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _list_tables(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {str(row["name"]) for row in rows}


def _read_rows(connection: sqlite3.Connection, table: str, *, limit: int | None) -> list[sqlite3.Row]:
    query = f"SELECT * FROM {table} ORDER BY id ASC"
    params: tuple[Any, ...] = ()
    if limit is not None:
        query += " LIMIT ?"
        params = (max(limit, 0),)
    return list(connection.execute(query, params).fetchall())


def _project_row_for_export(row: dict[str, Any], *, salt: str) -> tuple[dict[str, Any], PrivacyRedactionSummary]:
    projected: dict[str, Any] = {}
    for key, value in row.items():
        if key in PSEUDONYM_FIELDS and value not in {None, ""}:
            projected[key] = _pseudonymize(str(value), namespace=PSEUDONYM_FIELDS[key], salt=salt)
            continue
        if key.endswith("_json") and isinstance(value, str):
            projected[key.removesuffix("_json")] = _load_json_or_raw(value)
            continue
        projected[key] = value
    redacted, summary = redact_private_identifiers_in_value(projected, redact_all_strings=False)
    return redacted, summary


def _load_json_or_raw(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _pseudonymize(value: str, *, namespace: str, salt: str) -> str:
    digest = hashlib.sha256(f"{namespace}:{salt}:{value}".encode("utf-8")).hexdigest()[:16]
    return f"{namespace}_{digest}"


def _safe_label(label: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in label.strip())
    return cleaned.strip("-_") or "audit"


def parse_args() -> argparse.Namespace:
    settings = Settings()
    parser = argparse.ArgumentParser(description="Export a redacted SQLite audit package as JSONL files.")
    parser.add_argument("--db", default=settings.database_path, help="SQLite database path.")
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "audit_exports"), help="Audit package output root.")
    parser.add_argument("--label", default="audit", help="Short label included in the package directory.")
    parser.add_argument("--salt", default=None, help="Optional stable pseudonym salt. Omit to generate one per export.")
    parser.add_argument("--limit-per-table", type=int, default=None, help="Optional maximum rows per table.")
    parser.add_argument("--allow-watch", action="store_true", help="Allow export when integrity status is watch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = export_redacted_audit_package(
        db_path=args.db,
        out_dir=args.out_dir,
        label=args.label,
        salt=args.salt,
        limit_per_table=args.limit_per_table,
        allow_watch=args.allow_watch,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
