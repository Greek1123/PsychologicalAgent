from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from backup_sqlite_database import backup_sqlite_database  # noqa: E402
from cleanup_expired_data import cleanup_expired_data  # noqa: E402
from campus_support_agent.config import Settings  # noqa: E402
from campus_support_agent.database_integrity import build_database_integrity_report  # noqa: E402


def maintain_sqlite_database(
    *,
    db_path: str | Path,
    backup_dir: str | Path,
    session_data_retention_days: int,
    audit_log_retention_days: int,
    apply: bool = False,
    allow_watch: bool = False,
    skip_backup: bool = False,
    label: str = "maintenance",
    now: datetime | None = None,
) -> dict[str, Any]:
    current_time = now or datetime.now(timezone.utc)
    integrity = build_database_integrity_report(db_path)
    result: dict[str, Any] = {
        "db_path": str(Path(db_path)),
        "apply": apply,
        "allow_watch": allow_watch,
        "skip_backup": skip_backup,
        "integrity": integrity,
        "backup": {"created": False, "reason": "dry_run" if not apply else "pending"},
        "cleanup": None,
    }

    if integrity["status"] == "blocked":
        result["cleanup"] = {"apply": False, "skipped": "integrity_blocked"}
        result["backup"] = {"created": False, "reason": "integrity_blocked"}
        if apply:
            raise RuntimeError("Database integrity is blocked; refusing maintenance apply.")
        return result

    if integrity["status"] == "watch" and apply and not allow_watch:
        result["cleanup"] = {"apply": False, "skipped": "integrity_watch"}
        result["backup"] = {"created": False, "reason": "integrity_watch"}
        raise RuntimeError("Database integrity is watch; rerun with --allow-watch to preserve and clean anyway.")

    if apply:
        if skip_backup:
            result["backup"] = {"created": False, "reason": "skip_backup"}
        else:
            backup_metadata = backup_sqlite_database(
                db_path=db_path,
                backup_dir=backup_dir,
                label=label,
                require_integrity_ok=not allow_watch,
                timestamp=current_time,
            )
            result["backup"] = {"created": True, "metadata": backup_metadata}

    result["cleanup"] = cleanup_expired_data(
        db_path=db_path,
        session_data_retention_days=session_data_retention_days,
        audit_log_retention_days=audit_log_retention_days,
        apply=apply,
        now=current_time,
    )
    return result


def parse_args() -> argparse.Namespace:
    settings = Settings()
    parser = argparse.ArgumentParser(
        description="Run a safe SQLite maintenance workflow: integrity check, optional backup, then retention cleanup."
    )
    parser.add_argument("--db", default=settings.database_path, help="SQLite database path.")
    parser.add_argument("--backup-dir", default=str(ROOT / "data" / "backups"), help="Backup output directory.")
    parser.add_argument("--session-days", type=int, default=settings.session_data_retention_days)
    parser.add_argument("--audit-days", type=int, default=settings.audit_log_retention_days)
    parser.add_argument("--label", default="maintenance", help="Backup label used when --apply creates a backup.")
    parser.add_argument("--apply", action="store_true", help="Create backup and delete expired rows. Omit for dry-run.")
    parser.add_argument("--allow-watch", action="store_true", help="Allow apply when integrity status is watch.")
    parser.add_argument("--skip-backup", action="store_true", help="Apply cleanup without creating a backup first.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = maintain_sqlite_database(
        db_path=args.db,
        backup_dir=args.backup_dir,
        session_data_retention_days=args.session_days,
        audit_log_retention_days=args.audit_days,
        apply=args.apply,
        allow_watch=args.allow_watch,
        skip_backup=args.skip_backup,
        label=args.label,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
