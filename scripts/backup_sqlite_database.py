from __future__ import annotations

import argparse
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


def backup_sqlite_database(
    *,
    db_path: str | Path,
    backup_dir: str | Path,
    label: str = "manual",
    require_integrity_ok: bool = True,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    source = Path(db_path)
    if not source.exists():
        raise FileNotFoundError(f"Database not found: {source}")

    integrity = build_database_integrity_report(source)
    if require_integrity_ok and integrity["status"] != "ok":
        raise RuntimeError(f"Database integrity is {integrity['status']}; refusing backup without --allow-watch.")

    created_at = timestamp or datetime.now(timezone.utc)
    safe_label = _safe_label(label)
    target_dir = Path(backup_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    backup_path = target_dir / f"{source.stem}_{created_at.strftime('%Y%m%d_%H%M%S')}_{safe_label}.db"
    metadata_path = backup_path.with_suffix(".json")

    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as source_connection:
        with sqlite3.connect(backup_path) as backup_connection:
            source_connection.backup(backup_connection)

    backup_integrity = build_database_integrity_report(backup_path)
    metadata = {
        "source_db": str(source),
        "backup_db": str(backup_path),
        "metadata_path": str(metadata_path),
        "label": safe_label,
        "created_at": created_at.isoformat(),
        "source_integrity_status": integrity["status"],
        "backup_integrity_status": backup_integrity["status"],
        "source_size_bytes": source.stat().st_size,
        "backup_size_bytes": backup_path.stat().st_size,
        "table_counts": backup_integrity.get("table_counts") or {},
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def _safe_label(label: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in label.strip())
    return cleaned.strip("-_") or "manual"


def parse_args() -> argparse.Namespace:
    settings = Settings()
    parser = argparse.ArgumentParser(description="Create a verified SQLite backup using the SQLite backup API.")
    parser.add_argument("--db", default=settings.database_path, help="SQLite database path.")
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "backups"), help="Backup output directory.")
    parser.add_argument("--label", default="manual", help="Short label included in backup file name.")
    parser.add_argument(
        "--allow-watch",
        action="store_true",
        help="Allow backup when integrity status is watch. Blocked databases are still rejected by sqlite errors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = backup_sqlite_database(
        db_path=args.db,
        backup_dir=args.out_dir,
        label=args.label,
        require_integrity_ok=not args.allow_watch,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
