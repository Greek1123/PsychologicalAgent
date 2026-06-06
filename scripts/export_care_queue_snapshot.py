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

from campus_support_agent.config import Settings  # noqa: E402
from campus_support_agent.database_integrity import build_database_integrity_report  # noqa: E402
from campus_support_agent.privacy_views import normalize_view_role, project_care_queue_for_role  # noqa: E402
from campus_support_agent.storage import SQLiteSessionStore  # noqa: E402


def export_care_queue_snapshot(
    *,
    db_path: str | Path,
    out_dir: str | Path,
    role: str = "counselor",
    label: str = "care-queue",
    limit: int | None = 100,
    include_low_priority: bool = False,
    include_resolved: bool = False,
    workflow_state: str | None = None,
    owner: str | None = None,
    only_overdue: bool = False,
    allow_watch: bool = False,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    normalized_role = normalize_view_role(role)
    if normalized_role == "student":
        raise ValueError("Care queue snapshots support counselor, research, or admin roles.")

    source = Path(db_path)
    if not source.exists():
        raise FileNotFoundError(f"Database not found: {source}")
    integrity = build_database_integrity_report(source)
    if integrity["status"] == "blocked":
        raise RuntimeError("Database integrity is blocked; refusing care queue snapshot export.")
    if integrity["status"] == "watch" and not allow_watch:
        raise RuntimeError("Database integrity is watch; rerun with --allow-watch to export with warnings.")

    created_at = timestamp or datetime.now(timezone.utc)
    safe_label = _safe_label(label)
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{source.stem}_{created_at.strftime('%Y%m%d_%H%M%S')}_{safe_label}_{normalized_role}"
    json_path = target_dir / f"{base_name}.json"
    markdown_path = target_dir / f"{base_name}.md"

    store = SQLiteSessionStore(str(source))
    raw_queue = store.get_care_queue(
        limit=limit,
        include_low_priority=include_low_priority,
        include_resolved=include_resolved,
        workflow_state=workflow_state,
        owner=owner,
        only_overdue=only_overdue,
    )
    queue = project_care_queue_for_role(raw_queue, normalized_role)
    snapshot = {
        "source_db": str(source),
        "created_at": created_at.isoformat(),
        "role": normalized_role,
        "integrity_status": integrity["status"],
        "integrity_watch_items": integrity.get("watch_items") or [],
        "filters": {
            "limit": limit,
            "include_low_priority": include_low_priority,
            "include_resolved": include_resolved,
            "workflow_state": workflow_state,
            "owner": owner,
            "only_overdue": only_overdue,
        },
        "queue": queue,
    }
    json_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_render_markdown_snapshot(snapshot), encoding="utf-8")
    return {
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "role": normalized_role,
        "total_items": queue.get("total_items", 0),
        "integrity_status": integrity["status"],
        "privacy_redaction": queue.get("privacy_redaction"),
    }


def _render_markdown_snapshot(snapshot: dict[str, Any]) -> str:
    queue = snapshot.get("queue") or {}
    lines = [
        "# Care Queue Snapshot",
        "",
        f"- Created at: `{snapshot.get('created_at')}`",
        f"- Role: `{snapshot.get('role')}`",
        f"- Integrity: `{snapshot.get('integrity_status')}`",
        f"- Total items: `{queue.get('total_items', 0)}`",
        f"- Priority counts: `{json.dumps(queue.get('priority_counts') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- Workflow filters: `{json.dumps((queue.get('filters') or {}), ensure_ascii=False, sort_keys=True)}`",
        "",
        "| # | Session | Priority | Workflow | Owner | Route | Action | Created |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for index, item in enumerate(queue.get("items") or [], start=1):
        workflow = (item.get("evidence") or {}).get("human_workflow") or {}
        lines.append(
            "| {index} | `{session}` | `{priority}` | `{workflow_state}` | `{owner}` | `{route}` | `{action}` | `{created}` |".format(
                index=index,
                session=_cell(item.get("session_id")),
                priority=_cell(item.get("priority")),
                workflow_state=_cell(workflow.get("workflow_state")),
                owner=_cell(workflow.get("owner")),
                route=_cell(item.get("route")),
                action=_cell(item.get("recommended_action")),
                created=_cell(item.get("created_at")),
            )
        )
    return "\n".join(lines) + "\n"


def _cell(value: Any) -> str:
    if value is None or value == "":
        return "-"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _safe_label(label: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in label.strip())
    return cleaned.strip("-_") or "care-queue"


def parse_args() -> argparse.Namespace:
    settings = Settings()
    parser = argparse.ArgumentParser(description="Export a role-projected care queue snapshot as JSON and Markdown.")
    parser.add_argument("--db", default=settings.database_path, help="SQLite database path.")
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "care_queue_exports"), help="Snapshot output directory.")
    parser.add_argument("--role", default="counselor", choices=["counselor", "research", "admin"])
    parser.add_argument("--label", default="care-queue")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--include-low-priority", action="store_true")
    parser.add_argument("--include-resolved", action="store_true")
    parser.add_argument("--workflow-state", default=None)
    parser.add_argument("--owner", default=None)
    parser.add_argument("--only-overdue", action="store_true")
    parser.add_argument("--allow-watch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export_care_queue_snapshot(
        db_path=args.db,
        out_dir=args.out_dir,
        role=args.role,
        label=args.label,
        limit=args.limit,
        include_low_priority=args.include_low_priority,
        include_resolved=args.include_resolved,
        workflow_state=args.workflow_state,
        owner=args.owner,
        only_overdue=args.only_overdue,
        allow_watch=args.allow_watch,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
