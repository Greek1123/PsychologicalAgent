from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from campus_support_agent.config import Settings
from campus_support_agent.logging_utils import configure_logging
from campus_support_agent.reply_quality_export import export_reply_quality_bad_cases
from campus_support_agent.storage import SQLiteSessionStore


DEFAULT_OUTPUT = ROOT / "data" / "training" / "feedback_bad_cases" / "reply_quality_bad_cases.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export reply-quality monitor findings into reviewable bad-case JSONL."
    )
    parser.add_argument("--db", default=Settings().database_path, help="SQLite database path.")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT), help="Output bad-case JSONL path.")
    parser.add_argument("--session-id", default=None, help="Optional session filter.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max exported bad cases.")
    parser.add_argument(
        "--source-limit",
        type=int,
        default=None,
        help="Optional max support responses to scan before quality filtering.",
    )
    parser.add_argument(
        "--min-quality-score",
        type=int,
        default=80,
        help="Export replies whose quality_score is <= this value or has explicit issues.",
    )
    args = parser.parse_args()

    settings = Settings()
    configure_logging(settings)
    store = SQLiteSessionStore(db_path=args.db, max_messages=settings.max_history_turns * 2)
    records = store.list_support_responses(session_id=args.session_id, limit=args.source_limit)
    stats = export_reply_quality_bad_cases(
        records,
        args.out,
        min_quality_score=args.min_quality_score,
        limit=args.limit,
    )
    stats["session_id"] = args.session_id
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
