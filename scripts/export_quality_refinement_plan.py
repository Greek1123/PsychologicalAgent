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
from campus_support_agent.storage import SQLiteSessionStore


DEFAULT_OUTPUT = ROOT / "data" / "training" / "feedback_bad_cases" / "quality_refinement_plan.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a backend quality refinement plan.")
    parser.add_argument("--db", default=Settings().database_path, help="SQLite database path.")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT), help="Output JSON path.")
    parser.add_argument("--session-id", default=None, help="Optional session filter.")
    parser.add_argument("--source-limit", type=int, default=200, help="Max support responses to scan.")
    parser.add_argument("--bad-case-limit", type=int, default=100, help="Max bad cases to route.")
    parser.add_argument("--min-quality-score", type=int, default=80, help="Bad-case quality threshold.")
    parser.add_argument("--max-examples-per-bucket", type=int, default=8, help="Examples kept per route bucket.")
    args = parser.parse_args()

    settings = Settings()
    configure_logging(settings)
    store = SQLiteSessionStore(db_path=args.db, max_messages=settings.max_history_turns * 2)
    plan = store.get_quality_refinement_plan(
        session_id=args.session_id,
        source_limit=args.source_limit,
        bad_case_limit=args.bad_case_limit,
        min_quality_score=args.min_quality_score,
        max_examples_per_bucket=args.max_examples_per_bucket,
    )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(output), "total_bad_cases": plan["total_bad_cases"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
