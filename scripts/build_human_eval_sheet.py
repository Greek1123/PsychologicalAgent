from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# 这个脚本把测试集导成咨询师可直接评分的 CSV 表。
from campus_support_agent.human_eval_builder import build_human_eval_sheet


def main() -> None:
    parser = argparse.ArgumentParser(description="Build human evaluation CSV from style test set.")
    parser.add_argument(
        "--input",
        default=str(ROOT / "data" / "processed" / "style_splits" / "style_test.jsonl"),
        help="Input JSONL path.",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "data" / "processed" / "human_eval" / "style_eval_sheet.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--limit", type=int, default=100, help="Optional sample limit.")
    args = parser.parse_args()

    count = build_human_eval_sheet(args.input, args.out, limit=args.limit)
    print(f"wrote {count} rows")


if __name__ == "__main__":
    main()
