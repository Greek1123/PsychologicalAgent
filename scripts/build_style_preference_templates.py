from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# 这个脚本把 style_review 或 style_keep 转成偏好标注模板，服务单独的风格对齐阶段。
from campus_support_agent.preference_template_builder import build_preference_templates


def main() -> None:
    parser = argparse.ArgumentParser(description="Build style preference annotation templates.")
    parser.add_argument(
        "--input",
        default=str(ROOT / "data" / "processed" / "triaged_style" / "style_review.jsonl"),
        help="Input JSONL path.",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "data" / "processed" / "style_preference" / "style_dpo_template.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument("--limit", type=int, default=500, help="Optional max sample count.")
    args = parser.parse_args()

    count = build_preference_templates(args.input, args.out, limit=args.limit)
    print(f"wrote {count} samples")


if __name__ == "__main__":
    main()
