from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# 这个脚本把单轮问答扩成多轮 style_sft，服务“先训练说话习惯”的路线。
from campus_support_agent.single_turn_expander import expand_single_turn_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand single-turn counseling data into multi-turn style SFT.")
    parser.add_argument(
        "--input",
        default=str(ROOT / "data" / "Psychology-10K-ZH(1).json"),
        help="Input single-turn dataset path.",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "data" / "processed" / "style_sft_from_single_turn.jsonl"),
        help="Output JSONL path.",
    )
    args = parser.parse_args()

    count = expand_single_turn_dataset(args.input, args.out)
    print(f"wrote {count} samples")


if __name__ == "__main__":
    main()
