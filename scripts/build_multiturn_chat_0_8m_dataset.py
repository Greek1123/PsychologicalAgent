from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from campus_support_agent.general_dialog_builder import build_general_multiturn_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a general multi-turn warmup dataset from multiturn_chat_0.8M.")
    parser.add_argument(
        "--input",
        default=str(ROOT / "data" / "_tmp_multiturn" / "multiturn_chat_0.8M" / "multiturn_chat_0.8M.json"),
        help="Path to multiturn_chat_0.8M.json.",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "data" / "training" / "general_multiturn" / "general_phase0_from_multiturn_0_8m.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument("--min-turns", type=int, default=6, help="Minimum kept message count.")
    parser.add_argument("--max-turns", type=int, default=16, help="Maximum kept message count.")
    parser.add_argument("--limit", type=int, default=20000, help="Optional sample cap.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed before sampling.")
    args = parser.parse_args()

    stats = build_general_multiturn_dataset(
        args.input,
        args.out,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        limit=args.limit,
        seed=args.seed,
    )
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
