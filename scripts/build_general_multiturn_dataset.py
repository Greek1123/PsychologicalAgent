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
    parser = argparse.ArgumentParser(description="Build a general multi-turn warmup dataset from dialog_release.")
    parser.add_argument(
        "--input",
        default=str(ROOT / "data" / "training" / "dialog_release.json"),
        help="Path to dialog_release.json.",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "data" / "training" / "general_multiturn" / "general_phase0_train_ms_swift.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument("--min-turns", type=int, default=10, help="Minimum kept message count.")
    parser.add_argument("--max-turns", type=int, default=20, help="Maximum kept message count.")
    parser.add_argument("--limit", type=int, default=6000, help="Optional sample cap.")
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
