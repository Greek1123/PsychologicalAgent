from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from campus_support_agent.general_phase0_mixer import build_mixed_general_phase0_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a mixed Phase 0 chat dataset from base and multiturn_0.8M data.")
    parser.add_argument(
        "--base-input",
        default=str(ROOT / "data" / "training" / "general_multiturn" / "general_phase0_train_ms_swift.jsonl"),
        help="Existing general Phase 0 JSONL.",
    )
    parser.add_argument(
        "--augment-input",
        default=str(
            ROOT / "data" / "training" / "general_multiturn" / "general_phase0_from_multiturn_0_8m.jsonl"
        ),
        help="Multiturn 0.8M candidate JSONL.",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "data" / "training" / "general_multiturn" / "general_phase0_mixed_train_ms_swift.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument("--target-total", type=int, default=8000, help="Final mixed dataset size.")
    parser.add_argument("--augment-ratio", type=float, default=0.35, help="Share of augment data to keep.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    args = parser.parse_args()

    stats = build_mixed_general_phase0_dataset(
        args.base_input,
        args.augment_input,
        args.out,
        target_total=args.target_total,
        augment_ratio=args.augment_ratio,
        seed=args.seed,
    )
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
