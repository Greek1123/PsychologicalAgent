from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from campus_support_agent.style_training_pack_builder import build_style_training_pack


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the staged style-first training pack.")
    parser.add_argument(
        "--style-train",
        default=str(ROOT / "data" / "processed" / "style_splits" / "style_train.jsonl"),
        help="Real multi-turn train JSONL.",
    )
    parser.add_argument(
        "--style-dev",
        default=str(ROOT / "data" / "processed" / "style_splits" / "style_dev.jsonl"),
        help="Real multi-turn dev JSONL.",
    )
    parser.add_argument(
        "--style-test",
        default=str(ROOT / "data" / "processed" / "style_splits" / "style_test.jsonl"),
        help="Real multi-turn test JSONL.",
    )
    parser.add_argument(
        "--expanded-single-turn",
        default=str(ROOT / "data" / "processed" / "style_sft_from_single_turn.jsonl"),
        help="Expanded single-turn auxiliary JSONL.",
    )
    parser.add_argument(
        "--preference",
        default=str(ROOT / "data" / "processed" / "style_preference" / "style_dpo_template.jsonl"),
        help="Preference template JSONL.",
    )
    parser.add_argument(
        "--outdir",
        default=str(ROOT / "data" / "training" / "style_first_pack"),
        help="Output directory.",
    )
    parser.add_argument("--synthetic-ratio", type=float, default=0.25, help="Synthetic-to-real ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    manifest = build_style_training_pack(
        args.style_train,
        args.style_dev,
        args.style_test,
        args.expanded_single_turn,
        args.preference,
        args.outdir,
        synthetic_ratio=args.synthetic_ratio,
        seed=args.seed,
    )
    print(json.dumps(manifest["phase1"], ensure_ascii=False))


if __name__ == "__main__":
    main()
