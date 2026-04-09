from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from campus_support_agent.weak_input_builder import build_weak_input_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a weak-input repair dataset for chat-first SFT.")
    parser.add_argument(
        "--out",
        default=str(ROOT / "data" / "training" / "weak_input" / "weak_input_phase0_5_train_ms_swift.jsonl"),
        help="Output JSONL path.",
    )
    args = parser.parse_args()

    stats = build_weak_input_dataset(args.out)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
