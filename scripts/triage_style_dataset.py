from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# 这个脚本负责把 style 训练集分成 keep / review / drop 三类，便于人工复核。
from campus_support_agent.style_data_filter import triage_style_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Triage style dataset quality.")
    parser.add_argument(
        "--input",
        default=str(ROOT / "data" / "processed" / "style_sft_bilingual.jsonl"),
        help="Input style SFT JSONL path.",
    )
    parser.add_argument(
        "--outdir",
        default=str(ROOT / "data" / "processed" / "triaged_style"),
        help="Output directory.",
    )
    args = parser.parse_args()

    counts = triage_style_dataset(args.input, args.outdir)
    print(counts)


if __name__ == "__main__":
    main()
