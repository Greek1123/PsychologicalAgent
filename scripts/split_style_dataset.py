from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# 这个脚本把已经清洗好的 style 数据切成 train/dev/test。
from campus_support_agent.style_dataset_splitter import split_style_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Split cleaned style dataset into train/dev/test.")
    parser.add_argument(
        "--input",
        default=str(ROOT / "data" / "processed" / "triaged_style" / "style_keep.jsonl"),
        help="Input cleaned style JSONL path.",
    )
    parser.add_argument(
        "--outdir",
        default=str(ROOT / "data" / "processed" / "style_splits"),
        help="Output directory.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    counts = split_style_dataset(args.input, args.outdir, seed=args.seed)
    print(counts)


if __name__ == "__main__":
    main()
