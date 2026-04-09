from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# 这个脚本用来把原始中英多轮对话转换成可直接训练的 style_sft.jsonl。
from campus_support_agent.style_dataset_builder import write_style_sft_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bilingual style SFT dataset from raw JSON dialogues.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            str(ROOT / "data" / "cn_data_version7.json"),
            str(ROOT / "data" / "en_data_version7.json"),
        ],
        help="Input dataset files.",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "data" / "processed" / "style_sft_bilingual.jsonl"),
        help="Output JSONL path.",
    )
    args = parser.parse_args()

    count = write_style_sft_dataset(args.inputs, args.out)
    print(f"wrote {count} samples")


if __name__ == "__main__":
    main()
