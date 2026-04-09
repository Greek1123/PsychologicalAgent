from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# 这个脚本提供一个简单入口，方便在项目根目录直接生成双语训练模板。
from campus_support_agent.dataset_templates import write_bilingual_training_templates


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate bilingual training dataset templates.")
    parser.add_argument(
        "--out",
        dest="output_dir",
        default=str(ROOT / "data" / "training_templates"),
        help="Output directory for template JSONL files.",
    )
    args = parser.parse_args()

    paths = write_bilingual_training_templates(args.output_dir)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
