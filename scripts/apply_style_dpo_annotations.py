from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from campus_support_agent.style_preference_merge import apply_style_dpo_annotations


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply edited DPO annotation CSV back to preference JSONL.")
    parser.add_argument(
        "--input",
        default=str(ROOT / "data" / "training" / "style_first_pack" / "style_phase2_preference.jsonl"),
        help="Original preference JSONL path.",
    )
    parser.add_argument(
        "--annotations",
        default=str(ROOT / "data" / "processed" / "style_preference" / "style_dpo_annotation_sheet.csv"),
        help="Edited annotation CSV path.",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "data" / "training" / "style_first_pack" / "style_phase2_preference_annotated.jsonl"),
        help="Output JSONL path.",
    )
    args = parser.parse_args()

    stats = apply_style_dpo_annotations(args.input, args.annotations, args.out)
    print(stats)


if __name__ == "__main__":
    main()
