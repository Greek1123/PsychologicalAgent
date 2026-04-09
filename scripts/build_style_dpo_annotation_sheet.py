from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from campus_support_agent.style_preference_annotation import build_style_dpo_annotation_sheet


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CSV sheet for style DPO rejected annotation.")
    parser.add_argument(
        "--input",
        default=str(ROOT / "data" / "training" / "style_first_pack" / "style_phase2_preference.jsonl"),
        help="Input preference JSONL path.",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "data" / "processed" / "style_preference" / "style_dpo_annotation_sheet.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--no-candidate",
        action="store_true",
        help="Leave candidate_rejected blank.",
    )
    args = parser.parse_args()

    count = build_style_dpo_annotation_sheet(
        args.input,
        args.out,
        include_candidate=not args.no_candidate,
    )
    print(f"wrote {count} rows")


if __name__ == "__main__":
    main()
