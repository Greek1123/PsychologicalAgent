from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .logging_utils import get_logger


logger = get_logger("style_preference_merge")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_annotation_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {str(row.get("id", "")).strip(): row for row in rows if str(row.get("id", "")).strip()}


def apply_style_dpo_annotations(
    preference_jsonl_path: str,
    annotation_csv_path: str,
    output_jsonl_path: str,
) -> dict[str, int]:
    source = Path(preference_jsonl_path)
    annotation_csv = Path(annotation_csv_path)
    output = Path(output_jsonl_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    records = _read_jsonl(source)
    annotations = _read_annotation_csv(annotation_csv)

    updated = 0
    kept_blank = 0
    written = 0

    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            record_id = str(record.get("id", "")).strip()
            annotation = annotations.get(record_id)
            merged = dict(record)

            if annotation:
                rejected = str(annotation.get("rejected", "")).strip()
                if rejected:
                    merged["rejected"] = rejected
                    meta = dict(merged.get("meta") or {})
                    annotator_notes = str(annotation.get("annotator_notes", "")).strip()
                    failure_modes = str(annotation.get("failure_modes", "")).strip()
                    if annotator_notes:
                        meta["annotator_notes"] = annotator_notes
                    if failure_modes:
                        meta["failure_modes"] = failure_modes.split("|")
                    merged["meta"] = meta
                    updated += 1
                else:
                    kept_blank += 1

            handle.write(json.dumps(merged, ensure_ascii=False) + "\n")
            written += 1

    stats = {
        "written": written,
        "updated_with_rejected": updated,
        "left_blank": kept_blank,
    }
    logger.info(
        "Applied DPO annotations from %s into %s (written=%s updated=%s blank=%s)",
        annotation_csv,
        output,
        written,
        updated,
        kept_blank,
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge edited DPO annotation CSV back into preference JSONL.")
    parser.add_argument("--input", required=True, help="Original preference JSONL path.")
    parser.add_argument("--annotations", required=True, help="Edited annotation CSV path.")
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    args = parser.parse_args()

    stats = apply_style_dpo_annotations(args.input, args.annotations, args.out)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
