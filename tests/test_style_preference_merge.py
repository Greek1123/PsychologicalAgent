from __future__ import annotations

import csv
import json
import shutil
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.style_preference_merge import apply_style_dpo_annotations


@contextmanager
def _temporary_workspace_dir() -> Path:
    parent = ROOT / "tmp_test_artifacts"
    parent.mkdir(parents=True, exist_ok=True)
    tmpdir = parent / f"pref-merge-{uuid4().hex}"
    tmpdir.mkdir(parents=True, exist_ok=True)
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


class StylePreferenceMergeTests(unittest.TestCase):
    def test_apply_annotations_updates_rejected_and_meta(self) -> None:
        with _temporary_workspace_dir() as tmpdir:
            input_path = tmpdir / "style_phase2_preference.jsonl"
            csv_path = tmpdir / "style_dpo_annotation_sheet.csv"
            output_path = tmpdir / "style_phase2_preference_annotated.jsonl"

            record = {
                "id": "pref-1",
                "chosen": "That sounds exhausting. What feels heaviest right now?",
                "rejected": "",
                "meta": {},
            }
            input_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

            with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "id",
                        "failure_modes",
                        "rejected",
                        "annotator_notes",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "id": "pref-1",
                        "failure_modes": "formulaic_opening|generic_reassurance",
                        "rejected": "It sounds like you are under a lot of pressure. Do not worry too much.",
                        "annotator_notes": "Too canned.",
                    }
                )

            stats = apply_style_dpo_annotations(str(input_path), str(csv_path), str(output_path))
            self.assertEqual(stats["updated_with_rejected"], 1)

            merged = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(
                merged["rejected"],
                "It sounds like you are under a lot of pressure. Do not worry too much.",
            )
            self.assertEqual(merged["meta"]["annotator_notes"], "Too canned.")
            self.assertEqual(
                merged["meta"]["failure_modes"],
                ["formulaic_opening", "generic_reassurance"],
            )


if __name__ == "__main__":
    unittest.main()
