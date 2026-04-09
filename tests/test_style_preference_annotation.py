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

from campus_support_agent.style_preference_annotation import build_style_dpo_annotation_sheet


@contextmanager
def _temporary_workspace_dir() -> Path:
    parent = ROOT / "tmp_test_artifacts"
    parent.mkdir(parents=True, exist_ok=True)
    tmpdir = parent / f"pref-annotation-{uuid4().hex}"
    tmpdir.mkdir(parents=True, exist_ok=True)
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


class StylePreferenceAnnotationTests(unittest.TestCase):
    def test_build_annotation_sheet_with_candidates(self) -> None:
        with _temporary_workspace_dir() as tmpdir:
            input_path = tmpdir / "style_phase2_preference.jsonl"
            output_path = tmpdir / "style_dpo_annotation_sheet.csv"

            record = {
                "id": "pref-1",
                "language": "zh",
                "stage_goal": "gentle_probe_and_reframe",
                "prompt": [
                    {"role": "user", "content": "最近考试很多，我睡不好。"},
                ],
                "chosen": "听起来你最近一直很紧绷。我们可以先看看今晚最困扰你的是什么？",
                "rejected": "",
                "review_notes": "therapy_heavy; needs softer phrasing",
            }
            input_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

            count = build_style_dpo_annotation_sheet(str(input_path), str(output_path))
            self.assertEqual(count, 1)

            with output_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            self.assertIn("formulaic_opening", rows[0]["failure_modes"])
            self.assertIn("therapy_heavy", rows[0]["failure_modes"])
            self.assertTrue(rows[0]["candidate_rejected"])

    def test_build_annotation_sheet_without_candidates(self) -> None:
        with _temporary_workspace_dir() as tmpdir:
            input_path = tmpdir / "style_phase2_preference.jsonl"
            output_path = tmpdir / "style_dpo_annotation_sheet.csv"

            record = {
                "id": "pref-2",
                "language": "en",
                "stage_goal": "summary_and_next_step",
                "prompt": [
                    {"role": "user", "content": "I do not know how to calm down."},
                ],
                "chosen": "It sounds like you have been carrying a lot. What feels most intense right now?",
                "rejected": "",
                "review_notes": "",
            }
            input_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

            count = build_style_dpo_annotation_sheet(str(input_path), str(output_path), include_candidate=False)
            self.assertEqual(count, 1)

            with output_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(rows[0]["candidate_rejected"], "")


if __name__ == "__main__":
    unittest.main()
