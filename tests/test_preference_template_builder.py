from __future__ import annotations

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

from campus_support_agent.preference_template_builder import build_preference_templates


@contextmanager
def _temporary_workspace_dir() -> Path:
    parent = ROOT / "tmp_test_artifacts"
    parent.mkdir(parents=True, exist_ok=True)
    tmpdir = parent / f"pref-builder-{uuid4().hex}"
    tmpdir.mkdir(parents=True, exist_ok=True)
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


class PreferenceTemplateBuilderTests(unittest.TestCase):
    def test_build_preference_templates_from_raw_style_sample(self) -> None:
        with _temporary_workspace_dir() as tmpdir:
            input_path = tmpdir / "style_review.jsonl"
            output_path = tmpdir / "style_dpo_template.jsonl"

            sample = {
                "id": "style-1",
                "language": "zh",
                "stage_goal": "gentle_probe_and_reframe",
                "messages": [
                    {"role": "system", "content": "You are a support assistant."},
                    {"role": "user", "content": "I have been under a lot of stress lately."},
                    {"role": "assistant", "content": "That sounds exhausting. What feels heaviest right now?"},
                ],
            }
            input_path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

            count = build_preference_templates(str(input_path), str(output_path), limit=10)
            self.assertEqual(count, 1)

            row = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["task_type"], "style_preference")
            self.assertEqual(row["chosen"], "That sounds exhausting. What feels heaviest right now?")
            self.assertEqual(row["rejected"], "")

    def test_build_preference_templates_from_triaged_review_record(self) -> None:
        with _temporary_workspace_dir() as tmpdir:
            input_path = tmpdir / "style_review.jsonl"
            output_path = tmpdir / "style_dpo_template.jsonl"

            wrapped_sample = {
                "bucket": "review",
                "quality_score": 5,
                "quality_reasons": ["therapy_heavy", "needs softer phrasing"],
                "sample": {
                    "id": "style-2",
                    "language": "en",
                    "stage_goal": "summary_and_next_step",
                    "task_type": "style_support",
                    "messages": [
                        {"role": "system", "content": "You are a support assistant."},
                        {"role": "user", "content": "I still feel stuck in this situation."},
                        {"role": "assistant", "content": "It makes sense that you feel stuck. What have you already tried?"},
                        {"role": "user", "content": "Mostly just thinking about it alone."},
                    ],
                },
            }
            input_path.write_text(json.dumps(wrapped_sample, ensure_ascii=False) + "\n", encoding="utf-8")

            count = build_preference_templates(str(input_path), str(output_path), limit=10)
            self.assertEqual(count, 1)

            row = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["chosen"], "It makes sense that you feel stuck. What have you already tried?")
            self.assertEqual(row["review_notes"], "therapy_heavy; needs softer phrasing")
            self.assertEqual(row["meta"]["source_bucket"], "review")


if __name__ == "__main__":
    unittest.main()
