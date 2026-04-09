from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.dataset_templates import write_bilingual_training_templates
from campus_support_agent.storage import SQLiteSessionStore
from campus_support_agent.training_export import export_training_dataset


class TrainingExportTests(unittest.TestCase):
    def test_export_training_dataset_writes_sft_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "agent.db"
            output_path = Path(tmpdir) / "export" / "train.jsonl"

            store = SQLiteSessionStore(str(db_path), max_messages=8)
            response_payload = {
                "risk": {"level": "medium"},
                "entropy": {"score": 54, "level": 3, "balance_state": "strained"},
                "entropy_reduction": {"target_state": "stable", "expected_delta_score": -10},
                "assessment": {"primary_emotions": ["焦虑"]},
                "plan": {"summary": "先稳住节律"},
                "campus_resources": [],
                "safety": {"disclaimer": "test"},
            }
            store.store_support_response(
                session_id="student-001",
                response_id="resp-1",
                source="text",
                input_text="最近考试很多，我睡不好。",
                transcript=None,
                student_context={"grade": "大二"},
                conversation_history=[{"role": "assistant", "content": "之前我们聊过睡眠问题。"}],
                response_payload=response_payload,
            )

            count = export_training_dataset(
                db_path=str(db_path),
                output_path=str(output_path),
                export_format="sft",
            )

            self.assertEqual(count, 1)
            lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            sample = json.loads(lines[0])
            self.assertEqual(sample["id"], "resp-1")
            self.assertEqual(sample["session_id"], "student-001")
            self.assertEqual(sample["language"], "zh")
            self.assertEqual(sample["task_type"], "analysis_support")
            self.assertEqual(sample["messages"][-1]["role"], "assistant")

    def test_export_training_dataset_writes_record_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "agent.db"
            output_path = Path(tmpdir) / "record.jsonl"

            store = SQLiteSessionStore(str(db_path), max_messages=8)
            response_payload = {
                "risk": {"level": "low"},
                "entropy": {"score": 20, "level": 1, "balance_state": "stable"},
                "entropy_reduction": {"target_state": "stable", "expected_delta_score": -6},
                "assessment": {"primary_emotions": ["平静"]},
                "plan": {"summary": "继续观察"},
                "campus_resources": [],
                "safety": {"disclaimer": "test"},
            }
            store.store_support_response(
                session_id="student-002",
                response_id="resp-2",
                source="text",
                input_text="今天心情还可以。",
                transcript=None,
                student_context={},
                conversation_history=[],
                response_payload=response_payload,
            )

            count = export_training_dataset(
                db_path=str(db_path),
                output_path=str(output_path),
                export_format="record",
            )

            self.assertEqual(count, 1)
            sample = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(sample["record_id"], "resp-2")
            self.assertEqual(sample["language"], "zh")
            self.assertEqual(sample["messages"][-1]["content"], "今天心情还可以。")
            self.assertIn("target", sample)

    def test_write_bilingual_training_templates_creates_template_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_bilingual_training_templates(tmpdir)

            self.assertEqual(len(paths), 3)
            style_sft = Path(tmpdir) / "style_sft_template.jsonl"
            self.assertTrue(style_sft.exists())

            samples = [json.loads(line) for line in style_sft.read_text(encoding="utf-8").splitlines()]
            languages = {sample["language"] for sample in samples}
            self.assertEqual(languages, {"zh", "en"})


if __name__ == "__main__":
    unittest.main()
