from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.dataset_templates import write_bilingual_training_templates
from campus_support_agent.storage import SQLiteSessionStore
from campus_support_agent.training_export import export_training_dataset


def _work_dir() -> Path:
    path = ROOT / "tmp_test_artifacts" / f"training_export_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class TrainingExportTests(unittest.TestCase):
    def test_export_training_dataset_writes_sft_jsonl(self) -> None:
        work_dir = _work_dir()
        db_path = work_dir / "agent.db"
        output_path = work_dir / "export" / "train.jsonl"

        store = SQLiteSessionStore(str(db_path), max_messages=8)
        store.store_support_response(
            session_id="student-001",
            response_id="resp-1",
            source="text",
            input_text="我最近考试很多，晚上总是睡不好。",
            transcript=None,
            student_context={"grade": "大二"},
            conversation_history=[{"role": "assistant", "content": "我在，你可以慢慢说。"}],
            response_payload={
                "risk": {"level": "medium"},
                "entropy": {"score": 54, "level": 3, "balance_state": "strained"},
                "entropy_reduction": {"target_state": "stable", "expected_delta_score": -10},
                "assessment": {"primary_emotions": ["焦虑"]},
                "plan": {"summary": "先稳定睡眠和任务节奏。"},
                "campus_resources": [],
                "safety": {"disclaimer": "test"},
            },
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
        work_dir = _work_dir()
        db_path = work_dir / "agent.db"
        output_path = work_dir / "record.jsonl"

        store = SQLiteSessionStore(str(db_path), max_messages=8)
        store.store_support_response(
            session_id="student-002",
            response_id="resp-2",
            source="text",
            input_text="今天心情还可以。",
            transcript=None,
            student_context={},
            conversation_history=[],
            response_payload={
                "risk": {"level": "low"},
                "entropy": {"score": 20, "level": 1, "balance_state": "stable"},
                "entropy_reduction": {"target_state": "stable", "expected_delta_score": -6},
                "assessment": {"primary_emotions": ["平静"]},
                "plan": {"summary": "保持当前节奏。"},
                "campus_resources": [],
                "safety": {"disclaimer": "test"},
            },
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

    def test_export_training_dataset_writes_bad_case_jsonl(self) -> None:
        work_dir = _work_dir()
        db_path = work_dir / "agent.db"
        output_path = work_dir / "bad_cases.jsonl"

        store = SQLiteSessionStore(str(db_path), max_messages=8)
        store.store_support_response(
            session_id="student-003",
            response_id="resp-bad",
            source="text",
            input_text="我不想说，我怕别人知道。",
            transcript=None,
            student_context={},
            conversation_history=[{"role": "assistant", "content": "你可以慢慢说。"}],
            response_payload={
                "reply_text": "那我们先换个话题吧。",
                "risk": {"level": "low"},
                "entropy": {"score": 30, "level": 2},
                "local_policy": {"policy_name": "privacy_boundary"},
            },
        )
        store.store_support_response(
            session_id="student-003",
            response_id="resp-good",
            source="text",
            input_text="谢谢你。",
            transcript=None,
            student_context={},
            conversation_history=[],
            response_payload={"reply_text": "不客气。", "risk": {"level": "low"}},
        )
        store.append_intervention_feedback(
            session_id="student-003",
            response_id="resp-bad",
            helpful_score=-2,
            mood_after=35,
            user_note="没有回应隐私担心",
            tags=["privacy_missed", "too_short"],
        )
        store.append_intervention_feedback(
            session_id="student-003",
            response_id="resp-good",
            helpful_score=2,
            tags=["helpful"],
        )

        count = export_training_dataset(
            db_path=str(db_path),
            output_path=str(output_path),
            export_format="bad_case",
        )

        self.assertEqual(count, 1)
        sample = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(sample["response_id"], "resp-bad")
        self.assertEqual(sample["feedback"]["helpful_score"], -2)
        self.assertEqual(sample["assistant_reply"], "那我们先换个话题吧。")
        self.assertTrue(sample["failure_review"]["rewrite_needed"])
        self.assertEqual(sample["sft_draft"]["rejected"], "那我们先换个话题吧。")
        self.assertEqual(sample["sft_draft"]["chosen"], "")

    def test_export_training_dataset_writes_review_case_jsonl_without_feedback(self) -> None:
        work_dir = _work_dir()
        db_path = work_dir / "agent.db"
        output_path = work_dir / "review_cases.jsonl"

        store = SQLiteSessionStore(str(db_path), max_messages=8)
        store.store_support_response(
            session_id="student-004",
            response_id="resp-review",
            source="text",
            input_text="我一回宿舍就很烦。",
            transcript=None,
            student_context={},
            conversation_history=[{"role": "assistant", "content": "我在听。"}],
            response_payload={
                "reply_text": "你应该主动沟通，不要太敏感。",
                "risk": {"level": "medium"},
                "entropy": {"score": 45, "level": 3},
                "local_policy": {"policy_name": "interpersonal_stress"},
            },
        )

        count = export_training_dataset(
            db_path=str(db_path),
            output_path=str(output_path),
            export_format="review_case",
        )

        self.assertEqual(count, 1)
        sample = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(sample["response_id"], "resp-review")
        self.assertEqual(sample["feedback"]["source"], "manual_review")
        self.assertEqual(sample["assistant_reply"], "你应该主动沟通，不要太敏感。")
        self.assertFalse(sample["failure_review"]["rewrite_needed"])
        self.assertEqual(sample["sft_draft"]["rejected"], "你应该主动沟通，不要太敏感。")
        self.assertEqual(sample["sft_draft"]["chosen"], "")

    def test_write_bilingual_training_templates_creates_template_files(self) -> None:
        work_dir = _work_dir()
        paths = write_bilingual_training_templates(str(work_dir))

        self.assertEqual(len(paths), 3)
        style_sft = work_dir / "style_sft_template.jsonl"
        self.assertTrue(style_sft.exists())

        samples = [json.loads(line) for line in style_sft.read_text(encoding="utf-8").splitlines()]
        languages = {sample["language"] for sample in samples}
        self.assertEqual(languages, {"zh", "en"})


if __name__ == "__main__":
    unittest.main()
