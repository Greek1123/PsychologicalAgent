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

from campus_support_agent.feedback_preference_builder import build_feedback_preference_dataset


def _work_dir() -> Path:
    path = ROOT / "tmp_test_artifacts" / f"feedback_preference_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class FeedbackPreferenceBuilderTests(unittest.TestCase):
    def test_build_feedback_preference_dataset_exports_ready_pairs(self) -> None:
        work_dir = _work_dir()
        input_path = work_dir / "bad_cases.jsonl"
        preference_path = work_dir / "feedback_preference.jsonl"
        ms_swift_path = work_dir / "feedback_dpo_ms_swift.jsonl"
        _write_jsonl(
            input_path,
            [
                {
                    "id": "bad_case_1",
                    "response_id": "resp-1",
                    "session_id": "student-001",
                    "language": "zh",
                    "input_text": "我不想说，我怕别人知道。",
                    "assistant_reply": "那我们先换个话题吧。",
                    "feedback": {"helpful_score": -2, "tags": ["privacy_missed"]},
                    "risk": {"level": "low"},
                    "entropy": {"score": 30},
                    "failure_review": {
                        "preferred_reply": "你可以放心，这里的对话不会主动告诉你的同学或舍友。你不想细说也没关系，我们可以先只聊你现在最需要被保护的边界。",
                        "review_status": "reviewed",
                    },
                    "sft_draft": {
                        "messages": [
                            {"role": "assistant", "content": "你可以慢慢说。"},
                            {"role": "user", "content": "我不想说，我怕别人知道。"},
                        ],
                        "rejected": "那我们先换个话题吧。",
                        "chosen": "",
                    },
                },
                {
                    "id": "bad_case_2",
                    "response_id": "resp-2",
                    "input_text": "1",
                    "assistant_reply": "2",
                    "sft_draft": {
                        "messages": [{"role": "user", "content": "1"}],
                        "rejected": "2",
                        "chosen": "",
                    },
                },
            ],
        )

        stats = build_feedback_preference_dataset(
            str(input_path),
            str(preference_path),
            str(ms_swift_path),
        )

        self.assertEqual(stats["ready_for_dpo"], 1)
        self.assertEqual(stats["pending_chosen"], 1)
        preference = json.loads(preference_path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(preference["task_type"], "feedback_preference")
        self.assertEqual(preference["rejected"], "那我们先换个话题吧。")
        self.assertIn("不会主动告诉", preference["chosen"])
        self.assertEqual(preference["prompt"][-1]["role"], "user")
        ms_swift = json.loads(ms_swift_path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(ms_swift["messages"][-1]["role"], "assistant")
        self.assertEqual(ms_swift["rejected_response"], "那我们先换个话题吧。")

    def test_build_feedback_preference_dataset_accepts_sft_draft_chosen(self) -> None:
        work_dir = _work_dir()
        input_path = work_dir / "bad_cases.jsonl"
        preference_path = work_dir / "feedback_preference.jsonl"
        _write_jsonl(
            input_path,
            [
                {
                    "id": "bad_case_3",
                    "input_text": "我压力好大。",
                    "assistant_reply": "别担心。",
                    "sft_draft": {
                        "messages": [{"role": "user", "content": "我压力好大。"}],
                        "rejected": "别担心。",
                        "chosen": "听起来你已经撑了一阵子。我们先不急着解决全部问题，先把今晚最压着你的那件事拿出来一点点看。",
                    },
                }
            ],
        )

        stats = build_feedback_preference_dataset(str(input_path), str(preference_path))

        self.assertEqual(stats["ready_for_dpo"], 1)
        preference = json.loads(preference_path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(preference["chosen"], "听起来你已经撑了一阵子。我们先不急着解决全部问题，先把今晚最压着你的那件事拿出来一点点看。")


if __name__ == "__main__":
    unittest.main()
