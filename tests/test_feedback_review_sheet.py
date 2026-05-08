from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.feedback_review_sheet import apply_feedback_review_sheet, build_feedback_review_sheet


def _work_dir() -> Path:
    path = ROOT / "tmp_test_artifacts" / f"feedback_review_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class FeedbackReviewSheetTests(unittest.TestCase):
    def test_build_and_apply_feedback_review_sheet(self) -> None:
        work_dir = _work_dir()
        review_cases = work_dir / "review_cases.jsonl"
        sheet = work_dir / "review_sheet.csv"
        reviewed = work_dir / "reviewed_cases.jsonl"
        _write_jsonl(
            review_cases,
            [
                {
                    "id": "review_case_1",
                    "response_id": "resp-1",
                    "session_id": "student-001",
                    "input_text": "我不想说，我怕别人知道。",
                    "assistant_reply": "那我们先换个话题吧。",
                    "risk": {"level": "low"},
                    "entropy": {"score": 30},
                    "local_policy": {"policy_name": "privacy_boundary"},
                    "failure_review": {"preferred_reply": "", "rewrite_needed": False},
                    "feedback": {"tags": []},
                    "sft_draft": {
                        "messages": [{"role": "user", "content": "我不想说，我怕别人知道。"}],
                        "rejected": "那我们先换个话题吧。",
                        "chosen": "",
                    },
                }
            ],
        )

        count = build_feedback_review_sheet(str(review_cases), str(sheet))

        self.assertEqual(count, 1)
        with sheet.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["response_id"], "resp-1")
        rows[0]["mark_bad"] = "1"
        rows[0]["problem_tags"] = "privacy_missed,too_short"
        rows[0]["chosen"] = "这里的对话不会主动告诉你的舍友。你不想细说也可以，我们先保护你的边界。"
        rows[0]["review_note"] = "原回复回避了隐私担心"
        with sheet.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        stats = apply_feedback_review_sheet(str(review_cases), str(sheet), str(reviewed))

        self.assertEqual(stats["marked_bad"], 1)
        self.assertEqual(stats["updated_with_chosen"], 1)
        updated = json.loads(reviewed.read_text(encoding="utf-8").splitlines()[0])
        self.assertTrue(updated["failure_review"]["rewrite_needed"])
        self.assertEqual(updated["failure_review"]["review_status"], "reviewed")
        self.assertEqual(updated["feedback"]["tags"], ["privacy_missed", "too_short"])
        self.assertIn("不会主动告诉", updated["sft_draft"]["chosen"])


if __name__ == "__main__":
    unittest.main()
