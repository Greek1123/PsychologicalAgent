from __future__ import annotations

import csv
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from campus_support_agent.feedback_ai_reviewer import autofill_review_sheet_with_ai


class FeedbackAiReviewerTest(unittest.TestCase):
    def _work_dir(self) -> Path:
        path = ROOT / "tmp_test_artifacts" / f"feedback_ai_reviewer_{uuid4().hex}"
        path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

    def test_autofill_review_sheet_writes_chosen_and_tags(self) -> None:
        work_dir = self._work_dir()
        input_csv = work_dir / "review_sheet.csv"
        output_csv = work_dir / "review_sheet_ai.csv"
        with input_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "id",
                    "response_id",
                    "session_id",
                    "input_text",
                    "assistant_reply",
                    "risk_level",
                    "entropy_score",
                    "local_policy",
                    "mark_bad",
                    "problem_tags",
                    "chosen",
                    "review_note",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "id": "review_case_1",
                    "response_id": "resp-1",
                    "session_id": "student-1",
                    "input_text": "我不想说，我怕别人知道，手机号13812345678",
                    "assistant_reply": "没关系。",
                    "risk_level": "low",
                    "entropy_score": "30",
                    "local_policy": "support",
                    "mark_bad": "",
                    "problem_tags": "",
                    "chosen": "",
                    "review_note": "",
                }
            )

        captured_prompts: list[str] = []

        def fake_chat_completion(**kwargs):
            captured_prompts.append(kwargs["user_prompt"])
            return (
                '{"mark_bad":"1","problem_tags":["privacy_missed","too_short","unknown"],'
                '"chosen":"这里的对话会尊重你的隐私和边界，你不用急着说细节。",'
                '"review_note":"原回复没有回应隐私担心。"}'
            )

        with patch("campus_support_agent.feedback_ai_reviewer._chat_completion", side_effect=fake_chat_completion):
            stats = autofill_review_sheet_with_ai(
                input_csv_path=str(input_csv),
                output_csv_path=str(output_csv),
                base_url="https://example.test/v1",
                model="reviewer-model",
                api_key="test-key",
                limit=1,
            )

        self.assertEqual(stats["reviewed"], 1)
        self.assertIn("[手机号]", captured_prompts[0])
        self.assertNotIn("13812345678", captured_prompts[0])

        with output_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(rows[0]["mark_bad"], "1")
        self.assertEqual(rows[0]["problem_tags"], "privacy_missed,too_short")
        self.assertIn("尊重你的隐私", rows[0]["chosen"])
        self.assertIn("隐私担心", rows[0]["review_note"])


if __name__ == "__main__":
    unittest.main()
