from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.human_eval_builder import build_human_eval_sheet


class HumanEvalBuilderTests(unittest.TestCase):
    def test_build_human_eval_sheet_outputs_csv_from_wrapped_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "style_test.jsonl"
            output_path = Path(tmpdir) / "eval.csv"

            sample = {
                "bucket": "keep",
                "quality_score": 7,
                "quality_reasons": [],
                "sample": {
                    "id": "style-1",
                    "language": "zh",
                    "task_type": "style_support",
                    "stage_goal": "summary_and_next_step",
                    "messages": [
                        {"role": "user", "content": "最近我总是睡不好。"},
                        {"role": "assistant", "content": "听起来这段时间你一直很绷着。"},
                    ],
                },
            }
            input_path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

            count = build_human_eval_sheet(str(input_path), str(output_path), limit=10)
            self.assertEqual(count, 1)

            with output_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["id"], "style-1")
            self.assertIn("最近我总是睡不好。", rows[0]["user_turns"])
            self.assertIn("听起来这段时间你一直很绷着。", rows[0]["assistant_turns"])


if __name__ == "__main__":
    unittest.main()
