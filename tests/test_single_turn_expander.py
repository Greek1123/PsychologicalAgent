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

from campus_support_agent.single_turn_expander import expand_single_turn_dataset


class SingleTurnExpanderTests(unittest.TestCase):
    def test_expand_single_turn_dataset_creates_multiturn_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "single.json"
            output_path = Path(tmpdir) / "expanded.jsonl"

            payload = [
                {
                    "instruction": "Provide a helpful response.",
                    "input": "最近我感到很焦虑。",
                    "output": "听起来你最近承受了不少压力，我们可以先看看最影响你的那件事。",
                }
            ]
            input_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            count = expand_single_turn_dataset(str(input_path), str(output_path))
            self.assertEqual(count, 1)
            sample = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(sample["language"], "zh")
            self.assertEqual(len(sample["messages"]), 5)
            self.assertEqual(sample["messages"][-1]["role"], "assistant")


if __name__ == "__main__":
    unittest.main()
