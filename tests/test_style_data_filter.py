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

from campus_support_agent.style_data_filter import classify_style_sample, triage_style_dataset


class StyleDataFilterTests(unittest.TestCase):
    def test_good_style_sample_goes_to_keep(self) -> None:
        sample = {
            "id": "sample-1",
            "messages": [
                {"role": "system", "content": "你是一个支持助手。"},
                {"role": "user", "content": "最近考试很多，我很焦虑。"},
                {"role": "assistant", "content": "听起来你最近一直绷得很紧。"},
                {"role": "user", "content": "是的，晚上也睡不好。"},
                {"role": "assistant", "content": "如果我们先只看今晚最难受的部分，你觉得是睡不着，还是一想到考试就慌？"},
            ],
        }
        result = classify_style_sample(sample)
        self.assertEqual(result["bucket"], "keep")

    def test_therapy_heavy_sample_goes_to_review(self) -> None:
        sample = {
            "id": "sample-2",
            "messages": [
                {"role": "system", "content": "你是一位精通理情行为疗法的心理咨询师。"},
                {"role": "user", "content": "我很困扰。"},
                {"role": "assistant", "content": "我们先检查你的非理性信念。"},
                {"role": "user", "content": "我不知道该怎么办。"},
                {"role": "assistant", "content": "请继续描述你的思维逻辑。"},
            ],
        }
        result = classify_style_sample(sample)
        self.assertEqual(result["bucket"], "review")

    def test_short_sample_goes_to_drop(self) -> None:
        sample = {
            "id": "sample-3",
            "messages": [
                {"role": "user", "content": "我很难受。"},
                {"role": "assistant", "content": "说说看。"},
            ],
        }
        result = classify_style_sample(sample)
        self.assertEqual(result["bucket"], "drop")

    def test_triage_style_dataset_writes_buckets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "style.jsonl"
            outdir = Path(tmpdir) / "triage"

            samples = [
                {
                    "id": "keep-1",
                    "messages": [
                        {"role": "user", "content": "最近压力很大。"},
                        {"role": "assistant", "content": "听起来你最近真的很不容易。"},
                        {"role": "user", "content": "我睡不好。"},
                        {"role": "assistant", "content": "如果先看今晚这一刻，你觉得最难受的是睡不着，还是担心明天？"},
                    ],
                },
                {
                    "id": "drop-1",
                    "messages": [
                        {"role": "user", "content": "难受。"},
                        {"role": "assistant", "content": "嗯。"},
                    ],
                },
            ]
            with input_path.open("w", encoding="utf-8") as handle:
                for sample in samples:
                    handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

            counts = triage_style_dataset(str(input_path), str(outdir))
            self.assertEqual(counts["keep"], 1)
            self.assertEqual(counts["drop"], 1)
            self.assertTrue((outdir / "style_keep.jsonl").exists())
            self.assertTrue((outdir / "style_drop.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
