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

from campus_support_agent.style_dataset_splitter import split_style_dataset


class StyleDatasetSplitterTests(unittest.TestCase):
    def test_split_style_dataset_creates_three_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "style_keep.jsonl"
            outdir = Path(tmpdir) / "splits"

            samples = []
            for idx in range(10):
                samples.append(
                    {
                        "bucket": "keep",
                        "quality_score": 7,
                        "quality_reasons": [],
                        "sample": {
                            "id": f"zh-{idx}",
                            "language": "zh",
                            "stage_goal": "gentle_probe_and_reframe",
                            "messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
                        },
                    }
                )
                samples.append(
                    {
                        "bucket": "keep",
                        "quality_score": 7,
                        "quality_reasons": [],
                        "sample": {
                            "id": f"en-{idx}",
                            "language": "en",
                            "stage_goal": "emotional_containment",
                            "messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
                        },
                    }
                )

            with input_path.open("w", encoding="utf-8") as handle:
                for sample in samples:
                    handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

            counts = split_style_dataset(str(input_path), str(outdir), seed=7)
            self.assertEqual(sum(counts.values()), 20)
            self.assertTrue((outdir / "style_train.jsonl").exists())
            self.assertTrue((outdir / "style_dev.jsonl").exists())
            self.assertTrue((outdir / "style_test.jsonl").exists())

            first_record = json.loads((outdir / "style_train.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("messages", first_record)
            self.assertNotIn("sample", first_record)


if __name__ == "__main__":
    unittest.main()
