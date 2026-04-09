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

from campus_support_agent.style_dataset_builder import (
    build_style_sft_samples_from_file,
    write_style_sft_dataset,
)


class StyleDatasetBuilderTests(unittest.TestCase):
    def test_build_style_sft_samples_from_cn_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cn_data_version7.json"
            payload = [
                {
                    "dialog_id": "1",
                    "stage": "Early Session",
                    "topic": "Academic Stress",
                    "dialog": [
                        {"speaker": "Seeker", "content": "最近考试很多，我很焦虑。"},
                        {"speaker": "Supporter", "content": "听起来你最近压力很大。"},
                        {"speaker": "Seeker", "content": "对，而且晚上睡不好。"},
                        {"speaker": "Supporter", "content": "我们可以先看看今晚最困扰你的部分。"},
                    ],
                }
            ]
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            samples = build_style_sft_samples_from_file(str(path))
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["language"], "zh")
            self.assertEqual(samples[0]["task_type"], "style_support")
            self.assertEqual(samples[0]["messages"][0]["role"], "system")

    def test_short_dialog_is_filtered(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "en_data_version7.json"
            payload = [
                {
                    "dialog_id": "2",
                    "stage": "",
                    "dialog": [
                        {"speaker": "Seeker", "content": "I feel bad."},
                        {"speaker": "Supporter", "content": "Tell me more."},
                    ],
                }
            ]
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            samples = build_style_sft_samples_from_file(str(path))
            self.assertEqual(samples, [])

    def test_write_style_sft_dataset_outputs_bilingual_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cn_path = Path(tmpdir) / "cn_data_version7.json"
            en_path = Path(tmpdir) / "en_data_version7.json"
            out_path = Path(tmpdir) / "style_sft_bilingual.jsonl"

            cn_payload = [
                {
                    "dialog_id": "1",
                    "stage": "Early Session",
                    "dialog": [
                        {"speaker": "Seeker", "content": "最近压力很大。"},
                        {"speaker": "Supporter", "content": "听起来你最近一直很紧绷。"},
                        {"speaker": "Seeker", "content": "是的，我晚上也睡不好。"},
                        {"speaker": "Supporter", "content": "我们先从今晚最难受的地方开始。"},
                    ],
                }
            ]
            en_payload = [
                {
                    "dialog_id": "2",
                    "stage": "Late Session",
                    "dialog": [
                        {"speaker": "Seeker", "content": "I feel overwhelmed lately."},
                        {"speaker": "Supporter", "content": "That sounds exhausting to carry."},
                        {"speaker": "Seeker", "content": "Yes, especially at night."},
                        {"speaker": "Supporter", "content": "Let us slow it down and name the hardest part first."},
                    ],
                }
            ]
            cn_path.write_text(json.dumps(cn_payload, ensure_ascii=False), encoding="utf-8")
            en_path.write_text(json.dumps(en_payload, ensure_ascii=False), encoding="utf-8")

            count = write_style_sft_dataset([str(cn_path), str(en_path)], str(out_path))
            self.assertEqual(count, 2)

            lines = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
            languages = {line["language"] for line in lines}
            self.assertEqual(languages, {"zh", "en"})


if __name__ == "__main__":
    unittest.main()
