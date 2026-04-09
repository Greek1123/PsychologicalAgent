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

from campus_support_agent.ms_swift_style_builder import build_ms_swift_style_datasets


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class MsSwiftStyleBuilderTests(unittest.TestCase):
    def test_build_ms_swift_style_datasets_exports_sft_and_dpo(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pack_dir = Path(tmpdir) / "style_pack"
            outdir = Path(tmpdir) / "ms_swift"
            pack_dir.mkdir(parents=True, exist_ok=True)

            sft_sample = {
                "id": "style-1",
                "messages": [
                    {"role": "system", "content": "You are supportive."},
                    {"role": "user", "content": "I feel stressed."},
                    {"role": "assistant", "content": "That sounds heavy."},
                ],
            }
            dpo_template = {
                "id": "pref-1",
                "prompt": [{"role": "user", "content": "I feel stuck."}],
                "chosen": "That makes sense. What feels hardest?",
                "rejected": "You should just calm down.",
            }
            pending_template = {
                "id": "pref-2",
                "prompt": [{"role": "user", "content": "I feel lost."}],
                "chosen": "I am here with you.",
                "rejected": "",
            }

            _write_jsonl(pack_dir / "style_phase1_train.jsonl", [sft_sample])
            _write_jsonl(pack_dir / "style_phase1_dev.jsonl", [sft_sample])
            _write_jsonl(pack_dir / "style_phase1_test.jsonl", [sft_sample])
            _write_jsonl(pack_dir / "style_phase2_preference.jsonl", [dpo_template, pending_template])

            manifest = build_ms_swift_style_datasets(str(pack_dir), str(outdir))

            self.assertEqual(manifest["phase1"]["train"], 1)
            self.assertEqual(manifest["phase2"]["ready_for_dpo"], 1)
            self.assertEqual(manifest["phase2"]["pending_rejected_annotation"], 1)

            dpo_records = [
                json.loads(line)
                for line in (outdir / "style_phase2_preference_ms_swift.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(dpo_records[0]["rejected_response"], "You should just calm down.")


if __name__ == "__main__":
    unittest.main()
