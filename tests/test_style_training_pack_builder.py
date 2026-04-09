from __future__ import annotations

import json
import sys
import shutil
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.style_training_pack_builder import build_style_training_pack


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class StyleTrainingPackBuilderTests(unittest.TestCase):
    def test_build_style_training_pack_writes_phase_files(self) -> None:
        temp_root = ROOT / "tmp_test_artifacts"
        temp_root.mkdir(exist_ok=True)
        tmpdir = temp_root / f"style_pack_{uuid.uuid4().hex}"
        tmpdir.mkdir(parents=True, exist_ok=True)
        try:
            root = tmpdir
            style_train = root / "style_train.jsonl"
            style_dev = root / "style_dev.jsonl"
            style_test = root / "style_test.jsonl"
            expanded = root / "expanded.jsonl"
            preference = root / "preference.jsonl"
            outdir = root / "pack"

            real_samples = [
                {
                    "id": "real-1",
                    "language": "zh",
                    "task_type": "style_support",
                    "stage_goal": "gentle_probe_and_reframe",
                    "messages": [
                        {"role": "user", "content": "I feel overwhelmed."},
                        {"role": "assistant", "content": "That sounds heavy."},
                    ],
                },
                {
                    "id": "real-2",
                    "language": "en",
                    "task_type": "style_support",
                    "stage_goal": "emotional_containment",
                    "messages": [
                        {"role": "user", "content": "I cannot settle down."},
                        {"role": "assistant", "content": "Let us slow it down together."},
                    ],
                },
                {
                    "id": "real-bad-1",
                    "language": "zh",
                    "task_type": "style_support",
                    "stage_goal": "gentle_probe_and_reframe",
                    "messages": [
                        {"role": "assistant", "content": "你愿意继续说说吗？"},
                        {"role": "user", "content": "嗯"},
                        {"role": "assistant", "content": "我在听。"},
                    ],
                },
                {
                    "id": "real-bad-2",
                    "language": "en",
                    "task_type": "style_support",
                    "stage_goal": "emotional_containment",
                    "messages": [
                        {"role": "system", "content": "Be warm."},
                        {"role": "assistant", "content": "Tell me more."},
                        {"role": "user", "content": "1"},
                        {"role": "assistant", "content": "2"},
                    ],
                },
            ]
            dev_samples = [real_samples[0]]
            test_samples = [real_samples[1]]
            expanded_samples = [
                {
                    "id": "synthetic-1",
                    "language": "zh",
                    "task_type": "style_support",
                    "stage_goal": "gentle_probe_and_reframe",
                    "messages": [
                        {"role": "user", "content": "I am tired."},
                        {"role": "assistant", "content": "What part feels hardest?"},
                    ],
                },
                {
                    "id": "synthetic-2",
                    "language": "zh",
                    "task_type": "style_support",
                    "stage_goal": "gentle_probe_and_reframe",
                    "messages": [
                        {"role": "user", "content": "I am anxious."},
                        {"role": "assistant", "content": "What seems to be driving it?"},
                    ],
                },
            ]
            preference_samples = [
                {
                    "id": "pref-1",
                    "language": "zh",
                    "task_type": "style_preference",
                    "stage_goal": "gentle_probe_and_reframe",
                    "prompt": [{"role": "user", "content": "I feel bad."}],
                    "chosen": "That sounds hard.",
                    "rejected": "",
                    "review_notes": "",
                }
            ]

            _write_jsonl(style_train, real_samples)
            _write_jsonl(style_dev, dev_samples)
            _write_jsonl(style_test, test_samples)
            _write_jsonl(expanded, expanded_samples)
            _write_jsonl(preference, preference_samples)

            manifest = build_style_training_pack(
                str(style_train),
                str(style_dev),
                str(style_test),
                str(expanded),
                str(preference),
                str(outdir),
                synthetic_ratio=0.5,
                seed=7,
            )

            self.assertEqual(manifest["phase1"]["raw_real_multiturn_train"], 4)
            self.assertEqual(manifest["phase1"]["clean_real_multiturn_train"], 2)
            self.assertEqual(manifest["phase1"]["synthetic_train"], 1)
            self.assertEqual(manifest["phase1"]["final_train"], 3)
            self.assertEqual(manifest["phase1"]["dropped_dialogues"]["train"]["weak_user_start"], 2)
            self.assertEqual(manifest["phase2"]["preference_candidates"], 1)
            self.assertTrue((outdir / "style_phase1_train.jsonl").exists())
            self.assertTrue((outdir / "style_phase2_preference.jsonl").exists())
            self.assertTrue((outdir / "style_training_manifest.json").exists())

            train_records = [
                json.loads(line)
                for line in (outdir / "style_phase1_train.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(
                all(
                    next(
                        (message["role"] for message in record["messages"] if message["role"] != "system"),
                        None,
                    )
                    == "user"
                    for record in train_records
                )
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
