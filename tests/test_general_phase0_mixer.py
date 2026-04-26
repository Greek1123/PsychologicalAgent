from __future__ import annotations

import json
import shutil
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.general_phase0_mixer import build_mixed_general_phase0_dataset


@contextmanager
def _temporary_workspace_dir() -> Path:
    parent = ROOT / "tmp_test_artifacts"
    parent.mkdir(parents=True, exist_ok=True)
    tmpdir = parent / f"general-phase0-mixer-{uuid4().hex}"
    tmpdir.mkdir(parents=True, exist_ok=True)
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _record(sample_id: str, content: str) -> dict[str, object]:
    return {
        "id": sample_id,
        "messages": [
            {"role": "user", "content": f"{content} user"},
            {"role": "assistant", "content": f"{content} assistant"},
        ],
        "meta": {"source": "test"},
    }


class GeneralPhase0MixerTests(unittest.TestCase):
    def test_build_mixed_general_phase0_dataset_keeps_ratio_and_tags_sources(self) -> None:
        with _temporary_workspace_dir() as root:
            base_input = root / "base.jsonl"
            augment_input = root / "augment.jsonl"
            output = root / "mixed.jsonl"

            base_input.write_text(
                "\n".join(json.dumps(_record(f"b-{idx}", f"base-{idx}"), ensure_ascii=False) for idx in range(10)),
                encoding="utf-8",
            )
            augment_input.write_text(
                "\n".join(
                    json.dumps(_record(f"a-{idx}", f"augment-{idx}"), ensure_ascii=False) for idx in range(20)
                ),
                encoding="utf-8",
            )

            stats = build_mixed_general_phase0_dataset(
                str(base_input),
                str(augment_input),
                str(output),
                target_total=10,
                augment_ratio=0.4,
                seed=7,
            )

            self.assertEqual(stats["written"], 10)
            self.assertEqual(stats["base_selected"], 6)
            self.assertEqual(stats["augment_selected"], 4)

            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 10)
            self.assertTrue(any(record["meta"]["mixed_source"] == "base_general_phase0" for record in records))
            self.assertTrue(any(record["meta"]["mixed_source"] == "augment_multiturn_0_8m" for record in records))

    def test_build_mixed_general_phase0_dataset_deduplicates_message_pairs(self) -> None:
        with _temporary_workspace_dir() as root:
            base_input = root / "base.jsonl"
            augment_input = root / "augment.jsonl"
            output = root / "mixed.jsonl"

            duplicate = _record("dup", "same")
            base_input.write_text(
                "\n".join(
                    [
                        json.dumps(duplicate, ensure_ascii=False),
                        json.dumps(_record("base-only", "base-only"), ensure_ascii=False),
                    ]
                ),
                encoding="utf-8",
            )
            augment_input.write_text(
                "\n".join(
                    [
                        json.dumps(duplicate, ensure_ascii=False),
                        json.dumps(_record("augment-only", "augment-only"), ensure_ascii=False),
                    ]
                ),
                encoding="utf-8",
            )

            stats = build_mixed_general_phase0_dataset(
                str(base_input),
                str(augment_input),
                str(output),
                target_total=4,
                augment_ratio=0.5,
                seed=7,
            )

            self.assertEqual(stats["written"], 3)
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            signatures = {
                tuple((message["role"], message["content"]) for message in record["messages"]) for record in records
            }
            self.assertEqual(len(signatures), 3)


if __name__ == "__main__":
    unittest.main()
