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

from campus_support_agent.weak_input_builder import build_weak_input_dataset


@contextmanager
def _temporary_workspace_dir() -> Path:
    parent = ROOT / "tmp_test_artifacts"
    parent.mkdir(parents=True, exist_ok=True)
    tmpdir = parent / f"weak-input-{uuid4().hex}"
    tmpdir.mkdir(parents=True, exist_ok=True)
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


class WeakInputBuilderTests(unittest.TestCase):
    def test_build_weak_input_dataset_writes_bilingual_records(self) -> None:
        with _temporary_workspace_dir() as root:
            output_path = root / "weak_input.jsonl"

            stats = build_weak_input_dataset(str(output_path))

            self.assertTrue(output_path.exists())
            self.assertGreater(stats["written"], 0)
            self.assertGreater(stats["zh"], 0)
            self.assertGreater(stats["en"], 0)

            records = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(all(record["stage_goal"] == "weak_input_repair" for record in records))
            self.assertTrue(all(record["messages"][0]["role"] == "user" for record in records))


if __name__ == "__main__":
    unittest.main()
