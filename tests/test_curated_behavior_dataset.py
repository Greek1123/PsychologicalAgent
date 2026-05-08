from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_curated_behavior_dataset import (
    build_curated_behavior_dataset,
    build_ms_swift_messages_only_dataset,
)


class CuratedBehaviorDatasetTests(unittest.TestCase):
    def test_builds_clean_role_boundary_dataset(self) -> None:
        output = ROOT / "tmp_test_artifacts" / "curated_behavior_dataset" / "curated.jsonl"
        stats = build_curated_behavior_dataset(output, limit=80, seed=123)

        self.assertEqual(stats["written"], 80)
        records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        categories = {record["meta"]["category"] for record in records}
        self.assertIn("role_boundary_repair", categories)
        self.assertIn("privacy_boundary", categories)
        self.assertIn("noisy_input", categories)

        joined = "\n".join(
            message["content"]
            for record in records
            for message in record["messages"]
        )
        self.assertIn("我最近", joined)
        self.assertIn("你说得对", joined)
        self.assertNotIn("我也很怕挂科", joined)
        self.assertNotIn("因为我的作业还没做完", joined)

        for record in records:
            roles = [message["role"] for message in record["messages"]]
            self.assertEqual(roles[0], "user")
            self.assertIn("assistant", roles)

    def test_builds_messages_only_ms_swift_file(self) -> None:
        root = ROOT / "tmp_test_artifacts" / "curated_behavior_dataset"
        source = root / "source.jsonl"
        output = root / "messages_only.jsonl"
        build_curated_behavior_dataset(source, limit=5, seed=123)

        stats = build_ms_swift_messages_only_dataset(source, output)

        self.assertEqual(stats["written"], 5)
        record = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(set(record), {"messages"})
        self.assertEqual(record["messages"][0]["role"], "user")


if __name__ == "__main__":
    unittest.main()
