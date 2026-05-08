from __future__ import annotations

import json
import unittest
from pathlib import Path

from campus_support_agent.integrated_behavior_sft_builder import (
    SourceSpec,
    build_integrated_behavior_sft_dataset,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class IntegratedBehaviorSftBuilderTest(unittest.TestCase):
    def test_builds_deduped_chinese_dataset(self) -> None:
        root = Path("tmp_test_artifacts") / "integrated_behavior_sft_builder"
        root.mkdir(parents=True, exist_ok=True)
        source_a = root / "a.jsonl"
        source_b = root / "b.jsonl"
        out = root / "out.jsonl"
        good = {
            "messages": [
                {"role": "user", "content": "我最近压力很大，晚上睡不着。"},
                {"role": "assistant", "content": "听起来你这几天真的绷得很紧，我们先把今晚要做的事放小一点。"},
            ]
        }
        _write_jsonl(
            source_a,
            [
                good,
                {"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]},
            ],
        )
        _write_jsonl(source_b, [good])

        result = build_integrated_behavior_sft_dataset(
            out=str(out),
            specs=[
                SourceSpec("a", source_a, 0),
                SourceSpec("b", source_b, 0),
            ],
            min_cjk_ratio=0.45,
            max_chars=1000,
        )

        self.assertEqual(result["written"], 1)
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows[0]["messages"][0]["content"], "我最近压力很大，晚上睡不着。")


if __name__ == "__main__":
    unittest.main()
