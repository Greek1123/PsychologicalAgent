from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.feedback_behavior_sft_builder import build_feedback_behavior_sft_dataset


def _work_dir() -> Path:
    path = ROOT / "tmp_test_artifacts" / f"feedback_behavior_sft_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class FeedbackBehaviorSftBuilderTests(unittest.TestCase):
    def test_builds_user_context_without_bad_assistant_history(self) -> None:
        work_dir = _work_dir()
        source = work_dir / "preferences.jsonl"
        output = work_dir / "behavior_sft.jsonl"
        _write_jsonl(
            source,
            [
                {
                    "prompt": [
                        {"role": "user", "content": "最近考试很多，我睡不好。"},
                        {"role": "assistant", "content": "当前减熵重点：降低认知熵。"},
                        {"role": "user", "content": "我不想细说，怕别人知道。"},
                    ],
                    "chosen": "可以，不细说也没关系。你担心的是隐私和安全感，我们先不碰具体细节；你只需要说说现在更想安静一会儿，还是想要一个很小的缓解办法。",
                    "rejected": "当前减熵重点：降低认知熵。",
                }
            ],
        )

        stats = build_feedback_behavior_sft_dataset(str(source), str(output))

        self.assertEqual(stats["written"], 1)
        record = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
        messages = record["messages"]
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertIn("最近考试很多", messages[0]["content"])
        self.assertIn("怕别人知道", messages[0]["content"])
        self.assertNotIn("当前减熵重点", messages[0]["content"])
        self.assertNotIn("当前减熵重点", messages[1]["content"])

    def test_skips_low_quality_chosen_reply(self) -> None:
        work_dir = _work_dir()
        source = work_dir / "preferences.jsonl"
        output = work_dir / "behavior_sft.jsonl"
        _write_jsonl(
            source,
            [
                {
                    "prompt": [{"role": "user", "content": "我压力很大"}],
                    "chosen": "当前减熵重点：降低认知熵。",
                    "rejected": "你好，感谢你前来咨询。",
                }
            ],
        )

        stats = build_feedback_behavior_sft_dataset(str(source), str(output))

        self.assertEqual(stats["written"], 0)
        self.assertEqual(stats["skipped_low_quality"], 1)

    def test_drops_noisy_previous_user_turns(self) -> None:
        work_dir = _work_dir()
        source = work_dir / "preferences.jsonl"
        output = work_dir / "behavior_sft.jsonl"
        _write_jsonl(
            source,
            [
                {
                    "prompt": [
                        {"role": "user", "content": "傻逼吗"},
                        {"role": "user", "content": "我最近压力很大"},
                        {"role": "user", "content": "我最近压力很大"},
                        {"role": "user", "content": "晚上总睡不好"},
                    ],
                    "chosen": "压力大到影响睡眠，确实会很消耗人。我们先不急着把所有事讲清楚，可以先从今晚怎么稍微好睡一点开始。",
                    "rejected": "你好，感谢你前来咨询。",
                }
            ],
        )

        stats = build_feedback_behavior_sft_dataset(str(source), str(output))

        self.assertEqual(stats["written"], 1)
        record = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
        self.assertNotIn("傻逼", record["messages"][0]["content"])
        self.assertEqual(record["messages"][0]["content"].count("我最近压力很大"), 1)


if __name__ == "__main__":
    unittest.main()
