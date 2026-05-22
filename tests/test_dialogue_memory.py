from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.dialogue_memory import build_conversation_memory, build_memory_system_message


class DialogueMemoryTests(unittest.TestCase):
    def test_builds_topic_and_boundary_memory(self) -> None:
        history = [
            {"role": "user", "content": "我最近考试压力好大，晚上睡不着。"},
            {"role": "assistant", "content": "我们先把今晚的目标放小一点。"},
            {"role": "user", "content": "但我不太想细说，我怕别人知道。"},
        ]

        memory = build_conversation_memory(history, current_text="我现在只想停一下。")

        self.assertIn("考试/学业压力", memory.active_topics)
        self.assertIn("睡眠与身体状态", memory.active_topics)
        self.assertTrue(any("隐私" in item or "边界" in item for item in memory.user_boundaries))
        self.assertIn("隐私", memory.continuity_focus)

    def test_memory_system_message_requires_continuity(self) -> None:
        prompt = build_memory_system_message(
            [
                {"role": "user", "content": "你平时会喝咖啡吗？"},
                {"role": "assistant", "content": "可以，我们先轻松聊几句。"},
            ],
            current_text="我最近靠咖啡硬撑，晚上更睡不着。",
        )

        self.assertIn("最近用户表达", prompt)
        self.assertIn("睡眠与身体状态", prompt)
        self.assertIn("不要像第一次聊天一样重新开场", prompt)


if __name__ == "__main__":
    unittest.main()
