from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.dialogue_state import DialogueStage, classify_dialogue_state


class DialogueStateTests(unittest.TestCase):
    def test_privacy_boundary_has_priority(self) -> None:
        state = classify_dialogue_state("我不想说，我怕别人知道")

        self.assertEqual(state.stage, DialogueStage.PRIVACY_BOUNDARY)
        self.assertTrue(state.should_preserve_privacy)
        self.assertTrue(state.should_avoid_questions)

    def test_weak_input_after_boundary_avoids_advice(self) -> None:
        state = classify_dialogue_state(
            "？",
            conversation_history=[{"role": "user", "content": "我不想说"}],
        )

        self.assertEqual(state.stage, DialogueStage.WEAK_INPUT)
        self.assertTrue(state.should_avoid_questions)

    def test_dorm_distress_is_detected(self) -> None:
        state = classify_dialogue_state("我一回宿舍就很烦")

        self.assertEqual(state.stage, DialogueStage.DORM_DISTRESS)

    def test_sleep_pressure_is_detected(self) -> None:
        state = classify_dialogue_state("我最近压力好大，晚上睡不好")

        self.assertEqual(state.stage, DialogueStage.SLEEP_PRESSURE)


if __name__ == "__main__":
    unittest.main()
