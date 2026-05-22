from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.feedback_adaptation import build_feedback_adaptation
from campus_support_agent.schemas import SupportPlan
from campus_support_agent.strategy_execution import apply_feedback_adaptation_to_plan


class FeedbackAdaptationTests(unittest.TestCase):
    def test_negative_pressure_feedback_reduces_next_question_pressure(self) -> None:
        adaptation = build_feedback_adaptation(
            feedback_summary={
                "negative_count": 2,
                "positive_count": 0,
                "average_helpful_score": -1.5,
                "common_tags": {"too_many_questions": 1, "missed_context": 1},
            },
            recent_feedback=[{"helpful_score": -2, "tags": ["too_many_questions"]}],
        )

        self.assertEqual(adaptation.mode, "repair_next_turn")
        self.assertEqual(adaptation.question_pressure, "low")
        self.assertTrue(adaptation.should_collect_bad_case)
        self.assertIn("reflect_user_context_first", adaptation.preferred_moves)

    def test_positive_feedback_keeps_working_pattern(self) -> None:
        adaptation = build_feedback_adaptation(
            feedback_summary={
                "negative_count": 0,
                "positive_count": 3,
                "average_helpful_score": 1.7,
                "common_tags": {},
            }
        )

        self.assertEqual(adaptation.mode, "keep_working_pattern")
        self.assertFalse(adaptation.should_collect_bad_case)

    def test_repair_mode_rewrites_next_plan(self) -> None:
        adaptation = build_feedback_adaptation(
            feedback_summary={
                "negative_count": 1,
                "positive_count": 0,
                "average_helpful_score": -1,
                "common_tags": {"too_short": 1, "repetitive": 1},
            }
        )
        plan = SupportPlan(
            summary="generic",
            immediate_support=["old"],
            campus_actions=[],
            self_regulation=[],
            follow_up=["old follow"],
        )

        updated = apply_feedback_adaptation_to_plan(plan, feedback_adaptation=adaptation)

        self.assertNotEqual(updated.summary, "generic")
        self.assertNotEqual(updated.immediate_support[0], "old")
        self.assertTrue(updated.self_regulation)
        self.assertNotEqual(updated.follow_up[0], "old follow")


if __name__ == "__main__":
    unittest.main()
