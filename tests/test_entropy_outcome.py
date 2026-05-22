from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.entropy_outcome import build_entropy_reduction_outcome


class EntropyOutcomeTests(unittest.TestCase):
    def test_falling_entropy_is_improving(self) -> None:
        outcome = build_entropy_reduction_outcome(
            session_id="student-outcome-a",
            records=[
                {"risk_level": "medium", "entropy_score": 62},
                {"risk_level": "medium", "entropy_score": 49},
            ],
            entropy_trace=[{"score": 62}, {"score": 49}],
            feedback_summary={"positive_count": 1, "negative_count": 0},
            care_pathway={"route": "continue_observation", "priority": "low"},
        )

        self.assertEqual(outcome.status, "improving")
        self.assertEqual(outcome.entropy_delta, -13)
        self.assertGreaterEqual(outcome.effectiveness_score, 70)
        self.assertEqual(outcome.next_action, "maintain_working_strategy")

    def test_rising_entropy_is_deteriorating(self) -> None:
        outcome = build_entropy_reduction_outcome(
            session_id="student-outcome-b",
            records=[
                {"risk_level": "low", "entropy_score": 40},
                {"risk_level": "medium", "entropy_score": 55},
            ],
            entropy_trace=[{"score": 40}, {"score": 55}],
            feedback_summary={},
            care_pathway={"route": "monitor_next_turn", "priority": "medium"},
        )

        self.assertEqual(outcome.status, "deteriorating")
        self.assertEqual(outcome.risk_shift, "up")
        self.assertLess(outcome.effectiveness_score, 50)

    def test_negative_feedback_requires_strategy_repair(self) -> None:
        outcome = build_entropy_reduction_outcome(
            session_id="student-outcome-c",
            records=[{"risk_level": "low", "entropy_score": 30}, {"risk_level": "low", "entropy_score": 31}],
            entropy_trace=[{"score": 30}, {"score": 31}],
            feedback_summary={"positive_count": 0, "negative_count": 2},
            care_pathway={"route": "repair_reply_style", "priority": "medium"},
        )

        self.assertEqual(outcome.status, "needs_strategy_repair")
        self.assertEqual(outcome.feedback_signal, "negative")
        self.assertEqual(outcome.next_action, "repair_next_reply_style")

    def test_urgent_pathway_overrides_entropy_change(self) -> None:
        outcome = build_entropy_reduction_outcome(
            session_id="student-outcome-d",
            records=[{"risk_level": "critical", "entropy_score": 80}, {"risk_level": "medium", "entropy_score": 60}],
            entropy_trace=[{"score": 80}, {"score": 60}],
            feedback_summary={},
            care_pathway={"route": "urgent_safety", "priority": "critical"},
        )

        self.assertEqual(outcome.status, "crisis_priority")
        self.assertEqual(outcome.next_action, "activate_safety_protocol")


if __name__ == "__main__":
    unittest.main()
