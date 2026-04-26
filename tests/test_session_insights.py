from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.session_insights import build_session_insight


class SessionInsightTests(unittest.TestCase):
    def test_empty_session_returns_observe_card(self) -> None:
        insight = build_session_insight(
            session_id="empty",
            records=[],
            entropy_trace=[],
            referral_events=[],
        )

        self.assertEqual(insight["risk_route"], "observe")
        self.assertEqual(insight["evidence"]["response_count"], 0)
        self.assertGreaterEqual(len(insight["recommended_next_steps"]), 1)

    def test_referral_and_sleep_policy_build_manual_followup_card(self) -> None:
        insight = build_session_insight(
            session_id="student-a",
            records=[
                {
                    "risk_level": "medium",
                    "entropy_score": 42,
                    "balance_state": "strained",
                    "local_policy_name": "exam_anxiety",
                },
                {
                    "risk_level": "medium",
                    "entropy_score": 68,
                    "balance_state": "fragile",
                    "local_policy_name": "sleep_appetite_drift",
                },
            ],
            entropy_trace=[
                {"score": 42, "level": 2, "balance_state": "strained"},
                {"score": 68, "level": 4, "balance_state": "fragile"},
            ],
            referral_events=[
                {
                    "urgency": "recommended",
                    "manual_referral_recommended": True,
                    "reasons": ["policy_escalation_watch"],
                }
            ],
        )

        self.assertEqual(insight["risk_route"], "manual_followup")
        self.assertEqual(insight["entropy_trend"], "rising")
        self.assertIn("睡眠饮食节律受影响", insight["state_summary"])
        self.assertTrue(any("睡眠" in item or "饮食" in item for item in insight["watch_items"]))
        self.assertTrue(any("人工" in item for item in insight["recommended_next_steps"]))


if __name__ == "__main__":
    unittest.main()
