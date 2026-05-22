from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.longitudinal_profile import build_longitudinal_state_profile


class LongitudinalProfileTests(unittest.TestCase):
    def test_sustained_high_entropy_recommends_manual_followup(self) -> None:
        profile = build_longitudinal_state_profile(
            session_id="student-a",
            records=[
                {
                    "risk_level": "medium",
                    "primary_state": "academic_sleep_stress",
                    "state_profile": {"stress_domains": ["academic", "sleep"]},
                    "dynamic_action": "soften_and_stabilize",
                },
                {
                    "risk_level": "medium",
                    "primary_state": "academic_sleep_stress",
                    "state_profile": {"stress_domains": ["academic"]},
                    "dynamic_action": "soften_and_stabilize",
                },
                {
                    "risk_level": "medium",
                    "primary_state": "academic_sleep_stress",
                    "state_profile": {"stress_domains": ["sleep"]},
                    "dynamic_action": "human_followup_watch",
                },
            ],
            entropy_trace=[
                {"score": 66},
                {"score": 70},
                {"score": 73},
            ],
            referral_events=[],
            feedback_summary={"total_feedback": 0},
        )

        self.assertEqual(profile.entropy_course, "sustained_high")
        self.assertEqual(profile.recommended_care_level, "manual_followup")
        self.assertIn("academic_sleep_stress", profile.dominant_states)
        self.assertIn("academic", profile.dominant_stress_domains)
        self.assertIn("recommend_human_followup", profile.priority_actions)

    def test_negative_feedback_without_high_entropy_recommends_strategy_repair(self) -> None:
        profile = build_longitudinal_state_profile(
            session_id="student-b",
            records=[
                {"risk_level": "low", "primary_state": "privacy_boundary", "state_profile": {}},
                {"risk_level": "low", "primary_state": "privacy_boundary", "state_profile": {}},
            ],
            entropy_trace=[{"score": 20}, {"score": 23}],
            referral_events=[],
            feedback_summary={"negative_count": 2, "positive_count": 0, "total_feedback": 2},
        )

        self.assertEqual(profile.recommended_care_level, "strategy_repair")
        self.assertIn("repair_response_style", profile.priority_actions)
        self.assertEqual(profile.next_review_hours, 24)

    def test_critical_risk_overrides_falling_entropy(self) -> None:
        profile = build_longitudinal_state_profile(
            session_id="student-c",
            records=[
                {"risk_level": "critical", "primary_state": "safety_risk", "state_profile": {}},
                {"risk_level": "medium", "primary_state": "sadness_distress", "state_profile": {}},
            ],
            entropy_trace=[{"score": 80}, {"score": 55}],
            referral_events=[],
            feedback_summary={},
        )

        self.assertEqual(profile.risk_course, "critical_seen")
        self.assertEqual(profile.recommended_care_level, "urgent")
        self.assertEqual(profile.next_review_hours, 1)


if __name__ == "__main__":
    unittest.main()
