from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.care_pathway import build_care_pathway_decision
from campus_support_agent.longitudinal_profile import build_longitudinal_state_profile


def _profile(*, care_level: str = "observe", entropy_score: int = 35):
    profile = build_longitudinal_state_profile(
        session_id="student-path",
        records=[{"risk_level": "low", "primary_state": "academic_stress", "state_profile": {}}],
        entropy_trace=[{"score": entropy_score}],
        referral_events=[],
        feedback_summary={},
    )
    profile.recommended_care_level = care_level
    profile.next_review_hours = {
        "urgent": 1,
        "manual_followup": 12,
        "watch_closely": 24,
        "strategy_repair": 24,
        "observe": 72,
    }[care_level]
    return profile


class CarePathwayTests(unittest.TestCase):
    def test_urgent_profile_pauses_ai_only_reply(self) -> None:
        decision = build_care_pathway_decision(
            session_id="student-path",
            longitudinal_profile=_profile(care_level="urgent", entropy_score=82),
            latest_referral_decision={"urgency": "urgent", "should_refer": True},
        )

        self.assertEqual(decision.route, "urgent_safety")
        self.assertEqual(decision.priority, "critical")
        self.assertTrue(decision.should_notify_human)
        self.assertTrue(decision.should_pause_ai_only_reply)
        self.assertIn("activate_crisis_protocol", decision.backend_actions)

    def test_negative_feedback_routes_to_style_repair(self) -> None:
        decision = build_care_pathway_decision(
            session_id="student-path",
            longitudinal_profile=_profile(care_level="strategy_repair"),
            latest_feedback_adaptation={"mode": "repair_next_turn"},
            feedback_summary={"negative_count": 3},
        )

        self.assertEqual(decision.route, "repair_reply_style")
        self.assertEqual(decision.user_visible_mode, "acknowledge_and_repair")
        self.assertFalse(decision.should_notify_human)
        self.assertIn("reduce_question_pressure", decision.backend_actions)

    def test_watch_level_routes_to_next_turn_monitoring(self) -> None:
        decision = build_care_pathway_decision(
            session_id="student-path",
            longitudinal_profile=_profile(care_level="watch_closely", entropy_score=66),
            latest_dynamic_adjustment={"action": "soften_and_stabilize"},
        )

        self.assertEqual(decision.route, "monitor_next_turn")
        self.assertEqual(decision.priority, "medium")
        self.assertIn("compare_entropy_next_turn", decision.backend_actions)

    def test_observe_level_keeps_normal_support(self) -> None:
        decision = build_care_pathway_decision(
            session_id="student-path",
            longitudinal_profile=_profile(care_level="observe"),
        )

        self.assertEqual(decision.route, "continue_observation")
        self.assertEqual(decision.priority, "low")
        self.assertFalse(decision.should_notify_human)


if __name__ == "__main__":
    unittest.main()
