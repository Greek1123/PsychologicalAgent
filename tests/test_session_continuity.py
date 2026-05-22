from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.session_continuity import (
    build_session_continuity_summary,
    enrich_student_context_with_continuity,
)


class SessionContinuityTests(unittest.TestCase):
    def test_detects_privacy_boundary_stage(self) -> None:
        summary = build_session_continuity_summary(
            session_id="student-a",
            records=[
                {
                    "primary_state": "privacy_boundary",
                    "risk_level": "medium",
                    "entropy_score": 46,
                    "state_profile": {
                        "stress_domains": ["social"],
                        "boundary_flags": ["privacy_concern"],
                    },
                    "reply_text": "你可以不用细说。",
                }
            ],
            conversation_history=[
                {"role": "user", "content": "我不想说，我怕别人知道。"},
                {"role": "assistant", "content": "你可以不用细说。"},
            ],
        )

        self.assertEqual(summary["dialogue_stage"], "boundary_building")
        self.assertIn("privacy_concern", summary["user_boundaries"])
        self.assertIn("不要追问隐私细节", summary["avoid_next_turn"])
        self.assertIn("先明确不会逼用户细说", summary["recommended_next_moves"])

    def test_enriches_student_context_without_overwriting_existing_fields(self) -> None:
        summary = build_session_continuity_summary(
            session_id="student-b",
            records=[
                {
                    "primary_state": "academic_pressure",
                    "risk_level": "medium",
                    "entropy_score": 42,
                    "state_profile": {"stress_domains": ["academic"]},
                }
            ],
        )

        context = enrich_student_context_with_continuity({"grade": "大一"}, summary)

        self.assertEqual(context["grade"], "大一")
        self.assertEqual(context["session_continuity"]["dialogue_stage"], "initial_contact")
        self.assertIn("不要重新自我介绍", context["session_continuity"]["avoid_next_turn"])


if __name__ == "__main__":
    unittest.main()
