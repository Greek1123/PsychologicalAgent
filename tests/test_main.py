from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ["DATABASE_PATH"] = str(ROOT / "test_main_runtime.db")

from campus_support_agent import main


class MainFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        main.get_settings.cache_clear()
        main.get_agent.cache_clear()
        main.get_session_store.cache_clear()

    def test_support_text_adds_system_flags_and_referral(self) -> None:
        session_id = "test-main-session"
        response = main.support_text(
            {
                "session_id": session_id,
                "text": "我这几天一直睡不好，也吃不下东西。",
                "student_context": {},
                "conversation_history": [],
            }
        )
        self.assertIn("system_flags", response)
        self.assertIn("referral_decision", response)
        self.assertTrue(response["referral_decision"]["should_refer"])
        self.assertIn(response["referral_decision"]["urgency"], {"watch", "recommended", "urgent"})

    def test_session_analysis_and_overview_are_available(self) -> None:
        session_id = "test-analysis-session"
        main.support_text(
            {
                "session_id": session_id,
                "text": "我今天本来还好，一回宿舍就烦。",
                "student_context": {},
                "conversation_history": [],
            }
        )
        analysis = main.get_session_analysis(session_id)
        overview = main.get_overview_analytics(limit=20)

        self.assertEqual(analysis["session_id"], session_id)
        self.assertGreaterEqual(analysis["total_responses"], 1)
        self.assertIn("local_policies", analysis)
        self.assertIn("risk_levels", overview)
        self.assertGreaterEqual(overview["total_records"], 1)


if __name__ == "__main__":
    unittest.main()
