from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ["DATABASE_PATH"] = str(ROOT / "test_main_runtime.db")

from campus_support_agent import main


class MainFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_env = {
            key: os.environ.get(key)
            for key in ("LLM_PROVIDER", "LOCAL_CHECKPOINT_PATH", "LOCAL_BASE_MODEL_PATH")
        }
        os.environ["LLM_PROVIDER"] = "mock"
        main.get_settings.cache_clear()
        main.get_agent.cache_clear()
        main.get_session_store.cache_clear()

    def tearDown(self) -> None:
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
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
        referrals = main.get_session_referrals(session_id)
        self.assertGreaterEqual(referrals["total_events"], 1)
        self.assertEqual(referrals["referral_events"][-1]["response_id"], response["response_id"])

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
        self.assertIn("referral_events", analysis)
        self.assertIn("session_insight", analysis)
        self.assertIn("risk_route", analysis["session_insight"])
        self.assertIn("risk_levels", overview)
        self.assertIn("manual_referral_count", overview)
        self.assertIn("risk_routes", overview)
        self.assertGreaterEqual(overview["total_records"], 1)

    def test_model_status_reports_local_checkpoint_configuration(self) -> None:
        os.environ["LLM_PROVIDER"] = "local_checkpoint"
        os.environ["LOCAL_CHECKPOINT_PATH"] = str(ROOT / "missing-checkpoint")
        os.environ["LOCAL_BASE_MODEL_PATH"] = str(ROOT / "missing-base-model")
        main.get_settings.cache_clear()

        status = main.get_model_status()

        self.assertEqual(status["llm_provider"], "local_checkpoint")
        self.assertTrue(status["local_checkpoint"]["enabled"])
        self.assertFalse(status["local_checkpoint"]["checkpoint_exists"])
        self.assertFalse(status["local_checkpoint"]["base_model_exists"])

    def test_session_feedback_updates_analysis_and_overview(self) -> None:
        session_id = f"test-feedback-session-{uuid4().hex}"
        response = main.support_text(
            {
                "session_id": session_id,
                "text": "我最近压力很大，晚上总是睡不好。",
                "student_context": {},
                "conversation_history": [],
            }
        )

        result = main.submit_session_feedback(
            session_id,
            {
                "response_id": response["response_id"],
                "helpful_score": 2,
                "mood_after": 72,
                "user_note": "感觉比刚才稳一点",
                "tags": ["helpful", "clear"],
            },
        )
        feedback = main.get_session_feedback(session_id)
        analysis = main.get_session_analysis(session_id)
        overview = main.get_overview_analytics(limit=50)

        self.assertEqual(result["feedback"]["response_id"], response["response_id"])
        self.assertEqual(result["feedback_summary"]["positive_count"], 1)
        self.assertEqual(feedback["total_feedback"], 1)
        self.assertEqual(analysis["feedback_summary"]["total_feedback"], 1)
        self.assertEqual(analysis["intervention_feedback"][0]["mood_after"], 72)
        self.assertGreaterEqual(overview["feedback_summary"]["total_feedback"], 1)

    def test_session_feedback_rejects_invalid_payload(self) -> None:
        with self.assertRaises(HTTPException):
            main.submit_session_feedback(
                "test-invalid-feedback",
                {
                    "response_id": "resp-invalid",
                    "helpful_score": 5,
                },
            )


if __name__ == "__main__":
    unittest.main()
