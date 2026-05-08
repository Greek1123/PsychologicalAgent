from __future__ import annotations

import sys
import unittest
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.storage import SQLiteSessionStore


def _test_db_path() -> Path:
    return ROOT / f"test_storage_{uuid4().hex}.db"


class SQLiteSessionStoreTests(unittest.TestCase):
    def test_store_persists_history_and_entropy_across_instances(self) -> None:
        db_path = _test_db_path()

        store = SQLiteSessionStore(str(db_path), max_messages=6)
        store.append_exchange("session-a", user_text="我很焦虑", assistant_text="先呼吸")
        store.append_entropy_snapshot(
            "session-a",
            response_id="r1",
            score=58,
            level=3,
            balance_state="strained",
            dominant_drivers=["认知负荷(考试)"],
        )

        reloaded = SQLiteSessionStore(str(db_path), max_messages=6)
        history = reloaded.get_history("session-a")
        trace = reloaded.get_entropy_trace("session-a")

        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[1]["content"], "先呼吸")
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0]["score"], 58)
        self.assertEqual(reloaded.get_last_entropy("session-a")["balance_state"], "strained")

    def test_clear_removes_persisted_session(self) -> None:
        db_path = _test_db_path()

        store = SQLiteSessionStore(str(db_path), max_messages=6)
        store.append_exchange("session-b", user_text="u1", assistant_text="a1")
        store.append_entropy_snapshot(
            "session-b",
            response_id="r1",
            score=40,
            level=2,
            balance_state="strained",
            dominant_drivers=["情绪强度(焦虑)"],
        )
        store.clear("session-b")

        self.assertEqual(store.get_history("session-b"), [])
        self.assertEqual(store.get_entropy_trace("session-b"), [])
        self.assertIsNone(store.get_last_entropy("session-b"))

    def test_store_support_response_can_be_loaded_for_export(self) -> None:
        db_path = _test_db_path()

        store = SQLiteSessionStore(str(db_path), max_messages=6)
        store.store_support_response(
            session_id="session-c",
            response_id="resp-c",
            source="text",
            input_text="最近很烦。",
            transcript=None,
            student_context={"grade": "大一"},
            conversation_history=[{"role": "assistant", "content": "之前我们讨论过睡眠。"}],
            response_payload={"risk": {"level": "medium"}, "plan": {"summary": "test"}},
        )

        rows = store.list_support_responses(session_id="session-c")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["response_id"], "resp-c")
        self.assertEqual(rows[0]["student_context"]["grade"], "大一")

    def test_referral_events_can_be_recorded_and_cleared(self) -> None:
        db_path = _test_db_path()

        store = SQLiteSessionStore(str(db_path), max_messages=6)
        store.append_referral_event(
            session_id="session-d",
            response_id="resp-d",
            urgency="recommended",
            reasons=["policy_escalation_watch"],
            policy_name="sleep_appetite_drift",
            risk_level="medium",
            entropy_score=55,
            manual_referral_recommended=True,
        )

        events = store.get_referral_events("session-d")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["urgency"], "recommended")
        self.assertTrue(events[0]["manual_referral_recommended"])
        analysis = store.get_session_analysis("session-d")
        self.assertIn("session_insight", analysis)
        self.assertEqual(analysis["session_insight"]["evidence"]["referral_event_count"], 1)

        store.clear("session-d")
        self.assertEqual(store.get_referral_events("session-d"), [])

    def test_intervention_feedback_is_summarized_and_cleared(self) -> None:
        db_path = _test_db_path()

        store = SQLiteSessionStore(str(db_path), max_messages=6)
        store.store_support_response(
            session_id="session-e",
            response_id="resp-e1",
            source="text",
            input_text="u1",
            transcript=None,
            student_context={},
            conversation_history=[],
            response_payload={"reply_text": "a1", "risk": {"level": "low"}},
        )
        store.store_support_response(
            session_id="session-e",
            response_id="resp-e2",
            source="text",
            input_text="u2",
            transcript=None,
            student_context={},
            conversation_history=[],
            response_payload={"reply_text": "a2", "risk": {"level": "medium"}},
        )
        first = store.append_intervention_feedback(
            session_id="session-e",
            response_id="resp-e1",
            helpful_score=2,
            mood_after=68,
            user_note="这次有帮助",
            tags=["helpful", "grounding"],
        )
        store.append_intervention_feedback(
            session_id="session-e",
            response_id="resp-e2",
            helpful_score=-1,
            mood_after=42,
            tags=["too_short"],
        )

        feedback = store.get_intervention_feedback("session-e")
        summary = store.summarize_intervention_feedback("session-e")
        analysis = store.get_session_analysis("session-e")
        overview = store.get_overview_stats()

        self.assertEqual(first["tags"], ["helpful", "grounding"])
        self.assertEqual(len(feedback), 2)
        self.assertEqual(summary["total_feedback"], 2)
        self.assertEqual(summary["average_helpful_score"], 0.5)
        self.assertEqual(summary["positive_count"], 1)
        self.assertEqual(summary["negative_count"], 1)
        self.assertEqual(summary["average_mood_after"], 55)
        self.assertEqual(summary["common_tags"]["helpful"], 1)
        self.assertEqual(analysis["feedback_summary"]["total_feedback"], 2)
        self.assertEqual(len(analysis["intervention_feedback"]), 2)
        self.assertEqual(overview["feedback_summary"]["total_feedback"], 2)
        bad_cases = store.list_feedback_cases(session_id="session-e")
        self.assertEqual(len(bad_cases), 1)
        self.assertEqual(bad_cases[0]["response_id"], "resp-e2")
        self.assertEqual(bad_cases[0]["feedback"]["helpful_score"], -1)

        store.clear("session-e")
        self.assertEqual(store.get_intervention_feedback("session-e"), [])


if __name__ == "__main__":
    unittest.main()
