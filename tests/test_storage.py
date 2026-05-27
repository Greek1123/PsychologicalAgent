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

    def test_session_analysis_tracks_entropy_orchestration_timeline(self) -> None:
        db_path = _test_db_path()

        store = SQLiteSessionStore(str(db_path), max_messages=6)
        store.store_support_response(
            session_id="session-orchestration",
            response_id="resp-o1",
            source="text",
            input_text="我不想说，我怕别人知道。",
            transcript=None,
            student_context={},
            conversation_history=[],
            response_payload={
                "reply_text": "你可以不用细说，我会先尊重你的边界。",
                "risk": {"level": "medium", "score": 46},
                "entropy": {"score": 58, "level": 3, "trend": {"delta": 6}},
                "state_profile": {
                    "primary_state": "privacy_boundary",
                    "stress_domains": ["social"],
                    "boundary_flags": ["privacy_concern"],
                },
                "dynamic_adjustment": {"action": "soften_and_stabilize"},
                "feedback_adaptation": {"mode": "reduce_question_pressure"},
                "entropy_orchestration": {
                    "route": "boundary_respecting_support",
                    "next_focus": "先建立安全感，再允许用户少量表达。",
                    "user_visible_goal": "让用户确认这里可以不急着解释。",
                    "constraints": ["尊重用户不想细说", "不要展示心理熵术语"],
                    "risk_control": "watch",
                },
            },
        )

        analysis = store.get_session_analysis("session-orchestration")
        overview = store.get_overview_stats()

        self.assertEqual(analysis["latest_entropy_orchestration"]["route"], "boundary_respecting_support")
        self.assertEqual(analysis["orchestration_routes"]["boundary_respecting_support"], 1)
        self.assertEqual(analysis["orchestration_timeline"][0]["route"], "boundary_respecting_support")
        self.assertEqual(analysis["session_continuity"]["dialogue_stage"], "boundary_building")
        self.assertEqual(
            analysis["next_orchestration_recommendation"]["recommended_action"],
            "respect_boundary_and_offer_low_pressure_support",
        )
        self.assertEqual(overview["orchestration_routes"]["boundary_respecting_support"], 1)
        self.assertEqual(overview["current_orchestration_routes"]["boundary_respecting_support"], 1)
        self.assertIn("current_dialogue_stages", overview)

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
        self.assertIn("session_continuity", analysis)
        self.assertIn("longitudinal_profile", analysis)
        self.assertIn("care_pathway", analysis)
        self.assertIn("entropy_reduction_outcome", analysis)
        self.assertIn("orchestration_timeline", analysis)
        self.assertIn("next_orchestration_recommendation", analysis)
        self.assertEqual(analysis["session_insight"]["evidence"]["referral_event_count"], 1)
        self.assertEqual(analysis["session_continuity"]["dialogue_stage"], "initial_contact")
        self.assertEqual(analysis["longitudinal_profile"]["recommended_care_level"], "manual_followup")
        self.assertEqual(analysis["care_pathway"]["route"], "human_followup_recommended")
        self.assertEqual(analysis["entropy_reduction_outcome"]["status"], "needs_human_followup")

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
        self.assertEqual(analysis["longitudinal_profile"]["recommended_care_level"], "observe")
        self.assertEqual(analysis["care_pathway"]["route"], "continue_observation")
        self.assertEqual(analysis["entropy_reduction_outcome"]["status"], "deteriorating")
        self.assertEqual(len(analysis["intervention_feedback"]), 2)
        self.assertEqual(overview["feedback_summary"]["total_feedback"], 2)
        self.assertIn("care_pathway_routes", overview)
        self.assertIn("current_care_pathway_routes", overview)
        self.assertIn("entropy_outcome_statuses", overview)
        self.assertIn("current_entropy_outcome_statuses", overview)
        self.assertIn("orchestration_routes", overview)
        self.assertIn("current_orchestration_routes", overview)
        self.assertIn("current_dialogue_stages", overview)
        self.assertGreaterEqual(overview["care_pathway_routes"].get("continue_observation", 0), 2)
        self.assertGreaterEqual(overview["current_care_pathway_routes"].get("continue_observation", 0), 1)
        self.assertGreaterEqual(overview["entropy_outcome_statuses"].get("stable_observe", 0), 1)
        bad_cases = store.list_feedback_cases(session_id="session-e")
        self.assertEqual(len(bad_cases), 1)
        self.assertEqual(bad_cases[0]["response_id"], "resp-e2")
        self.assertEqual(bad_cases[0]["feedback"]["helpful_score"], -1)

        store.clear("session-e")
        self.assertEqual(store.get_intervention_feedback("session-e"), [])

    def test_care_queue_prioritizes_latest_session_records(self) -> None:
        db_path = _test_db_path()

        store = SQLiteSessionStore(str(db_path), max_messages=6)
        store.store_support_response(
            session_id="session-low",
            response_id="resp-low",
            source="text",
            input_text="u-low",
            transcript=None,
            student_context={},
            conversation_history=[],
            response_payload={
                "reply_text": "a-low",
                "risk": {"level": "low", "score": 10},
                "entropy": {"score": 20, "trend": {"delta": 0}},
            },
        )
        store.store_support_response(
            session_id="session-high",
            response_id="resp-high-old",
            source="text",
            input_text="u-high-old",
            transcript=None,
            student_context={},
            conversation_history=[],
            response_payload={
                "reply_text": "a-high-old",
                "risk": {"level": "low", "score": 10},
                "entropy": {"score": 25, "trend": {"delta": 0}},
            },
        )
        store.store_support_response(
            session_id="session-high",
            response_id="resp-high",
            source="text",
            input_text="u-high",
            transcript=None,
            student_context={},
            conversation_history=[],
            response_payload={
                "reply_text": "a-high",
                "risk": {"level": "high", "score": 80},
                "entropy": {"score": 72, "trend": {"delta": 12}},
                "referral_decision": {"should_refer": True, "urgency": "recommended"},
            },
        )

        queue = store.get_care_queue()
        queue_with_low = store.get_care_queue(include_low_priority=True)

        self.assertEqual(queue["total_items"], 1)
        self.assertEqual(queue["items"][0]["session_id"], "session-high")
        self.assertEqual(queue["items"][0]["priority"], "high")
        self.assertEqual(queue["items"][0]["recommended_action"], "recommend_human_followup")
        self.assertEqual(queue_with_low["total_items"], 2)

    def test_human_intervention_status_closes_care_queue_item(self) -> None:
        db_path = _test_db_path()

        store = SQLiteSessionStore(str(db_path), max_messages=6)
        store.store_support_response(
            session_id="session-human",
            response_id="resp-human",
            source="text",
            input_text="u-human",
            transcript=None,
            student_context={},
            conversation_history=[],
            response_payload={
                "reply_text": "a-human",
                "risk": {"level": "high", "score": 82},
                "entropy": {"score": 76, "trend": {"delta": 10}},
                "referral_decision": {"should_refer": True, "urgency": "recommended"},
            },
        )

        acknowledged = store.append_human_intervention(
            session_id="session-human",
            response_id="resp-human",
            status="acknowledged",
            handler_id="counselor-001",
            note="已查看，准备联系学生。",
            next_action="same_day_checkin",
            tags=["manual_followup"],
        )
        queue = store.get_care_queue()

        self.assertEqual(acknowledged["status"], "acknowledged")
        self.assertEqual(queue["total_items"], 1)
        self.assertEqual(queue["items"][0]["evidence"]["human_intervention"]["handler_id"], "counselor-001")

        store.append_human_intervention(
            session_id="session-human",
            response_id="resp-human",
            status="resolved",
            handler_id="counselor-001",
            note="已完成线下确认。",
            next_action="continue_observation",
            tags=["resolved"],
        )
        open_queue = store.get_care_queue()
        full_queue = store.get_care_queue(include_resolved=True)
        analysis = store.get_session_analysis("session-human")

        self.assertEqual(open_queue["total_items"], 0)
        self.assertEqual(full_queue["total_items"], 1)
        self.assertEqual(full_queue["items"][0]["evidence"]["human_intervention"]["status"], "resolved")
        self.assertEqual(analysis["latest_human_intervention"]["status"], "resolved")


if __name__ == "__main__":
    unittest.main()
