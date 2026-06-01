from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException
from fastapi.testclient import TestClient

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
        processing_health = main.get_processing_health(limit=20)

        self.assertEqual(analysis["session_id"], session_id)
        self.assertGreaterEqual(analysis["total_responses"], 1)
        self.assertIn("local_policies", analysis)
        self.assertIn("referral_events", analysis)
        self.assertIn("session_insight", analysis)
        self.assertIn("risk_route", analysis["session_insight"])
        self.assertIn("session_continuity", analysis)
        self.assertIn("entropy_reduction_outcome", analysis)
        self.assertIn("strategy_layer_summary", analysis)
        self.assertIn("latest", analysis["strategy_layer_summary"])
        self.assertIn("strategy_ids", analysis["strategy_layer_summary"])
        self.assertIn("next_backend_focus", analysis["strategy_layer_summary"])
        self.assertIn("orchestration_timeline", analysis)
        self.assertIn("next_orchestration_recommendation", analysis)
        self.assertIn("risk_levels", overview)
        self.assertIn("manual_referral_count", overview)
        self.assertIn("risk_routes", overview)
        self.assertIn("care_pathway_routes", overview)
        self.assertIn("current_care_pathway_routes", overview)
        self.assertIn("entropy_outcome_statuses", overview)
        self.assertIn("current_entropy_outcome_statuses", overview)
        self.assertIn("orchestration_routes", overview)
        self.assertIn("current_orchestration_routes", overview)
        self.assertIn("current_dialogue_stages", overview)
        self.assertIn("processing_consistency_summary", overview)
        self.assertIn("current_processing_consistency_summary", overview)
        self.assertIn("processing_consistency_bad_cases", overview)
        self.assertIn(processing_health["status"], {"ok", "watch", "blocked"})
        self.assertGreaterEqual(processing_health["records_seen"], 1)
        self.assertIn("processing_consistency", processing_health)
        self.assertIn("source_endpoints", processing_health)
        self.assertIn("strategy_layer_summary", overview)
        self.assertIn("current_sessions", overview["strategy_layer_summary"])
        self.assertIn("all_records", overview["strategy_layer_summary"])
        self.assertGreaterEqual(overview["total_records"], 1)
        care_queue = main.get_care_queue(limit=10, include_low_priority=True)
        self.assertIn("items", care_queue)
        self.assertIn("priority_counts", care_queue)

        strategy_layer = main.get_session_strategy_layer(session_id)
        strategy_overview = main.get_strategy_layer_overview(limit=20)
        self.assertEqual(strategy_layer["session_id"], session_id)
        self.assertIn("strategy_layer_summary", strategy_layer)
        self.assertIn("latest_state_profile", strategy_layer)
        self.assertIn("next_orchestration_recommendation", strategy_layer)
        self.assertIn("strategy_layer_summary", strategy_overview)
        self.assertIn("current_sessions", strategy_overview["strategy_layer_summary"])
        self.assertIn("recent_strategy_timeline", strategy_overview["strategy_layer_summary"])

        decision_trace = main.get_session_decision_trace(session_id)
        decision_overview = main.get_decision_trace_overview(limit=20)
        self.assertEqual(decision_trace["session_id"], session_id)
        self.assertGreaterEqual(decision_trace["returned_turns"], 1)
        self.assertIn("latest_decision", decision_trace)
        self.assertIn("summary", decision_trace)
        self.assertIn("risk", decision_trace["latest_decision"])
        self.assertIn("entropy", decision_trace["latest_decision"])
        self.assertIn("state", decision_trace["latest_decision"])
        self.assertIn("strategy", decision_trace["latest_decision"])
        self.assertIn("care", decision_trace["latest_decision"])
        self.assertIn("latest_session_decisions", decision_overview)
        self.assertIn("summary", decision_overview)

    def test_session_processing_timeline_escalates_dangerous_place_followup(self) -> None:
        session_id = f"test-processing-session-{uuid4().hex}"
        first = main.support_text(
            {
                "session_id": session_id,
                "text": "\u660e\u5929\u65e9\u4e0a\u8003\u8bd5\uff0c\u6211\u73b0\u5728\u5b8c\u5168\u7761\u4e0d\u7740\uff0c\u8d8a\u60f3\u8d8a\u614c\u3002",
                "student_context": {},
                "conversation_history": [],
            }
        )
        second = main.support_text(
            {
                "session_id": session_id,
                "text": "\u6211\u597d\u96be\u53d7\uff0c\u6211\u60f3\u53bb\u5929\u53f0\u51b7\u9759\u4e00\u4e0b",
                "student_context": {},
                "conversation_history": [],
            }
        )

        analysis = main.get_session_analysis(session_id)

        self.assertEqual(first["processing_summary"]["route"], "local_policy")
        self.assertEqual(second["risk"]["level"], "critical")
        self.assertEqual(second["processing_summary"]["route"], "crisis_safety")
        self.assertEqual(second["processing_summary"]["safety_priority"], "urgent")
        self.assertEqual(analysis["latest_processing_summary"]["route"], "crisis_safety")
        self.assertEqual(analysis["processing_timeline"][-1]["route"], "crisis_safety")
        self.assertEqual(analysis["processing_summary"]["latest_next_backend_action"], "activate_urgent_handoff")
        self.assertTrue(analysis["processing_summary"]["needs_human_attention"])
        self.assertEqual(analysis["processing_consistency"]["summary"]["status"], "ok")

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

    def test_ops_readiness_reports_deployment_checks(self) -> None:
        readiness = main.get_ops_readiness()

        self.assertIn(readiness["status"], {"ready", "degraded", "blocked"})
        self.assertIn("summary", readiness)
        self.assertIn("checks", readiness)
        self.assertTrue(any(check["name"] == "llm_provider" for check in readiness["checks"]))

    def test_processing_health_blocks_on_consistency_mismatch(self) -> None:
        health = main._build_processing_health_report(
            readiness={"status": "ready", "summary": "ok"},
            overview={
                "total_records": 1,
                "total_sessions": 1,
                "processing_consistency_summary": {"inconsistent_turns": 1, "status": "needs_review"},
                "current_processing_consistency_summary": {"inconsistent_turns": 0, "status": "ok"},
                "processing_consistency_bad_cases": [{"response_id": "bad"}],
            },
            reply_quality={"summary": {"needs_review": 0}},
            decision_trace={"summary": {"needs_attention": False}},
            limit=20,
        )

        self.assertEqual(health["status"], "blocked")
        self.assertIn("processing_consistency_mismatch", health["blocking_issues"])
        self.assertEqual(health["processing_consistency"]["bad_case_count"], 1)

    def test_frontend_contract_exposes_handoff_fields(self) -> None:
        contract = main.get_frontend_contract()

        self.assertEqual(contract["endpoints"]["text_support"]["path"], "/api/v1/support/text")
        self.assertEqual(contract["endpoints"]["audio_support"]["content_type"], "multipart/form-data")
        self.assertIn("cors", contract)
        self.assertIn("FRONTEND_ALLOWED_ORIGINS", contract["cors"]["env"])
        self.assertIn("http://127.0.0.1:5173", contract["cors"]["allowed_origins"])
        self.assertIn("text", contract["text_request_example"])
        self.assertIn("reply_text", contract["response_core_fields"])
        self.assertIn("risk", contract["response_core_fields"])
        self.assertIn("entropy", contract["response_core_fields"])
        self.assertIn("referral_decision", contract["response_core_fields"])
        self.assertIn("human_interventions", contract["response_core_fields"])
        self.assertIn("processing_summary", contract["response_core_fields"])
        self.assertIn("processing_consistency", contract["endpoints"]["session_analysis"]["processing_fields"])
        self.assertIn("human_interventions", contract["endpoints"])
        self.assertIn("role_view", contract["endpoints"])
        self.assertIn("ops_readiness", contract["endpoints"])
        self.assertIn("processing_health", contract["endpoints"])
        self.assertIn("backend_role_views", contract["frontend_display_policy"])
        self.assertIn("student_chat", contract["frontend_display_policy"])
        self.assertIn("research_dashboard", contract["frontend_display_policy"])
        self.assertIn("processing_summary", contract["frontend_display_policy"]["research_dashboard"])
        self.assertIn("critical", contract["risk_badges"])
        self.assertGreaterEqual(len(contract["demo_prompts"]), 4)

    def test_cors_preflight_allows_local_frontend_origin(self) -> None:
        client = TestClient(main.app)

        response = client.options(
            "/api/v1/support/text",
            headers={
                "Origin": "http://127.0.0.1:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], "http://127.0.0.1:5173")

    def test_session_role_view_projects_privacy_fields(self) -> None:
        session_id = f"test-view-session-{uuid4().hex}"
        main.support_text(
            {
                "session_id": session_id,
                "text": "我最近压力很大，晚上一直睡不好。",
                "student_context": {"student_name": "private"},
                "conversation_history": [],
            }
        )

        student_view = main.get_session_role_view(session_id, role="student")
        research_view = main.get_session_role_view(session_id, role="research")
        admin_view = main.get_session_role_view(session_id, role="admin")

        self.assertEqual(student_view["role"], "student")
        self.assertIn("reply_text", student_view["latest_response"])
        self.assertNotIn("system_flags", student_view["latest_response"])
        self.assertNotIn("processing_summary", student_view["latest_response"])
        self.assertNotIn("intervention_strategy", student_view["latest_response"])
        self.assertIn("text_redacted", research_view["latest_response"])
        self.assertNotIn("reply_text", research_view["latest_response"])
        self.assertNotIn("latest_reply_text", research_view["analysis"])
        self.assertIn("processing_summary", research_view["latest_response"])
        self.assertIn("system_flags", admin_view["latest_response"])
        self.assertIn("processing_summary", admin_view["latest_response"])

    def test_human_intervention_endpoint_updates_care_queue_state(self) -> None:
        session_id = f"test-human-session-{uuid4().hex}"
        response = main.support_text(
            {
                "session_id": session_id,
                "text": "我这几天一直睡不着，吃不下，也不太想见人。",
                "student_context": {},
                "conversation_history": [],
            }
        )

        created = main.append_session_human_intervention(
            session_id,
            {
                "response_id": response["response_id"],
                "status": "acknowledged",
                "handler_id": "counselor-001",
                "note": "已查看，准备低压力跟进。",
                "next_action": "same_day_checkin",
                "tags": ["manual_followup"],
            },
        )
        interventions = main.get_session_human_interventions(session_id)
        analysis = main.get_session_analysis(session_id)
        queue = main.get_care_queue(limit=20, include_low_priority=True)

        self.assertEqual(created["human_intervention"]["status"], "acknowledged")
        self.assertEqual(interventions["latest_human_intervention"]["handler_id"], "counselor-001")
        self.assertEqual(analysis["latest_human_intervention"]["status"], "acknowledged")
        self.assertTrue(
            any(
                item["session_id"] == session_id
                and item["evidence"]["human_intervention"]["status"] == "acknowledged"
                for item in queue["items"]
            )
        )

        main.append_session_human_intervention(
            session_id,
            {
                "response_id": response["response_id"],
                "status": "resolved",
                "handler_id": "counselor-001",
                "note": "已完成初步跟进。",
                "tags": ["resolved"],
            },
        )
        open_queue = main.get_care_queue(limit=20, include_low_priority=True)
        full_queue = main.get_care_queue(limit=20, include_low_priority=True, include_resolved=True)

        self.assertFalse(any(item["session_id"] == session_id for item in open_queue["items"]))
        self.assertTrue(any(item["session_id"] == session_id for item in full_queue["items"]))

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
        self.assertIn("entropy_reduction_outcome", analysis)
        self.assertGreaterEqual(overview["feedback_summary"]["total_feedback"], 1)
        self.assertIn("care_pathway_priorities", overview)

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
