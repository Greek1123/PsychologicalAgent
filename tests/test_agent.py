from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.agent import CampusSupportAgent
from campus_support_agent.config import Settings, load_env_file
from campus_support_agent.memory import InMemorySessionStore
from campus_support_agent.providers import MockLLMProvider, MockSTTProvider
from campus_support_agent.retrieval import CampusKnowledgeRetriever
from campus_support_agent.schemas import RiskLevel


class CampusSupportAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        settings = Settings()
        self.agent = CampusSupportAgent(
            settings=settings,
            llm_provider=MockLLMProvider(),
            stt_provider=MockSTTProvider(),
            retriever=CampusKnowledgeRetriever(settings),
        )

    def test_low_risk_text_generates_support_plan_and_entropy(self) -> None:
        response = self.agent.handle_text(
            text="最近考试很多，我晚上睡不好，担心自己会挂科。",
            student_context={"grade": "大二"},
        )
        self.assertEqual(response.risk.level, RiskLevel.MEDIUM)
        self.assertGreaterEqual(len(response.plan.immediate_support), 1)
        self.assertIn("焦虑", response.assessment.primary_emotions)
        self.assertGreaterEqual(len(response.campus_resources), 1)
        self.assertGreater(response.entropy.score, 0)
        self.assertIn(response.entropy.balance_state, {"stable", "strained", "fragile", "crisis"})
        self.assertGreaterEqual(len(response.entropy_reduction.targeted_drivers), 1)
        self.assertLess(response.entropy_reduction.expected_delta_score, 0)

    def test_critical_text_routes_to_crisis_response(self) -> None:
        response = self.agent.handle_text(text="我真的不想活了，想自杀。")
        self.assertEqual(response.risk.level, RiskLevel.CRITICAL)
        self.assertEqual(response.entropy.balance_state, "crisis")
        self.assertEqual(response.entropy.level, 5)
        self.assertIsNotNone(response.safety.emergency_notice)
        self.assertTrue(any(item.category == "emergency" for item in response.campus_resources))
        self.assertIn("风险压力", response.entropy_reduction.targeted_drivers)

    def test_audio_path_uses_transcript(self) -> None:
        response = self.agent.handle_audio(
            file_bytes=b"fake-audio",
            filename="sample.wav",
            content_type="audio/wav",
            student_context={"grade": "大一"},
        )
        self.assertEqual(response.source, "audio")
        self.assertIsNotNone(response.transcript)
        self.assertGreater(len(response.plan.follow_up), 0)
        self.assertGreater(response.entropy.score, 0)

    def test_session_store_keeps_recent_history_and_entropy(self) -> None:
        store = InMemorySessionStore(max_messages=4)
        store.append_exchange("session-a", user_text="u1", assistant_text="a1")
        store.append_exchange("session-a", user_text="u2", assistant_text="a2")
        store.append_exchange("session-a", user_text="u3", assistant_text="a3")
        store.append_entropy_snapshot(
            "session-a",
            response_id="r1",
            score=50,
            level=3,
            balance_state="strained",
            dominant_drivers=["认知负荷(考试)"],
        )
        store.append_entropy_snapshot(
            "session-a",
            response_id="r2",
            score=42,
            level=2,
            balance_state="strained",
            dominant_drivers=["生理失衡(失眠)"],
        )

        history = store.get_history("session-a")
        entropy_trace = store.get_entropy_trace("session-a")
        self.assertEqual(len(history), 4)
        self.assertEqual(history[0]["content"], "u2")
        self.assertEqual(history[-1]["content"], "a3")
        self.assertEqual(len(entropy_trace), 2)
        self.assertEqual(store.get_last_entropy("session-a")["score"], 42)

    def test_load_env_file_does_not_override_existing_env_by_default(self) -> None:
        original = os.environ.get("CAMPUS_NAME")
        os.environ["CAMPUS_NAME"] = "外部环境学校"

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                env_path = Path(tmpdir) / ".env"
                env_path.write_text("CAMPUS_NAME=文件中的学校\nLLM_PROVIDER=openai_compatible\n", encoding="utf-8")
                load_env_file(env_path, override=False)

                self.assertEqual(os.environ["CAMPUS_NAME"], "外部环境学校")
                self.assertEqual(os.environ["LLM_PROVIDER"], "openai_compatible")
        finally:
            if original is None:
                os.environ.pop("CAMPUS_NAME", None)
            else:
                os.environ["CAMPUS_NAME"] = original
            os.environ.pop("LLM_PROVIDER", None)


if __name__ == "__main__":
    unittest.main()
