from __future__ import annotations

import os
import sys
import unittest
import io
import math
import struct
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.agent import CampusSupportAgent
from campus_support_agent.config import Settings, load_env_file
from campus_support_agent.memory import InMemorySessionStore
from campus_support_agent.providers import LocalCheckpointLLMProvider, MockLLMProvider, MockSTTProvider, build_llm_provider
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
        self.assertIsNotNone(response.safety.emergency_notice)
        self.assertTrue(any(item.category == "emergency" for item in response.campus_resources))
        self.assertIn("风险压力", response.entropy_reduction.targeted_drivers)
        self.assertIsNotNone(response.referral_decision)
        self.assertTrue(response.referral_decision.should_refer)
        self.assertEqual(response.referral_decision.urgency, "urgent")

    def test_rooftop_cooling_off_routes_to_crisis_response(self) -> None:
        response = self.agent.handle_text(text="我好难受，我想去天台冷静一下")

        self.assertEqual(response.risk.level, RiskLevel.CRITICAL)
        self.assertIsNotNone(response.safety.emergency_notice)
        self.assertTrue(response.referral_decision.should_refer)
        self.assertEqual(response.referral_decision.urgency, "urgent")
        self.assertIn("天台", response.risk.trigger_terms)
        self.assertIn("不要一个人待着", response.reply_text)

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
        self.assertIsNotNone(response.multimodal_signal)
        self.assertFalse(response.multimodal_signal.analysis_available)
        self.assertIn("wav_parse_failed", response.multimodal_signal.analysis_notes)

    def test_audio_path_extracts_basic_wav_signal(self) -> None:
        sample_rate = 8000
        duration_seconds = 0.1
        frames = []
        for index in range(int(sample_rate * duration_seconds)):
            value = int(12000 * math.sin(2 * math.pi * 440 * index / sample_rate))
            frames.append(struct.pack("<h", value))
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(b"".join(frames))

        response = self.agent.handle_audio(
            file_bytes=buffer.getvalue(),
            filename="tone.wav",
            content_type="audio/wav",
            student_context={"grade": "大一"},
        )

        self.assertIsNotNone(response.multimodal_signal)
        self.assertTrue(response.multimodal_signal.analysis_available)
        self.assertEqual(response.multimodal_signal.format, "wav")
        self.assertEqual(response.multimodal_signal.sample_rate_hz, sample_rate)
        self.assertEqual(response.multimodal_signal.channels, 1)
        self.assertGreater(response.multimodal_signal.rms_energy or 0, 0)
        self.assertIn("multimodal_signal", response.to_dict())

    def test_privacy_concern_uses_local_dialogue_policy(self) -> None:
        response = self.agent.handle_text(text="我怕你会告诉别人。")
        self.assertIsNotNone(response.local_policy)
        self.assertEqual(response.local_policy.policy_name, "privacy_concern")
        self.assertEqual(response.local_policy.policy_stage, "rapport_boundary")
        self.assertIn("policy:local", response.metadata.model_backend)

    def test_local_policy_response_includes_referral_decision(self) -> None:
        response = self.agent.handle_text(text="我这几天一直睡不好，也吃不下东西。")
        self.assertIsNotNone(response.local_policy)
        self.assertIsNotNone(response.referral_decision)
        self.assertTrue(response.referral_decision.should_refer)
        self.assertIn(response.referral_decision.urgency, {"watch", "recommended", "urgent"})
        self.assertTrue(
            any(
                "sleep_appetite_drift" in reason or "elevated_entropy" in reason
                for reason in response.referral_decision.reasons
            )
        )

    def test_crisis_response_includes_urgent_referral(self) -> None:
        response = self.agent.handle_text(text="我不想活了，我想伤害自己。")
        self.assertIsNotNone(response.referral_decision)
        self.assertTrue(response.referral_decision.should_refer)
        self.assertEqual(response.referral_decision.urgency, "urgent")
        self.assertTrue(any("risk_level" in reason for reason in response.referral_decision.reasons))

    def test_academic_breakdown_does_not_route_to_crisis_without_safety_signal(self) -> None:
        response = self.agent.handle_text(
            text="我这几天真的快被期末压垮了，明明每天复习，但还是觉得什么都不会，越想越慌，控制不住比较。"
        )
        self.assertNotIn(response.risk.level, {RiskLevel.HIGH, RiskLevel.CRITICAL})
        self.assertIsNone(response.safety.emergency_notice)
        self.assertFalse(response.referral_decision.should_refer)

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
        original_provider = os.environ.get("LLM_PROVIDER")
        os.environ["CAMPUS_NAME"] = "外部环境学校"
        os.environ.pop("LLM_PROVIDER", None)

        try:
            env_path = ROOT / "test_env_override.env"
            env_path.write_text("CAMPUS_NAME=文件中的学校\nLLM_PROVIDER=openai_compatible\n", encoding="utf-8")
            load_env_file(env_path, override=False)

            self.assertEqual(os.environ["CAMPUS_NAME"], "外部环境学校")
            self.assertEqual(os.environ["LLM_PROVIDER"], "openai_compatible")
        finally:
            if original is None:
                os.environ.pop("CAMPUS_NAME", None)
            else:
                os.environ["CAMPUS_NAME"] = original
            if original_provider is None:
                os.environ.pop("LLM_PROVIDER", None)
            else:
                os.environ["LLM_PROVIDER"] = original_provider

    def test_build_local_checkpoint_provider_from_settings(self) -> None:
        settings = Settings(
            llm_provider="local_checkpoint",
            local_checkpoint_path="D:/psychologicalAgent/training/ms_swift/outputs/public_phase0_sft/v0/checkpoint-1",
            local_base_model_path="D:/llm_cache/modelscope/models/Qwen/Qwen3-4B-Instruct-2507",
            llm_max_tokens=256,
        )

        provider = build_llm_provider(settings)

        self.assertIsInstance(provider, LocalCheckpointLLMProvider)
        self.assertEqual(provider.name, "local_checkpoint")
        self.assertEqual(provider.max_tokens, 256)


if __name__ == "__main__":
    unittest.main()
