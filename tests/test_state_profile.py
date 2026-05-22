from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.schemas import (
    EntropyDimensions,
    EntropyTrend,
    PsychologicalEntropy,
    RiskAssessment,
    RiskLevel,
)
from campus_support_agent.safety import evaluate_text_risk
from campus_support_agent.state_profile import build_state_profile


def _risk(level: RiskLevel = RiskLevel.LOW) -> RiskAssessment:
    return RiskAssessment(level=level, score=10, reason="test")


def _entropy(score: int = 45) -> PsychologicalEntropy:
    return PsychologicalEntropy(
        score=score,
        level=3,
        balance_state="strained",
        driver_tags=["cognitive_load"],
        dominant_drivers=["test"],
        dimensions=EntropyDimensions(
            emotion_intensity=2,
            emotional_volatility=1,
            cognitive_load=3,
            physiological_imbalance=2,
            social_support_tension=1,
            risk_pressure=1,
        ),
        trend=EntropyTrend(previous_score=None, delta=None, direction="baseline"),
    )


class StateProfileTests(unittest.TestCase):
    def test_common_exam_sleep_pressure_is_not_crisis_routed(self) -> None:
        risk = evaluate_text_risk("我最近快期末考试了，压力好大，晚上睡不着。")

        self.assertEqual(risk.level, RiskLevel.MEDIUM)
        self.assertFalse(risk.needs_human_followup)

    def test_exam_and_sleep_pressure_are_structured(self) -> None:
        profile = build_state_profile(
            "最近快期末考试了，我好害怕挂科，压力好大，晚上总睡不着。",
            risk=_risk(RiskLevel.MEDIUM),
            entropy=_entropy(58),
        )

        self.assertEqual(profile.primary_state, "academic_sleep_stress")
        self.assertIn("academic", profile.stress_domains)
        self.assertIn("sleep", profile.stress_domains)
        self.assertIn("anxiety", profile.emotion_signals)
        self.assertIn("sleep_disruption", profile.body_signals)
        self.assertEqual(profile.recommended_focus, "sleep_stabilization_first")

    def test_privacy_boundary_prioritizes_confidentiality(self) -> None:
        profile = build_state_profile(
            "我不是很想说，我怕你会告诉别人。",
            risk=_risk(),
            entropy=_entropy(34),
        )

        self.assertEqual(profile.primary_state, "privacy_boundary")
        self.assertIn("privacy_concern", profile.boundary_flags)
        self.assertEqual(profile.recommended_focus, "confidentiality_and_control")

    def test_weak_input_uses_recent_context(self) -> None:
        profile = build_state_profile(
            "？",
            risk=_risk(),
            entropy=_entropy(28),
            conversation_history=[
                {"role": "user", "content": "我一回宿舍就烦，见到舍友就很难受。"},
                {"role": "assistant", "content": "我在。"},
            ],
        )

        self.assertTrue(profile.weak_input_detected)
        self.assertIn("dorm", profile.stress_domains)
        self.assertEqual(profile.recommended_focus, "low_pressure_presence")

    def test_crisis_risk_overrides_other_states(self) -> None:
        profile = build_state_profile(
            "我真的撑不住了，不想活了。",
            risk=RiskAssessment(
                level=RiskLevel.CRITICAL,
                score=95,
                reason="test",
                trigger_terms=["self_harm"],
                needs_human_followup=True,
            ),
            entropy=_entropy(90),
        )

        self.assertEqual(profile.primary_state, "safety_risk")
        self.assertEqual(profile.recommended_focus, "safety_and_human_referral")
        self.assertGreaterEqual(profile.intensity, 9)
        self.assertIn("self_harm", profile.risk_signals)

    def test_group_work_marginalization_maps_to_visibility_focus(self) -> None:
        profile = build_state_profile(
            "小组作业里组员一直不回我，我做了资料整理但怕老师觉得我没贡献。",
            risk=_risk(RiskLevel.MEDIUM),
            entropy=_entropy(42),
        )

        self.assertEqual(profile.primary_state, "group_work_marginalization")
        self.assertIn("group_work", profile.stress_domains)
        self.assertEqual(profile.recommended_focus, "contribution_visibility")

    def test_public_speaking_panic_maps_to_grounding_focus(self) -> None:
        profile = build_state_profile(
            "我明天要上台汇报，一想到所有人都看着我就手抖，脑子空白，怕忘词。",
            risk=_risk(RiskLevel.MEDIUM),
            entropy=_entropy(44),
        )

        self.assertEqual(profile.primary_state, "public_speaking_panic")
        self.assertIn("public_speaking", profile.stress_domains)
        self.assertEqual(profile.recommended_focus, "performance_grounding")

    def test_game_avoidance_loop_maps_to_escape_interruption_focus(self) -> None:
        profile = build_state_profile(
            "我最近一直打游戏到凌晨，白天不想去上课，感觉控制不了自己。",
            risk=_risk(RiskLevel.MEDIUM),
            entropy=_entropy(50),
        )

        self.assertEqual(profile.primary_state, "game_avoidance_loop")
        self.assertIn("game_avoidance", profile.stress_domains)
        self.assertEqual(profile.recommended_focus, "escape_loop_interruption")

    def test_family_middleman_maps_to_sustainable_boundary_focus(self) -> None:
        profile = build_state_profile(
            "父母离婚以后，我妈每天听我说话，还一直骂爸爸，我没人可以说话。",
            risk=_risk(RiskLevel.MEDIUM),
            entropy=_entropy(48),
        )

        self.assertEqual(profile.primary_state, "family_middleman_stress")
        self.assertIn("family_middleman", profile.stress_domains)
        self.assertEqual(profile.recommended_focus, "family_boundary_sustainability")

    def test_pet_grief_maps_to_self_blame_relief_focus(self) -> None:
        profile = build_state_profile(
            "我的宠物走了，我一直想如果我早点发现它会不会还在，都是我没照顾好。",
            risk=_risk(RiskLevel.MEDIUM),
            entropy=_entropy(45),
        )

        self.assertEqual(profile.primary_state, "pet_grief")
        self.assertIn("pet_grief", profile.stress_domains)
        self.assertEqual(profile.recommended_focus, "grief_without_self_blame")

    def test_campus_safety_fear_maps_to_reporting_focus(self) -> None:
        profile = build_state_profile(
            "我感觉有人尾随我，但是没有证据，现在不敢独自回宿舍。",
            risk=_risk(RiskLevel.MEDIUM),
            entropy=_entropy(55),
        )

        self.assertEqual(profile.primary_state, "campus_safety_fear")
        self.assertIn("safety_fear", profile.stress_domains)
        self.assertEqual(profile.recommended_focus, "safety_reporting_without_blame")


if __name__ == "__main__":
    unittest.main()
