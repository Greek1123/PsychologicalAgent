from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.intervention_strategy import select_intervention_strategy
from campus_support_agent.schemas import (
    EntropyDimensions,
    EntropyReductionStrategy,
    EntropyTrend,
    PsychologicalEntropy,
    RiskAssessment,
    RiskLevel,
)
from campus_support_agent.state_profile import build_state_profile


def _entropy(score: int = 35) -> PsychologicalEntropy:
    return PsychologicalEntropy(
        score=score,
        level=2,
        balance_state="stable",
        driver_tags=["cognitive_load"],
        dominant_drivers=["test"],
        dimensions=EntropyDimensions(
            emotion_intensity=2,
            emotional_volatility=1,
            cognitive_load=3,
            physiological_imbalance=1,
            social_support_tension=1,
            risk_pressure=1,
        ),
        trend=EntropyTrend(previous_score=None, delta=None, direction="baseline"),
    )


def _reduction() -> EntropyReductionStrategy:
    return EntropyReductionStrategy(
        target_state="stable",
        targeted_drivers=["cognitive_load"],
        rationale="test",
        core_actions=["test"],
        expected_delta_score=-6,
        review_window_hours=72,
    )


class InterventionStrategyTests(unittest.TestCase):
    def test_privacy_state_selects_confidentiality_strategy(self) -> None:
        risk = RiskAssessment(level=RiskLevel.MEDIUM, score=45, reason="test")
        entropy = _entropy()
        profile = build_state_profile("我不想说，我怕别人知道。", risk=risk, entropy=entropy)

        strategy = select_intervention_strategy(
            state_profile=profile,
            risk=risk,
            entropy=entropy,
            entropy_reduction=_reduction(),
        )

        self.assertEqual(strategy.strategy_id, "privacy_reassurance")
        self.assertEqual(strategy.response_mode, "trust_boundary")
        self.assertEqual(strategy.max_questions, 1)
        self.assertIn("privacy", strategy.tags)

    def test_crisis_risk_overrides_profile_focus(self) -> None:
        risk = RiskAssessment(
            level=RiskLevel.CRITICAL,
            score=95,
            reason="test",
            trigger_terms=["self_harm"],
            needs_human_followup=True,
        )
        entropy = _entropy(90)
        profile = build_state_profile("我不想活了。", risk=risk, entropy=entropy)

        strategy = select_intervention_strategy(
            state_profile=profile,
            risk=risk,
            entropy=entropy,
            entropy_reduction=_reduction(),
        )

        self.assertEqual(strategy.strategy_id, "safety_first")
        self.assertFalse(strategy.should_ask_question)
        self.assertEqual(strategy.priority, "urgent")

    def test_sleep_state_selects_sleep_stabilization(self) -> None:
        risk = RiskAssessment(level=RiskLevel.MEDIUM, score=45, reason="test")
        entropy = _entropy(42)
        profile = build_state_profile("我压力很大，晚上总是睡不着。", risk=risk, entropy=entropy)

        strategy = select_intervention_strategy(
            state_profile=profile,
            risk=risk,
            entropy=entropy,
            entropy_reduction=_reduction(),
        )

        self.assertEqual(strategy.strategy_id, "sleep_stabilization")
        self.assertEqual(strategy.response_mode, "practical_stabilization")


    def test_group_work_state_selects_visibility_strategy(self) -> None:
        risk = RiskAssessment(level=RiskLevel.MEDIUM, score=45, reason="test")
        entropy = _entropy(38)
        profile = build_state_profile("小组作业里他们一直不回我，我怕老师觉得我没贡献。", risk=risk, entropy=entropy)

        strategy = select_intervention_strategy(
            state_profile=profile,
            risk=risk,
            entropy=entropy,
            entropy_reduction=_reduction(),
        )

        self.assertEqual(profile.primary_state, "group_work_marginalization")
        self.assertEqual(strategy.strategy_id, "group_work_visibility")
        self.assertIn("visibility", strategy.tags)

    def test_game_avoidance_state_selects_escape_loop_strategy(self) -> None:
        risk = RiskAssessment(level=RiskLevel.MEDIUM, score=45, reason="test")
        entropy = _entropy(44)
        profile = build_state_profile("我最近每天打游戏到凌晨，白天课也不想去，感觉控制不了自己。", risk=risk, entropy=entropy)

        strategy = select_intervention_strategy(
            state_profile=profile,
            risk=risk,
            entropy=entropy,
            entropy_reduction=_reduction(),
        )

        self.assertEqual(profile.primary_state, "game_avoidance_loop")
        self.assertEqual(strategy.strategy_id, "escape_loop_interruption")
        self.assertIn("behavior_loop", strategy.tags)

    def test_public_speaking_state_selects_grounding_strategy(self) -> None:
        risk = RiskAssessment(level=RiskLevel.MEDIUM, score=45, reason="test")
        entropy = _entropy(40)
        profile = build_state_profile("我明天要上台汇报，一想到所有人都看着我就手抖，怕自己忘词。", risk=risk, entropy=entropy)

        strategy = select_intervention_strategy(
            state_profile=profile,
            risk=risk,
            entropy=entropy,
            entropy_reduction=_reduction(),
        )

        self.assertEqual(profile.primary_state, "public_speaking_panic")
        self.assertEqual(strategy.strategy_id, "performance_grounding")
        self.assertEqual(strategy.response_mode, "body_grounding_and_anchor_plan")

    def test_family_middleman_state_selects_boundary_strategy(self) -> None:
        risk = RiskAssessment(level=RiskLevel.MEDIUM, score=45, reason="test")
        entropy = _entropy(48)
        profile = build_state_profile("父母离婚以后我妈每天听我说话，还一直骂爸爸，我没人可以说话。", risk=risk, entropy=entropy)

        strategy = select_intervention_strategy(
            state_profile=profile,
            risk=risk,
            entropy=entropy,
            entropy_reduction=_reduction(),
        )

        self.assertEqual(profile.primary_state, "family_middleman_stress")
        self.assertEqual(strategy.strategy_id, "family_boundary_sustainability")
        self.assertIn("boundary", strategy.tags)

    def test_pet_grief_state_selects_self_blame_relief_strategy(self) -> None:
        risk = RiskAssessment(level=RiskLevel.MEDIUM, score=45, reason="test")
        entropy = _entropy(43)
        profile = build_state_profile("我的宠物走了，我一直觉得如果我早点发现它会不会还在，都是我没照顾好。", risk=risk, entropy=entropy)

        strategy = select_intervention_strategy(
            state_profile=profile,
            risk=risk,
            entropy=entropy,
            entropy_reduction=_reduction(),
        )

        self.assertEqual(profile.primary_state, "pet_grief")
        self.assertEqual(strategy.strategy_id, "grief_without_self_blame")
        self.assertIn("self_blame", strategy.tags)

    def test_campus_safety_state_selects_reporting_strategy(self) -> None:
        risk = RiskAssessment(level=RiskLevel.MEDIUM, score=45, reason="test")
        entropy = _entropy(52)
        profile = build_state_profile("我最近感觉有人尾随我，但是我没有证据，现在不敢独自回宿舍。", risk=risk, entropy=entropy)

        strategy = select_intervention_strategy(
            state_profile=profile,
            risk=risk,
            entropy=entropy,
            entropy_reduction=_reduction(),
        )

        self.assertEqual(profile.primary_state, "campus_safety_fear")
        self.assertEqual(strategy.strategy_id, "safety_reporting_without_blame")
        self.assertEqual(strategy.priority, "high")


if __name__ == "__main__":
    unittest.main()
