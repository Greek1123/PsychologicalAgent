from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.dynamic_adjustment import build_dynamic_adjustment
from campus_support_agent.entropy_orchestration import build_entropy_orchestration
from campus_support_agent.feedback_adaptation import build_feedback_adaptation
from campus_support_agent.intervention_strategy import select_intervention_strategy
from campus_support_agent.reduction import build_entropy_reduction_strategy
from campus_support_agent.schemas import (
    EntropyDimensions,
    EntropyTrend,
    PsychologicalEntropy,
    ReferralDecision,
    RiskAssessment,
    RiskLevel,
    StateProfile,
)


def _risk(level: RiskLevel = RiskLevel.MEDIUM, score: int = 45) -> RiskAssessment:
    return RiskAssessment(level=level, score=score, reason="test")


def _entropy(score: int = 45, *, trend: EntropyTrend | None = None) -> PsychologicalEntropy:
    return PsychologicalEntropy(
        score=score,
        level=2,
        balance_state="stable",
        driver_tags=["cognitive_load"],
        dominant_drivers=["考试压力"],
        dimensions=EntropyDimensions(
            emotion_intensity=2,
            emotional_volatility=2,
            cognitive_load=3,
            physiological_imbalance=2,
            social_support_tension=1,
            risk_pressure=1,
        ),
        trend=trend or EntropyTrend(previous_score=None, delta=None, direction="baseline"),
    )


def _profile() -> StateProfile:
    return StateProfile(
        primary_state="academic_sleep_stress",
        intensity=6,
        confidence=0.8,
        stress_domains=["academic", "sleep"],
        recommended_focus="sleep_stabilization_first",
    )


class EntropyOrchestrationTests(unittest.TestCase):
    def test_builds_boundary_respecting_route_from_memory(self) -> None:
        risk = _risk()
        entropy = _entropy()
        profile = _profile()
        reduction = build_entropy_reduction_strategy(entropy, risk, [])
        strategy = select_intervention_strategy(
            state_profile=profile,
            risk=risk,
            entropy=entropy,
            entropy_reduction=reduction,
        )
        dynamic = build_dynamic_adjustment(entropy=entropy, risk=risk, state_profile=profile)
        feedback = build_feedback_adaptation()

        orchestration = build_entropy_orchestration(
            text="那我可以只说感受吗？",
            conversation_history=[
                {"role": "user", "content": "我不是很想说，我害怕别人会知道。"},
                {"role": "assistant", "content": "你可以不用说具体细节。"},
            ],
            risk=risk,
            entropy=entropy,
            state_profile=profile,
            intervention_strategy=strategy,
            dynamic_adjustment=dynamic,
            feedback_adaptation=feedback,
            referral_decision=ReferralDecision(should_refer=False, urgency="none"),
        )

        self.assertEqual(orchestration.route, "boundary_respecting_support")
        self.assertTrue(any("不要像新会话" in item for item in orchestration.response_constraints))
        self.assertTrue("边界" in orchestration.user_visible_goal or "隐私" in " ".join(orchestration.boundary_flags))

    def test_crisis_route_overrides_other_signals(self) -> None:
        risk = _risk(RiskLevel.CRITICAL, 95)
        entropy = _entropy(70, trend=EntropyTrend(previous_score=55, delta=15, direction="up"))
        profile = _profile()
        reduction = build_entropy_reduction_strategy(entropy, risk, [])
        strategy = select_intervention_strategy(
            state_profile=profile,
            risk=risk,
            entropy=entropy,
            entropy_reduction=reduction,
        )
        dynamic = build_dynamic_adjustment(
            entropy=entropy,
            risk=risk,
            state_profile=profile,
            trend_override=entropy.trend,
        )

        orchestration = build_entropy_orchestration(
            text="我撑不下去了，想伤害自己。",
            conversation_history=[],
            risk=risk,
            entropy=entropy,
            state_profile=profile,
            intervention_strategy=strategy,
            dynamic_adjustment=dynamic,
            feedback_adaptation=build_feedback_adaptation(),
            referral_decision=ReferralDecision(should_refer=True, urgency="urgent"),
        )

        self.assertEqual(orchestration.route, "safety_first")
        self.assertIn("优先确认安全", orchestration.response_constraints)
        self.assertTrue(any("人工" in item or "转介" in item for item in orchestration.hidden_actions))


if __name__ == "__main__":
    unittest.main()
