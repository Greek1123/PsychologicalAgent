from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.dynamic_adjustment import build_dynamic_adjustment
from campus_support_agent.schemas import (
    EntropyDimensions,
    EntropyTrend,
    PsychologicalEntropy,
    RiskAssessment,
    RiskLevel,
    StateProfile,
)


def _risk(level: RiskLevel = RiskLevel.LOW, score: int = 15) -> RiskAssessment:
    return RiskAssessment(level=level, score=score, reason="test")


def _entropy(score: int = 42, *, trend: EntropyTrend | None = None) -> PsychologicalEntropy:
    return PsychologicalEntropy(
        score=score,
        level=4 if score >= 65 else 2,
        balance_state="fragile" if score >= 65 else "stable",
        driver_tags=["cognitive_load"],
        dominant_drivers=["exam"],
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
        recommended_focus="sleep_stabilization_first",
    )


class DynamicAdjustmentTests(unittest.TestCase):
    def test_first_observation_uses_baseline_support(self) -> None:
        adjustment = build_dynamic_adjustment(
            entropy=_entropy(40),
            risk=_risk(),
            state_profile=_profile(),
        )

        self.assertEqual(adjustment.stability_state, "first_observation")
        self.assertEqual(adjustment.action, "baseline_support")
        self.assertFalse(adjustment.should_refer)
        self.assertEqual(adjustment.next_focus, "sleep_stabilization_first")

    def test_rising_entropy_softens_and_stabilizes(self) -> None:
        adjustment = build_dynamic_adjustment(
            entropy=_entropy(58),
            risk=_risk(RiskLevel.MEDIUM, score=40),
            state_profile=_profile(),
            trend_override=EntropyTrend(previous_score=50, delta=8, direction="up"),
        )

        self.assertEqual(adjustment.stability_state, "rising_watch")
        self.assertEqual(adjustment.action, "soften_and_stabilize")
        self.assertTrue(adjustment.should_modify_strategy)

    def test_sharp_rise_escalates_support(self) -> None:
        adjustment = build_dynamic_adjustment(
            entropy=_entropy(70),
            risk=_risk(RiskLevel.MEDIUM, score=45),
            state_profile=_profile(),
            trend_override=EntropyTrend(previous_score=55, delta=15, direction="up"),
        )

        self.assertEqual(adjustment.stability_state, "escalating_entropy")
        self.assertEqual(adjustment.action, "escalate_support")
        self.assertEqual(adjustment.intensity_shift, "increase")

    def test_sustained_high_entropy_recommends_followup(self) -> None:
        adjustment = build_dynamic_adjustment(
            entropy=_entropy(72),
            risk=_risk(RiskLevel.MEDIUM, score=45),
            state_profile=_profile(),
            trend_override=EntropyTrend(previous_score=70, delta=2, direction="up"),
            entropy_trace=[
                {"score": 66},
                {"score": 70},
                {"score": 72},
            ],
        )

        self.assertEqual(adjustment.stability_state, "sustained_high_entropy")
        self.assertTrue(adjustment.should_refer)
        self.assertEqual(adjustment.action, "human_followup_watch")

    def test_improving_entropy_consolidates_current_strategy(self) -> None:
        adjustment = build_dynamic_adjustment(
            entropy=_entropy(42),
            risk=_risk(),
            state_profile=_profile(),
            trend_override=EntropyTrend(previous_score=55, delta=-13, direction="down"),
        )

        self.assertEqual(adjustment.stability_state, "improving")
        self.assertEqual(adjustment.intensity_shift, "decrease")
        self.assertFalse(adjustment.should_refer)

    def test_critical_risk_overrides_entropy_trend(self) -> None:
        adjustment = build_dynamic_adjustment(
            entropy=_entropy(38),
            risk=_risk(RiskLevel.CRITICAL, score=95),
            state_profile=_profile(),
            trend_override=EntropyTrend(previous_score=45, delta=-7, direction="down"),
        )

        self.assertEqual(adjustment.stability_state, "crisis")
        self.assertEqual(adjustment.action, "urgent_referral")
        self.assertTrue(adjustment.should_refer)


if __name__ == "__main__":
    unittest.main()
