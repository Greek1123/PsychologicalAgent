from __future__ import annotations

from typing import Any

from .schemas import (
    DynamicAdjustment,
    EntropyTrend,
    PsychologicalEntropy,
    RiskAssessment,
    RiskLevel,
    StateProfile,
)


HIGH_ENTROPY_SCORE = 65
VERY_HIGH_ENTROPY_SCORE = 75
RISING_DELTA = 6
ESCALATING_DELTA = 12
IMPROVING_DELTA = -8


def build_dynamic_adjustment(
    *,
    entropy: PsychologicalEntropy,
    risk: RiskAssessment,
    state_profile: StateProfile | None = None,
    trend_override: EntropyTrend | None = None,
    entropy_trace: list[dict[str, Any]] | None = None,
) -> DynamicAdjustment:
    """Turn entropy, risk, and recent trend into the next adjustment decision."""

    trend = trend_override or entropy.trend
    scores = _recent_scores(entropy.score, entropy_trace)
    reasons = _base_reasons(entropy=entropy, risk=risk, state_profile=state_profile)
    focus = _focus_for_state(state_profile)

    if risk.level == RiskLevel.CRITICAL:
        return _decision(
            stability_state="crisis",
            action="urgent_referral",
            intensity_shift="increase",
            trend=trend,
            should_modify_strategy=True,
            should_refer=True,
            review_window_hours=1,
            next_focus="safety_and_immediate_human_support",
            reasons=[*reasons, "risk:critical"],
        )

    if risk.level == RiskLevel.HIGH:
        return _decision(
            stability_state="high_risk_watch",
            action="human_followup_watch",
            intensity_shift="increase",
            trend=trend,
            should_modify_strategy=True,
            should_refer=True,
            review_window_hours=6,
            next_focus="risk_containment_and_human_followup",
            reasons=[*reasons, "risk:high"],
        )

    sustained_high = len(scores) >= 3 and all(score >= HIGH_ENTROPY_SCORE for score in scores[-3:])
    if sustained_high:
        return _decision(
            stability_state="sustained_high_entropy",
            action="human_followup_watch",
            intensity_shift="increase",
            trend=trend,
            should_modify_strategy=True,
            should_refer=True,
            review_window_hours=12,
            next_focus="reduce_load_and_check_support_access",
            reasons=[*reasons, "entropy:sustained_high"],
        )

    if _delta_at_least(trend, ESCALATING_DELTA) or (
        entropy.score >= HIGH_ENTROPY_SCORE and trend.direction == "up"
    ):
        return _decision(
            stability_state="escalating_entropy",
            action="escalate_support",
            intensity_shift="increase",
            trend=trend,
            should_modify_strategy=True,
            should_refer=entropy.score >= VERY_HIGH_ENTROPY_SCORE,
            review_window_hours=12,
            next_focus="stabilize_now_and_reduce_immediate_pressure",
            reasons=[*reasons, "trend:sharp_rise"],
        )

    if _delta_at_least(trend, RISING_DELTA):
        return _decision(
            stability_state="rising_watch",
            action="soften_and_stabilize",
            intensity_shift="increase",
            trend=trend,
            should_modify_strategy=True,
            should_refer=False,
            review_window_hours=24,
            next_focus="slow_down_and_identify_trigger",
            reasons=[*reasons, "trend:rising"],
        )

    if _delta_at_most(trend, IMPROVING_DELTA):
        return _decision(
            stability_state="improving",
            action="maintain_and_consolidate",
            intensity_shift="decrease",
            trend=trend,
            should_modify_strategy=False,
            should_refer=False,
            review_window_hours=72,
            next_focus="consolidate_working_support",
            reasons=[*reasons, "trend:improving"],
        )

    if trend.previous_score is None:
        return _decision(
            stability_state="first_observation",
            action="baseline_support",
            intensity_shift="maintain",
            trend=trend,
            should_modify_strategy=entropy.score >= HIGH_ENTROPY_SCORE,
            should_refer=entropy.score >= VERY_HIGH_ENTROPY_SCORE,
            review_window_hours=24 if entropy.score >= HIGH_ENTROPY_SCORE else 72,
            next_focus=focus,
            reasons=[*reasons, "trend:baseline"],
        )

    if entropy.score >= HIGH_ENTROPY_SCORE:
        return _decision(
            stability_state="high_entropy_watch",
            action="soften_and_stabilize",
            intensity_shift="maintain",
            trend=trend,
            should_modify_strategy=True,
            should_refer=False,
            review_window_hours=24,
            next_focus="keep_pressure_low_and_track_next_turn",
            reasons=[*reasons, "entropy:high"],
        )

    return _decision(
        stability_state="stable_watch",
        action="maintain_strategy",
        intensity_shift="maintain",
        trend=trend,
        should_modify_strategy=False,
        should_refer=False,
        review_window_hours=48,
        next_focus=focus,
        reasons=[*reasons, "trend:stable"],
    )


def _decision(
    *,
    stability_state: str,
    action: str,
    intensity_shift: str,
    trend: EntropyTrend,
    should_modify_strategy: bool,
    should_refer: bool,
    review_window_hours: int,
    next_focus: str,
    reasons: list[str],
) -> DynamicAdjustment:
    return DynamicAdjustment(
        adjustment_id=f"{stability_state}:{action}",
        stability_state=stability_state,
        action=action,
        intensity_shift=intensity_shift,
        trend_direction=trend.direction,
        trend_delta=trend.delta,
        should_modify_strategy=should_modify_strategy,
        should_refer=should_refer,
        review_window_hours=review_window_hours,
        next_focus=next_focus,
        reasons=_dedupe(reasons),
    )


def _recent_scores(current_score: int, entropy_trace: list[dict[str, Any]] | None) -> list[int]:
    scores: list[int] = []
    for item in entropy_trace or []:
        try:
            scores.append(int(item.get("score")))
        except (TypeError, ValueError):
            continue
    if not scores or scores[-1] != current_score:
        scores.append(current_score)
    return scores


def _base_reasons(
    *,
    entropy: PsychologicalEntropy,
    risk: RiskAssessment,
    state_profile: StateProfile | None,
) -> list[str]:
    reasons = [f"risk:{risk.level}", f"entropy_score:{entropy.score}", f"entropy_state:{entropy.balance_state}"]
    if state_profile is not None:
        reasons.append(f"state:{state_profile.primary_state}")
        if state_profile.weak_input_detected:
            reasons.append("input:weak")
        if state_profile.noisy_input_detected:
            reasons.append("input:noisy")
    return reasons


def _focus_for_state(state_profile: StateProfile | None) -> str:
    if state_profile is None:
        return "supportive_listening"
    return state_profile.recommended_focus or "supportive_listening"


def _delta_at_least(trend: EntropyTrend, threshold: int) -> bool:
    return trend.delta is not None and trend.delta >= threshold


def _delta_at_most(trend: EntropyTrend, threshold: int) -> bool:
    return trend.delta is not None and trend.delta <= threshold


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item not in result:
            result.append(item)
    return result
