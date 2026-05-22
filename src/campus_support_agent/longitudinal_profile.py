from __future__ import annotations

from collections import Counter
from typing import Any

from .schemas import LongitudinalStateProfile


RISK_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}


def build_longitudinal_state_profile(
    *,
    session_id: str,
    records: list[dict[str, Any]],
    entropy_trace: list[dict[str, Any]],
    referral_events: list[dict[str, Any]],
    feedback_summary: dict[str, Any] | None = None,
) -> LongitudinalStateProfile:
    feedback_summary = feedback_summary or {}
    scores = _entropy_scores(records, entropy_trace)
    risk_levels = [str(record.get("risk_level") or "low") for record in records]
    state_counts = Counter(
        str(record.get("primary_state"))
        for record in records
        if record.get("primary_state")
    )
    stress_domains = Counter(_iter_stress_domains(records))
    dynamic_actions = Counter(
        str(record.get("dynamic_action"))
        for record in records
        if record.get("dynamic_action")
    )

    entropy_course = _entropy_course(scores)
    risk_course = _risk_course(risk_levels, referral_events)
    volatility_score = _volatility_score(scores)
    average_entropy = round(sum(scores) / len(scores), 1) if scores else None
    latest_entropy = scores[-1] if scores else None
    peak_entropy = max(scores) if scores else None
    engagement_signal = _engagement_signal(records, feedback_summary)
    care_level = _recommended_care_level(
        risk_course=risk_course,
        entropy_course=entropy_course,
        latest_entropy=latest_entropy,
        peak_entropy=peak_entropy,
        volatility_score=volatility_score,
        referral_events=referral_events,
        feedback_summary=feedback_summary,
    )

    return LongitudinalStateProfile(
        profile_id=f"{session_id}:longitudinal:{len(records)}",
        session_id=session_id,
        observation_count=len(records),
        dominant_states=[item for item, _ in state_counts.most_common(3)],
        dominant_stress_domains=[item for item, _ in stress_domains.most_common(4)],
        entropy_course=entropy_course,
        risk_course=risk_course,
        average_entropy=average_entropy,
        latest_entropy_score=latest_entropy,
        peak_entropy_score=peak_entropy,
        volatility_score=volatility_score,
        engagement_signal=engagement_signal,
        recommended_care_level=care_level,
        next_review_hours=_next_review_hours(care_level),
        priority_actions=_priority_actions(
            care_level=care_level,
            entropy_course=entropy_course,
            risk_course=risk_course,
            dynamic_actions=dynamic_actions,
            feedback_summary=feedback_summary,
        ),
        evidence={
            "risk_levels": dict(Counter(risk_levels)),
            "dynamic_actions": dict(dynamic_actions),
            "referral_event_count": len(referral_events),
            "feedback_summary": feedback_summary,
            "entropy_points": len(scores),
        },
    )


def _entropy_scores(records: list[dict[str, Any]], entropy_trace: list[dict[str, Any]]) -> list[int]:
    scores: list[int] = []
    for item in entropy_trace:
        try:
            scores.append(int(item.get("score")))
        except (TypeError, ValueError):
            continue
    if scores:
        return scores

    for record in records:
        try:
            scores.append(int(record.get("entropy_score")))
        except (TypeError, ValueError):
            continue
    return scores


def _iter_stress_domains(records: list[dict[str, Any]]) -> list[str]:
    domains: list[str] = []
    for record in records:
        state_profile = record.get("state_profile") or {}
        for domain in state_profile.get("stress_domains") or []:
            clean = str(domain).strip()
            if clean:
                domains.append(clean)
    return domains


def _entropy_course(scores: list[int]) -> str:
    if len(scores) < 2:
        return "baseline"
    first = scores[0]
    latest = scores[-1]
    delta = latest - first
    if len(scores) >= 3 and all(score >= 65 for score in scores[-3:]):
        return "sustained_high"
    if delta >= 10:
        return "rising"
    if delta <= -10:
        return "falling"
    if _volatility_score(scores) >= 18:
        return "volatile"
    return "stable"


def _risk_course(risk_levels: list[str], referral_events: list[dict[str, Any]]) -> str:
    if "critical" in risk_levels:
        return "critical_seen"
    if "high" in risk_levels:
        return "high_seen"
    if any(event.get("manual_referral_recommended") for event in referral_events):
        return "manual_watch"
    if risk_levels.count("medium") >= 3:
        return "repeated_medium"
    return "low_to_medium"


def _volatility_score(scores: list[int]) -> int:
    if len(scores) < 2:
        return 0
    return max(abs(current - previous) for previous, current in zip(scores, scores[1:]))


def _engagement_signal(records: list[dict[str, Any]], feedback_summary: dict[str, Any]) -> str:
    total_feedback = int(feedback_summary.get("total_feedback") or 0)
    if len(records) >= 6 or total_feedback >= 3:
        return "active"
    if len(records) >= 2 or total_feedback:
        return "moderate"
    if records:
        return "initial"
    return "none"


def _recommended_care_level(
    *,
    risk_course: str,
    entropy_course: str,
    latest_entropy: int | None,
    peak_entropy: int | None,
    volatility_score: int,
    referral_events: list[dict[str, Any]],
    feedback_summary: dict[str, Any],
) -> str:
    if risk_course == "critical_seen":
        return "urgent"
    if risk_course in {"high_seen", "manual_watch"}:
        return "manual_followup"
    if entropy_course == "sustained_high" or (latest_entropy is not None and latest_entropy >= 70):
        return "manual_followup"
    if entropy_course in {"rising", "volatile"} or volatility_score >= 18:
        return "watch_closely"
    if peak_entropy is not None and peak_entropy >= 65:
        return "watch_closely"
    if int(feedback_summary.get("negative_count") or 0) >= 2:
        return "strategy_repair"
    if referral_events:
        return "watch_closely"
    return "observe"


def _next_review_hours(care_level: str) -> int:
    mapping = {
        "urgent": 1,
        "manual_followup": 12,
        "watch_closely": 24,
        "strategy_repair": 24,
        "observe": 72,
    }
    return mapping.get(care_level, 72)


def _priority_actions(
    *,
    care_level: str,
    entropy_course: str,
    risk_course: str,
    dynamic_actions: Counter[str],
    feedback_summary: dict[str, Any],
) -> list[str]:
    actions: list[str] = []
    if care_level == "urgent":
        actions.append("activate_safety_protocol")
    if care_level == "manual_followup":
        actions.append("recommend_human_followup")
    if entropy_course in {"rising", "sustained_high", "volatile"}:
        actions.append("reduce_intervention_pressure")
        actions.append("track_entropy_next_turn")
    if risk_course in {"repeated_medium", "manual_watch"}:
        actions.append("check_offline_support_access")
    if dynamic_actions.get("soften_and_stabilize", 0) >= 2:
        actions.append("continue_low_pressure_strategy")
    if int(feedback_summary.get("negative_count") or 0) > 0:
        actions.append("repair_response_style")
    if not actions:
        actions.append("continue_observation")
    return _dedupe(actions)


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item not in result:
            result.append(item)
    return result
