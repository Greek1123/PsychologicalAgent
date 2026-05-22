from __future__ import annotations

from collections import Counter
from typing import Any

from .schemas import EntropyTrendWarning


RISK_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}


def build_entropy_trend_warning(
    *,
    session_id: str,
    records: list[dict[str, Any]],
    entropy_trace: list[dict[str, Any]] | None = None,
    feedback_summary: dict[str, Any] | None = None,
    audit_summary: dict[str, Any] | None = None,
) -> EntropyTrendWarning:
    scores = _scores(records, entropy_trace or [])
    recent_scores = scores[-5:]
    risk_levels = [str(record.get("risk_level") or "low") for record in records]
    loop_actions = Counter(str(record.get("adjustment_loop_action")) for record in records if record.get("adjustment_loop_action"))
    feedback_summary = feedback_summary or {}
    audit_summary = audit_summary or {}

    reasons: list[str] = []
    level = "none"
    trend_state = _trend_state(recent_scores)
    recommended_action = "continue_observation"
    user_visible_mode = "normal_support"
    review_window_hours = 72

    if "critical" in risk_levels:
        level = "critical"
        recommended_action = "activate_safety_protocol"
        user_visible_mode = "safety_first"
        review_window_hours = 1
        reasons.append("critical_risk_seen")
    elif "high" in risk_levels:
        level = "high"
        recommended_action = "recommend_human_followup"
        user_visible_mode = "warm_human_linkage"
        review_window_hours = 12
        reasons.append("high_risk_seen")

    if trend_state == "sustained_high":
        level = _max_level(level, "high")
        recommended_action = "manual_followup_and_low_pressure_support"
        user_visible_mode = "stabilize_before_problem_solving"
        review_window_hours = min(review_window_hours, 12)
        reasons.append("entropy_sustained_high")
    elif trend_state == "rising":
        level = _max_level(level, "medium")
        recommended_action = "reduce_pressure_and_review_next_turn"
        user_visible_mode = "low_pressure_support"
        review_window_hours = min(review_window_hours, 24)
        reasons.append("entropy_rising")
    elif trend_state == "volatile":
        level = _max_level(level, "medium")
        recommended_action = "stabilize_and_monitor_volatility"
        user_visible_mode = "grounding_then_check"
        review_window_hours = min(review_window_hours, 24)
        reasons.append("entropy_volatile")

    if loop_actions.get("repair_reply_style", 0) >= 2 or int(feedback_summary.get("negative_count") or 0) >= 2:
        level = _max_level(level, "medium")
        recommended_action = "repair_intervention_style"
        user_visible_mode = "repair_conversation"
        review_window_hours = min(review_window_hours, 24)
        reasons.append("repeated_reply_repair_needed")

    latest_route = str(audit_summary.get("latest_decision_route") or "")
    if latest_route in {"emergency_referral", "human_support_recommended"}:
        level = _max_level(level, "high")
        recommended_action = "confirm_real_world_support_path"
        user_visible_mode = "warm_human_linkage"
        review_window_hours = min(review_window_hours, 12)
        reasons.append(f"latest_audit_route:{latest_route}")

    should_alert = level in {"medium", "high", "critical"}
    if not reasons:
        reasons.append("no_warning_signal")

    return EntropyTrendWarning(
        warning_id=f"{session_id}:trend_warning:{len(records)}",
        session_id=session_id,
        level=level,
        trend_state=trend_state,
        should_alert=should_alert,
        review_window_hours=review_window_hours,
        recommended_action=recommended_action,
        user_visible_mode=user_visible_mode,
        trigger_reasons=_dedupe(reasons),
        evidence={
            "scores": recent_scores,
            "latest_score": recent_scores[-1] if recent_scores else None,
            "score_delta": _delta(recent_scores),
            "volatility": _volatility(recent_scores),
            "risk_levels": dict(Counter(risk_levels)),
            "loop_actions": dict(loop_actions),
            "negative_feedback_count": int(feedback_summary.get("negative_count") or 0),
            "latest_audit_route": latest_route,
        },
    )


def summarize_trend_warnings(warnings: list[dict[str, Any]]) -> dict[str, Any]:
    levels: dict[str, int] = {}
    trend_states: dict[str, int] = {}
    actions: dict[str, int] = {}
    alert_count = 0
    for warning in warnings:
        level = str(warning.get("level") or "none")
        trend_state = str(warning.get("trend_state") or "unknown")
        action = str(warning.get("recommended_action") or "unknown")
        levels[level] = levels.get(level, 0) + 1
        trend_states[trend_state] = trend_states.get(trend_state, 0) + 1
        actions[action] = actions.get(action, 0) + 1
        if warning.get("should_alert"):
            alert_count += 1
    return {
        "total_warnings": len(warnings),
        "alert_count": alert_count,
        "levels": levels,
        "trend_states": trend_states,
        "recommended_actions": actions,
    }


def _scores(records: list[dict[str, Any]], entropy_trace: list[dict[str, Any]]) -> list[int]:
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


def _trend_state(scores: list[int]) -> str:
    if len(scores) < 2:
        return "baseline"
    if len(scores) >= 3 and all(score >= 65 for score in scores[-3:]):
        return "sustained_high"
    delta = _delta(scores)
    if delta is not None and delta >= 10:
        return "rising"
    if delta is not None and delta <= -10:
        return "falling"
    if _volatility(scores) >= 18:
        return "volatile"
    return "stable"


def _delta(scores: list[int]) -> int | None:
    if len(scores) < 2:
        return None
    return scores[-1] - scores[0]


def _volatility(scores: list[int]) -> int:
    if len(scores) < 2:
        return 0
    return max(abs(current - previous) for previous, current in zip(scores, scores[1:]))


def _max_level(current: str, candidate: str) -> str:
    rank = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
    return candidate if rank.get(candidate, 0) > rank.get(current, 0) else current


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item and item not in result:
            result.append(item)
    return result
