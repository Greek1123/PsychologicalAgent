from __future__ import annotations

from collections import Counter
from typing import Any


RISK_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}


def build_entropy_reduction_loop(
    *,
    session_id: str,
    records: list[dict[str, Any]],
    feedback_summary: dict[str, Any] | None = None,
    reply_quality_summary: dict[str, Any] | None = None,
    strategy_version: dict[str, Any] | None = None,
    intervention_effectiveness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score whether the intervention loop is actually reducing entropy.

    This is a backend-only evaluation layer. It connects entropy trajectory,
    risk shifts, user feedback, reply quality, and strategy-version decisions
    into one closed-loop verdict.
    """

    feedback_summary = feedback_summary or {}
    reply_quality_summary = reply_quality_summary or {}
    strategy_version = strategy_version or {}
    intervention_effectiveness = intervention_effectiveness or {}
    timeline = build_entropy_reduction_loop_timeline(records)
    summary = summarize_entropy_reduction_loop(
        timeline,
        feedback_summary=feedback_summary,
        reply_quality_summary=reply_quality_summary,
        strategy_version=strategy_version,
        intervention_effectiveness=intervention_effectiveness,
    )
    return {
        "session_id": session_id,
        "total_records": len(records),
        "summary": summary,
        "latest_loop_point": timeline[-1] if timeline else None,
        "timeline": timeline,
    }


def build_entropy_reduction_loop_timeline(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        previous = records[index - 1] if index > 0 else None
        entropy_delta = _entropy_delta(previous, record)
        risk_shift = _risk_shift(previous, record)
        strategy_family = _strategy_family(record)
        loop_score = _loop_score(record, entropy_delta, risk_shift)
        status = _point_status(record, entropy_delta, risk_shift, loop_score)
        timeline.append(
            {
                "response_id": record.get("response_id"),
                "created_at": record.get("created_at"),
                "input_preview": str(record.get("input_text") or "")[:160],
                "entropy_score": _safe_int(record.get("entropy_score")),
                "entropy_delta": entropy_delta,
                "risk_level": record.get("risk_level"),
                "risk_shift": risk_shift,
                "strategy_family": strategy_family,
                "orchestration_route": record.get("orchestration_route") or "legacy_or_unclassified",
                "dynamic_action": record.get("dynamic_action"),
                "adjustment_loop_action": record.get("adjustment_loop_action"),
                "reduction_goal_driver": record.get("reduction_goal_active_driver"),
                "loop_status": status,
                "loop_score": loop_score,
                "next_action": _point_next_action(status, record),
                "evidence": {
                    "risk_score": record.get("risk_score"),
                    "local_policy_name": record.get("local_policy_name"),
                    "feedback_adaptation_mode": record.get("feedback_adaptation_mode"),
                    "referral_urgency": record.get("referral_urgency"),
                    "referral_should_refer": bool(record.get("referral_should_refer")),
                },
            }
        )
    return timeline


def summarize_entropy_reduction_loop(
    timeline: list[dict[str, Any]],
    *,
    feedback_summary: dict[str, Any] | None = None,
    reply_quality_summary: dict[str, Any] | None = None,
    strategy_version: dict[str, Any] | None = None,
    intervention_effectiveness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    feedback_summary = feedback_summary or {}
    reply_quality_summary = reply_quality_summary or {}
    strategy_version = strategy_version or {}
    intervention_effectiveness = intervention_effectiveness or {}
    status_counts = Counter(str(item.get("loop_status") or "unknown") for item in timeline)
    strategy_counts = Counter(str(item.get("strategy_family") or "unknown") for item in timeline)
    scores = [int(item.get("loop_score") or 0) for item in timeline]
    entropy_values = [
        int(item["entropy_score"])
        for item in timeline
        if isinstance(item.get("entropy_score"), int)
    ]
    latest = timeline[-1] if timeline else {}
    overall_status = _overall_status(
        latest_status=str(latest.get("loop_status") or "insufficient_data"),
        average_score=round(sum(scores) / len(scores), 1) if scores else None,
        feedback_summary=feedback_summary,
        reply_quality_summary=reply_quality_summary,
        strategy_version=strategy_version,
        intervention_effectiveness=intervention_effectiveness,
    )
    return {
        "overall_status": overall_status,
        "average_loop_score": round(sum(scores) / len(scores), 1) if scores else None,
        "latest_loop_score": latest.get("loop_score"),
        "latest_loop_status": latest.get("loop_status"),
        "latest_next_action": latest.get("next_action"),
        "first_entropy_score": entropy_values[0] if entropy_values else None,
        "latest_entropy_score": entropy_values[-1] if entropy_values else None,
        "session_entropy_delta": (entropy_values[-1] - entropy_values[0]) if len(entropy_values) >= 2 else None,
        "best_entropy_score": min(entropy_values) if entropy_values else None,
        "worst_entropy_score": max(entropy_values) if entropy_values else None,
        "status_counts": dict(status_counts),
        "strategy_family_counts": dict(strategy_counts),
        "feedback_signal": _feedback_signal(feedback_summary),
        "reply_quality_signal": _reply_quality_signal(reply_quality_summary),
        "strategy_decision": strategy_version.get("decision"),
        "strategy_target_family": strategy_version.get("target_strategy_family"),
        "should_switch_strategy": bool(strategy_version.get("should_switch_strategy")),
        "recommended_backend_action": _recommended_backend_action(
            overall_status=overall_status,
            strategy_version=strategy_version,
            latest=latest,
        ),
    }


def build_entropy_reduction_loop_overview(session_loops: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    scores: list[int] = []
    watch_sessions: list[dict[str, Any]] = []
    for loop in session_loops:
        summary = loop.get("summary") or {}
        status = str(summary.get("overall_status") or "unknown")
        action = str(summary.get("recommended_backend_action") or "continue_observation")
        status_counts[status] += 1
        action_counts[action] += 1
        latest_score = summary.get("latest_loop_score")
        if isinstance(latest_score, int):
            scores.append(latest_score)
        if status in {"not_reducing", "strategy_repair_needed", "human_followup_needed", "crisis_priority"}:
            watch_sessions.append(
                {
                    "session_id": loop.get("session_id"),
                    "overall_status": status,
                    "latest_loop_score": latest_score,
                    "session_entropy_delta": summary.get("session_entropy_delta"),
                    "recommended_backend_action": action,
                    "strategy_decision": summary.get("strategy_decision"),
                    "strategy_target_family": summary.get("strategy_target_family"),
                }
            )
    return {
        "total_sessions": len(session_loops),
        "average_latest_loop_score": round(sum(scores) / len(scores), 1) if scores else None,
        "status_counts": dict(status_counts),
        "recommended_action_counts": dict(action_counts),
        "watch_sessions": watch_sessions[-30:],
    }


def _entropy_delta(previous: dict[str, Any] | None, current: dict[str, Any]) -> int | None:
    if previous is None:
        return None
    previous_score = _safe_int(previous.get("entropy_score"))
    current_score = _safe_int(current.get("entropy_score"))
    if previous_score is None or current_score is None:
        return None
    return current_score - previous_score


def _risk_shift(previous: dict[str, Any] | None, current: dict[str, Any]) -> str:
    if previous is None:
        return "baseline"
    previous_rank = RISK_RANK.get(str(previous.get("risk_level") or "low"), 1)
    current_rank = RISK_RANK.get(str(current.get("risk_level") or "low"), 1)
    if current_rank > previous_rank:
        return "up"
    if current_rank < previous_rank:
        return "down"
    return "flat"


def _strategy_family(record: dict[str, Any]) -> str:
    route = str(record.get("orchestration_route") or "")
    dynamic_action = str(record.get("dynamic_action") or "")
    adjustment_action = str(record.get("adjustment_loop_action") or "")
    if route in {"safety_first", "human_support_linkage"} or adjustment_action == "safety_first":
        return "human_linkage"
    if route == "repair_conversation" or dynamic_action == "repair_reply_style":
        return "trust_repair"
    if route == "stabilize_and_reduce_load" or dynamic_action == "soften_and_stabilize":
        return "stabilization"
    if route == "boundary_respecting_support":
        return "boundary_support"
    if route == "explore_and_clarify":
        return "gentle_exploration"
    return "supportive_continuity"


def _loop_score(record: dict[str, Any], entropy_delta: int | None, risk_shift: str) -> int:
    score = 55
    if entropy_delta is not None:
        if entropy_delta <= -12:
            score += 24
        elif entropy_delta <= -8:
            score += 18
        elif entropy_delta <= -3:
            score += 9
        elif entropy_delta >= 12:
            score -= 24
        elif entropy_delta >= 8:
            score -= 17
        elif entropy_delta >= 4:
            score -= 9
    if risk_shift == "down":
        score += 14
    elif risk_shift == "up":
        score -= 20
    if record.get("referral_should_refer") or record.get("risk_level") in {"high", "critical"}:
        score -= 8
    if record.get("feedback_adaptation_mode") == "repair_next_turn":
        score -= 12
    if record.get("dynamic_action") in {"soften_and_stabilize", "repair_reply_style"}:
        score += 3
    return max(0, min(100, score))


def _point_status(
    record: dict[str, Any],
    entropy_delta: int | None,
    risk_shift: str,
    loop_score: int,
) -> str:
    if record.get("risk_level") == "critical":
        return "crisis_priority"
    if record.get("risk_level") == "high" or record.get("referral_should_refer"):
        return "human_followup_needed"
    if record.get("feedback_adaptation_mode") == "repair_next_turn":
        return "strategy_repair_needed"
    if risk_shift == "up" or (entropy_delta is not None and entropy_delta >= 8):
        return "not_reducing"
    if entropy_delta is not None and entropy_delta <= -8 and risk_shift != "up":
        return "reducing"
    if loop_score >= 68:
        return "working"
    if loop_score <= 40:
        return "weak"
    return "holding"


def _point_next_action(status: str, record: dict[str, Any]) -> str:
    if status == "crisis_priority":
        return "activate_safety_protocol"
    if status == "human_followup_needed":
        return "recommend_human_followup"
    if status == "strategy_repair_needed":
        return "repair_reply_style"
    if status == "not_reducing":
        return "switch_or_reduce_intervention_pressure"
    if status == "reducing":
        return "maintain_and_consolidate"
    if status == "weak":
        return "inspect_bad_case_or_collect_feedback"
    if record.get("adjustment_loop_action"):
        return str(record["adjustment_loop_action"])
    return "continue_observation"


def _overall_status(
    *,
    latest_status: str,
    average_score: float | None,
    feedback_summary: dict[str, Any],
    reply_quality_summary: dict[str, Any],
    strategy_version: dict[str, Any],
    intervention_effectiveness: dict[str, Any],
) -> str:
    effectiveness_summary = intervention_effectiveness.get("summary") or {}
    effectiveness_status = str(effectiveness_summary.get("overall_status") or "")
    if latest_status in {"crisis_priority", "human_followup_needed", "strategy_repair_needed", "not_reducing"}:
        return latest_status
    if effectiveness_status in {"crisis_priority", "needs_human_followup"}:
        return "human_followup_needed"
    if _feedback_signal(feedback_summary) == "negative" or _reply_quality_signal(reply_quality_summary) == "poor":
        return "strategy_repair_needed"
    if strategy_version.get("decision") in {"repair", "revise"}:
        return "strategy_switching"
    if latest_status in {"reducing", "working"} and average_score is not None and average_score >= 62:
        return "reducing"
    if average_score is not None and average_score <= 42:
        return "not_reducing"
    return latest_status or "insufficient_data"


def _recommended_backend_action(
    *,
    overall_status: str,
    strategy_version: dict[str, Any],
    latest: dict[str, Any],
) -> str:
    if overall_status == "crisis_priority":
        return "activate_safety_protocol"
    if overall_status == "human_followup_needed":
        return "recommend_human_followup"
    if overall_status == "strategy_repair_needed":
        return "repair_strategy_and_collect_bad_case"
    if overall_status == "not_reducing":
        return "switch_strategy_or_lower_pressure"
    if overall_status == "strategy_switching":
        return f"apply_strategy_family:{strategy_version.get('target_strategy_family') or 'unknown'}"
    if overall_status == "reducing":
        return "maintain_strategy_and_track_decay"
    return str(latest.get("next_action") or "continue_observation")


def _feedback_signal(feedback_summary: dict[str, Any]) -> str:
    positive = int(feedback_summary.get("positive_count") or 0)
    negative = int(feedback_summary.get("negative_count") or 0)
    if positive == 0 and negative == 0:
        return "none"
    if negative > positive:
        return "negative"
    if positive > negative:
        return "positive"
    return "mixed"


def _reply_quality_signal(reply_quality_summary: dict[str, Any]) -> str:
    total = int(reply_quality_summary.get("total_replies") or 0)
    needs_review = int(reply_quality_summary.get("needs_review") or 0)
    if total <= 0:
        return "unknown"
    ratio = needs_review / total
    if ratio >= 0.4:
        return "poor"
    if ratio >= 0.15:
        return "mixed"
    return "clean"


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
