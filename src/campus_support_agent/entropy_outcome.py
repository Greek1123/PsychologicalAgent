from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .schemas import CarePathwayDecision, EntropyReductionOutcome


RISK_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}


def build_entropy_reduction_outcome(
    *,
    session_id: str,
    records: list[dict[str, Any]],
    entropy_trace: list[dict[str, Any]],
    feedback_summary: dict[str, Any] | None = None,
    care_pathway: CarePathwayDecision | dict[str, Any] | None = None,
) -> EntropyReductionOutcome:
    """Evaluate whether recent support is reducing entropy over the session."""

    feedback_summary = feedback_summary or {}
    pathway = _to_dict(care_pathway)
    scores = _entropy_scores(records, entropy_trace)
    entropy_delta = scores[-1] - scores[0] if len(scores) >= 2 else None
    risk_shift = _risk_shift(records)
    feedback_signal = _feedback_signal(feedback_summary)
    pathway_signal = _pathway_signal(pathway)
    status = _status(
        scores=scores,
        entropy_delta=entropy_delta,
        risk_shift=risk_shift,
        feedback_signal=feedback_signal,
        pathway_signal=pathway_signal,
        pathway=pathway,
    )
    effectiveness_score = _effectiveness_score(
        entropy_delta=entropy_delta,
        risk_shift=risk_shift,
        feedback_signal=feedback_signal,
        pathway_signal=pathway_signal,
    )

    return EntropyReductionOutcome(
        outcome_id=f"{session_id}:entropy_outcome:{len(records)}",
        session_id=session_id,
        status=status,
        effectiveness_score=effectiveness_score,
        entropy_delta=entropy_delta,
        risk_shift=risk_shift,
        feedback_signal=feedback_signal,
        pathway_signal=pathway_signal,
        summary=_summary(status),
        next_action=_next_action(status),
        evidence={
            "entropy_points": len(scores),
            "first_entropy_score": scores[0] if scores else None,
            "latest_entropy_score": scores[-1] if scores else None,
            "min_entropy_score": min(scores) if scores else None,
            "max_entropy_score": max(scores) if scores else None,
            "record_count": len(records),
            "feedback_summary": feedback_summary,
            "care_pathway_route": pathway.get("route"),
            "care_pathway_priority": pathway.get("priority"),
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


def _risk_shift(records: list[dict[str, Any]]) -> str:
    risks = [str(record.get("risk_level") or "low") for record in records if record.get("risk_level")]
    if len(risks) < 2:
        return "baseline"
    first = RISK_RANK.get(risks[0], 1)
    latest = RISK_RANK.get(risks[-1], 1)
    if latest > first:
        return "up"
    if latest < first:
        return "down"
    return "flat"


def _feedback_signal(feedback_summary: dict[str, Any]) -> str:
    positive = int(feedback_summary.get("positive_count") or 0)
    negative = int(feedback_summary.get("negative_count") or 0)
    if positive == 0 and negative == 0:
        return "none"
    if positive > 0 and negative > 0:
        return "mixed"
    if positive > negative:
        return "positive"
    if negative > positive:
        return "negative"
    return "mixed"


def _pathway_signal(pathway: dict[str, Any]) -> str:
    route = pathway.get("route")
    priority = pathway.get("priority")
    if route == "urgent_safety" or priority == "critical":
        return "crisis"
    if route == "human_followup_recommended" or priority == "high":
        return "needs_human_support"
    if route in {"monitor_next_turn", "repair_reply_style"} or priority == "medium":
        return "active_adjustment"
    if route == "continue_observation" or priority == "low":
        return "stable_support"
    return "unknown"


def _status(
    *,
    scores: list[int],
    entropy_delta: int | None,
    risk_shift: str,
    feedback_signal: str,
    pathway_signal: str,
    pathway: dict[str, Any],
) -> str:
    if pathway_signal == "crisis":
        return "crisis_priority"
    if pathway_signal == "needs_human_support":
        return "needs_human_followup"
    if len(scores) < 2 and feedback_signal == "none":
        return "insufficient_data"
    if feedback_signal == "negative":
        return "needs_strategy_repair"
    if risk_shift == "up" or (entropy_delta is not None and entropy_delta >= 10):
        return "deteriorating"
    if entropy_delta is not None and entropy_delta <= -8 and risk_shift != "up":
        return "improving"
    if feedback_signal == "positive" and (entropy_delta is None or entropy_delta <= 5):
        return "stable_helpful"
    if pathway.get("route") == "monitor_next_turn":
        return "watching"
    return "stable_observe"


def _effectiveness_score(
    *,
    entropy_delta: int | None,
    risk_shift: str,
    feedback_signal: str,
    pathway_signal: str,
) -> int:
    score = 50
    if entropy_delta is not None:
        if entropy_delta <= -15:
            score += 25
        elif entropy_delta <= -8:
            score += 18
        elif entropy_delta <= -3:
            score += 8
        elif entropy_delta >= 15:
            score -= 25
        elif entropy_delta >= 8:
            score -= 15
        elif entropy_delta >= 3:
            score -= 8

    if risk_shift == "down":
        score += 15
    elif risk_shift == "up":
        score -= 20

    if feedback_signal == "positive":
        score += 10
    elif feedback_signal == "negative":
        score -= 15
    elif feedback_signal == "mixed":
        score -= 3

    if pathway_signal == "needs_human_support":
        score -= 10
    elif pathway_signal == "crisis":
        score -= 20

    return max(0, min(100, score))


def _summary(status: str) -> str:
    mapping = {
        "insufficient_data": "More interaction data is needed before evaluating entropy reduction.",
        "crisis_priority": "Safety handling has priority over ordinary entropy-reduction evaluation.",
        "needs_strategy_repair": "User feedback suggests the support strategy needs repair.",
        "deteriorating": "Entropy or risk is rising; the current intervention is not yet stabilizing the session.",
        "needs_human_followup": "The session needs stronger offline or human support linkage.",
        "improving": "Entropy is moving downward without increased risk.",
        "stable_helpful": "The session is stable and user feedback is positive.",
        "watching": "The session is not clearly worse, but still needs next-turn monitoring.",
        "stable_observe": "The session is stable enough for continued observation.",
    }
    return mapping.get(status, "Entropy reduction status is available for observation.")


def _next_action(status: str) -> str:
    mapping = {
        "insufficient_data": "collect_next_entropy_point",
        "crisis_priority": "activate_safety_protocol",
        "needs_strategy_repair": "repair_next_reply_style",
        "deteriorating": "reduce_pressure_and_monitor_next_turn",
        "needs_human_followup": "recommend_human_followup",
        "improving": "maintain_working_strategy",
        "stable_helpful": "preserve_current_strategy",
        "watching": "compare_next_turn_entropy",
        "stable_observe": "continue_observation",
    }
    return mapping.get(status, "continue_observation")


def _to_dict(value: CarePathwayDecision | dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    return asdict(value)
