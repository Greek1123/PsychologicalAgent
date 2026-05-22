from __future__ import annotations

from collections import Counter
from typing import Any


RISK_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}


def build_intervention_effectiveness_report(
    *,
    session_id: str | None,
    records: list[dict[str, Any]],
    feedback_summary: dict[str, Any] | None = None,
    reply_quality_summary: dict[str, Any] | None = None,
    care_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    feedback_summary = feedback_summary or {}
    reply_quality_summary = reply_quality_summary or {}
    care_plan = care_plan or {}
    timeline = build_intervention_effectiveness_timeline(records)
    latest = timeline[-1] if timeline else None
    summary = summarize_intervention_effectiveness(
        timeline,
        feedback_summary=feedback_summary,
        reply_quality_summary=reply_quality_summary,
        care_plan=care_plan,
    )
    return {
        "session_id": session_id,
        "total_records": len(records),
        "summary": summary,
        "latest_effectiveness": latest,
        "timeline": timeline,
    }


def build_intervention_effectiveness_timeline(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        previous = records[index - 1] if index > 0 else None
        entropy_delta = _entropy_delta(previous, record)
        risk_shift = _risk_shift(previous, record)
        quality_penalty = _quality_penalty(record)
        status = _status(record, entropy_delta, risk_shift, quality_penalty)
        score = _effectiveness_score(record, entropy_delta, risk_shift, quality_penalty, status)
        timeline.append(
            {
                "response_id": record.get("response_id"),
                "created_at": record.get("created_at"),
                "input_preview": str(record.get("input_text") or "")[:160],
                "status": status,
                "effectiveness_score": score,
                "entropy_score": _safe_int(record.get("entropy_score")),
                "entropy_delta": entropy_delta,
                "risk_level": record.get("risk_level"),
                "risk_shift": risk_shift,
                "reply_quality_penalty": quality_penalty,
                "dynamic_action": record.get("dynamic_action"),
                "orchestration_route": record.get("orchestration_route") or "legacy_or_unclassified",
                "adjustment_loop_action": record.get("adjustment_loop_action"),
                "recommended_next_action": _recommended_next_action(status, record),
                "evidence": _evidence(record, entropy_delta, risk_shift, quality_penalty),
            }
        )
    return timeline


def summarize_intervention_effectiveness(
    timeline: list[dict[str, Any]],
    *,
    feedback_summary: dict[str, Any] | None = None,
    reply_quality_summary: dict[str, Any] | None = None,
    care_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    feedback_summary = feedback_summary or {}
    reply_quality_summary = reply_quality_summary or {}
    care_plan = care_plan or {}
    status_counts: Counter[str] = Counter(str(item.get("status") or "unknown") for item in timeline)
    scores = [int(item.get("effectiveness_score") or 0) for item in timeline]
    latest = timeline[-1] if timeline else {}
    overall_status = _overall_status(
        latest_status=str(latest.get("status") or "insufficient_data"),
        average_score=round(sum(scores) / len(scores), 1) if scores else None,
        feedback_summary=feedback_summary,
        reply_quality_summary=reply_quality_summary,
        care_plan=care_plan,
    )
    return {
        "overall_status": overall_status,
        "average_effectiveness_score": round(sum(scores) / len(scores), 1) if scores else None,
        "latest_effectiveness_score": latest.get("effectiveness_score"),
        "latest_status": latest.get("status"),
        "latest_recommended_next_action": latest.get("recommended_next_action"),
        "status_counts": dict(status_counts),
        "feedback_signal": _feedback_signal(feedback_summary),
        "reply_quality_signal": _reply_quality_signal(reply_quality_summary),
        "care_phase": care_plan.get("care_phase"),
        "next_backend_focus": _next_backend_focus(overall_status, latest),
    }


def build_intervention_effectiveness_overview(
    session_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    care_phase_counts: Counter[str] = Counter()
    scores: list[int] = []
    watch_sessions: list[dict[str, Any]] = []
    for report in session_reports:
        summary = report.get("summary") or {}
        status = str(summary.get("overall_status") or "unknown")
        status_counts[status] += 1
        care_phase = str(summary.get("care_phase") or "unknown")
        care_phase_counts[care_phase] += 1
        score = summary.get("latest_effectiveness_score")
        if isinstance(score, int):
            scores.append(score)
        if status in {"deteriorating", "needs_repair", "needs_human_followup", "crisis_priority"}:
            watch_sessions.append(
                {
                    "session_id": report.get("session_id"),
                    "overall_status": status,
                    "latest_effectiveness_score": score,
                    "next_action": summary.get("latest_recommended_next_action"),
                    "backend_focus": summary.get("next_backend_focus"),
                }
            )
    return {
        "total_sessions": len(session_reports),
        "average_latest_effectiveness_score": round(sum(scores) / len(scores), 1) if scores else None,
        "status_counts": dict(status_counts),
        "care_phase_counts": dict(care_phase_counts),
        "watch_sessions": watch_sessions[-30:],
    }


def _status(
    record: dict[str, Any],
    entropy_delta: int | None,
    risk_shift: str,
    quality_penalty: int,
) -> str:
    risk_level = str(record.get("risk_level") or "low")
    if risk_level == "critical":
        return "crisis_priority"
    if risk_level == "high" or record.get("referral_should_refer"):
        return "needs_human_followup"
    if quality_penalty >= 35:
        return "needs_repair"
    if risk_shift == "up" or (entropy_delta is not None and entropy_delta >= 10):
        return "deteriorating"
    if entropy_delta is not None and entropy_delta <= -8 and risk_shift != "up":
        return "improving"
    if entropy_delta is None:
        return "baseline"
    if -7 <= entropy_delta <= 4:
        return "stable"
    return "watching"


def _effectiveness_score(
    record: dict[str, Any],
    entropy_delta: int | None,
    risk_shift: str,
    quality_penalty: int,
    status: str,
) -> int:
    score = 55
    if entropy_delta is not None:
        if entropy_delta <= -12:
            score += 22
        elif entropy_delta <= -8:
            score += 16
        elif entropy_delta <= -3:
            score += 8
        elif entropy_delta >= 12:
            score -= 22
        elif entropy_delta >= 8:
            score -= 14
        elif entropy_delta >= 4:
            score -= 8
    if risk_shift == "down":
        score += 12
    elif risk_shift == "up":
        score -= 18
    score -= quality_penalty
    if status == "needs_human_followup":
        score -= 10
    elif status == "crisis_priority":
        score -= 25
    if record.get("dynamic_action") in {"soften_and_stabilize", "repair_reply_style"}:
        score += 3
    return max(0, min(100, score))


def _recommended_next_action(status: str, record: dict[str, Any]) -> str:
    if status == "crisis_priority":
        return "activate_safety_protocol"
    if status == "needs_human_followup":
        return "recommend_human_followup"
    if status == "needs_repair":
        return "repair_reply_style_before_new_advice"
    if status == "deteriorating":
        return "reduce_pressure_and_check_next_turn"
    if status == "improving":
        return "maintain_current_strategy"
    if status == "baseline":
        return "collect_next_entropy_point"
    if record.get("adjustment_loop_action"):
        return str(record["adjustment_loop_action"])
    return "continue_observation"


def _overall_status(
    *,
    latest_status: str,
    average_score: float | None,
    feedback_summary: dict[str, Any],
    reply_quality_summary: dict[str, Any],
    care_plan: dict[str, Any],
) -> str:
    if latest_status in {"crisis_priority", "needs_human_followup", "deteriorating", "needs_repair"}:
        return latest_status
    if str(care_plan.get("care_phase") or "") in {"safety", "human_followup"}:
        return "needs_human_followup"
    if _feedback_signal(feedback_summary) == "negative":
        return "needs_repair"
    if _reply_quality_signal(reply_quality_summary) == "poor":
        return "needs_repair"
    if average_score is not None and average_score >= 68:
        return "effective"
    if average_score is not None and average_score <= 42:
        return "needs_repair"
    return latest_status or "insufficient_data"


def _next_backend_focus(overall_status: str, latest: dict[str, Any]) -> str:
    mapping = {
        "crisis_priority": "safety_and_referral_first",
        "needs_human_followup": "human_support_linkage",
        "deteriorating": "reduce_question_pressure_and_monitor_entropy",
        "needs_repair": "repair_reply_quality_and_collect_bad_case",
        "effective": "maintain_strategy_and_track_decay",
        "improving": "consolidate_small_success",
        "stable": "continue_low_pressure_support",
        "baseline": "collect_more_context",
        "watching": "compare_next_turn_entropy",
    }
    return mapping.get(overall_status, str(latest.get("recommended_next_action") or "continue_observation"))


def _evidence(
    record: dict[str, Any],
    entropy_delta: int | None,
    risk_shift: str,
    quality_penalty: int,
) -> dict[str, Any]:
    return {
        "entropy_delta": entropy_delta,
        "risk_shift": risk_shift,
        "quality_penalty": quality_penalty,
        "risk_score": record.get("risk_score"),
        "dynamic_action": record.get("dynamic_action"),
        "feedback_adaptation_mode": record.get("feedback_adaptation_mode"),
        "local_policy_name": record.get("local_policy_name"),
        "referral_urgency": record.get("referral_urgency"),
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


def _quality_penalty(record: dict[str, Any]) -> int:
    response = record.get("response") or {}
    flags = response.get("system_flags") or {}
    if flags.get("final_reply_guardrail_applied"):
        return 20
    if flags.get("bad_case_candidate"):
        return 15
    return 0


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
    if total == 0:
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
