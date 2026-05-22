from __future__ import annotations

from typing import Any

from .schemas import GoalAttainmentEvaluation


RISK_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}


def build_goal_attainment_evaluation(
    *,
    previous_record: dict[str, Any] | None,
    current_record: dict[str, Any] | None,
) -> GoalAttainmentEvaluation | None:
    if not previous_record or not current_record:
        return None
    previous_goal = previous_record.get("reduction_goal") or {}
    if not previous_goal:
        return None

    entropy_delta = _entropy_delta(previous_record, current_record)
    risk_shift = _risk_shift(previous_record, current_record)
    feedback_signal = _feedback_signal(current_record)
    status = _status(entropy_delta, risk_shift, feedback_signal)
    completion_score = _completion_score(status, entropy_delta, risk_shift, feedback_signal)

    return GoalAttainmentEvaluation(
        evaluation_id=f"goal_eval:{previous_record.get('response_id')}:{current_record.get('response_id')}",
        status=status,
        completion_score=completion_score,
        entropy_delta=entropy_delta,
        risk_shift=risk_shift,
        driver=previous_goal.get("active_driver"),
        goal_text=previous_goal.get("reduction_goal"),
        interpretation=_interpretation(status, entropy_delta, risk_shift, previous_goal),
        next_adjustment=_next_adjustment(status, previous_goal),
        evidence={
            "previous_response_id": previous_record.get("response_id"),
            "current_response_id": current_record.get("response_id"),
            "previous_entropy": previous_record.get("entropy_score"),
            "current_entropy": current_record.get("entropy_score"),
            "previous_risk_level": previous_record.get("risk_level"),
            "current_risk_level": current_record.get("risk_level"),
            "target_entropy_delta": previous_goal.get("target_entropy_delta"),
            "previous_micro_intervention": previous_goal.get("micro_intervention"),
            "feedback_signal": feedback_signal,
        },
    )


def build_goal_attainment_timeline(records: list[dict[str, Any]], *, limit: int = 12) -> list[dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []
    pairs = zip(records, records[1:])
    for previous_record, current_record in pairs:
        evaluation = build_goal_attainment_evaluation(
            previous_record=previous_record,
            current_record=current_record,
        )
        if evaluation is not None:
            evaluations.append({
                "evaluation_id": evaluation.evaluation_id,
                "status": evaluation.status,
                "completion_score": evaluation.completion_score,
                "entropy_delta": evaluation.entropy_delta,
                "risk_shift": evaluation.risk_shift,
                "driver": evaluation.driver,
                "goal_text": evaluation.goal_text,
                "interpretation": evaluation.interpretation,
                "next_adjustment": evaluation.next_adjustment,
                "evidence": evaluation.evidence,
            })
    return evaluations[-limit:]


def summarize_goal_attainment(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    if not evaluations:
        return {
            "total_evaluated": 0,
            "average_completion_score": None,
            "status_counts": {},
            "latest_status": None,
            "latest_next_adjustment": None,
        }
    status_counts: dict[str, int] = {}
    total_score = 0
    for item in evaluations:
        status = str(item.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        total_score += int(item.get("completion_score") or 0)
    latest = evaluations[-1]
    return {
        "total_evaluated": len(evaluations),
        "average_completion_score": round(total_score / len(evaluations), 1),
        "status_counts": status_counts,
        "latest_status": latest.get("status"),
        "latest_next_adjustment": latest.get("next_adjustment"),
    }


def _entropy_delta(previous_record: dict[str, Any], current_record: dict[str, Any]) -> int | None:
    try:
        return int(current_record.get("entropy_score")) - int(previous_record.get("entropy_score"))
    except (TypeError, ValueError):
        return None


def _risk_shift(previous_record: dict[str, Any], current_record: dict[str, Any]) -> str:
    previous = RISK_RANK.get(str(previous_record.get("risk_level") or "low"), 1)
    current = RISK_RANK.get(str(current_record.get("risk_level") or "low"), 1)
    if current > previous:
        return "worse"
    if current < previous:
        return "improved"
    return "stable"


def _feedback_signal(record: dict[str, Any]) -> str:
    feedback = record.get("feedback") or {}
    try:
        helpful_score = int(feedback.get("helpful_score"))
    except (TypeError, ValueError):
        return "none"
    if helpful_score > 0:
        return "positive"
    if helpful_score < 0:
        return "negative"
    return "neutral"


def _status(entropy_delta: int | None, risk_shift: str, feedback_signal: str) -> str:
    if risk_shift == "worse" or (entropy_delta is not None and entropy_delta >= 10):
        return "worsened"
    if feedback_signal == "negative":
        return "ineffective"
    if risk_shift == "improved" or (entropy_delta is not None and entropy_delta <= -8):
        return "achieved"
    if entropy_delta is not None and entropy_delta <= -3:
        return "partially_achieved"
    if entropy_delta is not None and entropy_delta >= 4:
        return "not_achieved"
    return "holding"


def _completion_score(status: str, entropy_delta: int | None, risk_shift: str, feedback_signal: str) -> int:
    base = {
        "achieved": 85,
        "partially_achieved": 65,
        "holding": 50,
        "not_achieved": 35,
        "ineffective": 25,
        "worsened": 10,
    }.get(status, 40)
    if feedback_signal == "positive":
        base += 5
    if feedback_signal == "negative":
        base -= 10
    if risk_shift == "improved":
        base += 5
    if risk_shift == "worse":
        base -= 10
    if entropy_delta is not None and entropy_delta <= -12:
        base += 5
    return max(0, min(100, base))


def _interpretation(
    status: str,
    entropy_delta: int | None,
    risk_shift: str,
    previous_goal: dict[str, Any],
) -> str:
    driver = previous_goal.get("active_driver") or "unknown_driver"
    if status == "achieved":
        return f"上一轮围绕 {driver} 的熵减目标有效，熵值或风险出现下降。"
    if status == "partially_achieved":
        return f"上一轮围绕 {driver} 的目标有一定效果，但仍需继续观察。"
    if status == "holding":
        return f"上一轮围绕 {driver} 的目标暂时维持住状态，尚未形成明显下降。"
    if status == "worsened":
        return f"上一轮目标未能阻止恶化，熵值变化为 {entropy_delta}，风险变化为 {risk_shift}。"
    if status == "ineffective":
        return f"上一轮目标可能未被用户接受，需要降低压力或换一种支持方式。"
    return f"上一轮围绕 {driver} 的目标尚未达成，需要调整干预。"


def _next_adjustment(status: str, previous_goal: dict[str, Any]) -> str:
    driver = previous_goal.get("active_driver") or ""
    if status == "achieved":
        return "maintain_and_consolidate_goal"
    if status == "partially_achieved":
        return "continue_same_goal_with_smaller_step"
    if status == "holding":
        return "keep_goal_but_reduce_question_pressure"
    if status == "worsened":
        return "escalate_goal_priority_and_check_human_support"
    if status == "ineffective":
        return "repair_style_before_new_advice"
    if driver == "privacy_boundary_tension":
        return "return_to_boundary_reassurance"
    return "reselect_goal_next_turn"
