from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .schemas import EntropyAdjustmentLoop


def build_entropy_adjustment_loop(
    *,
    session_id: str,
    records: list[dict[str, Any]] | None = None,
    feedback_summary: dict[str, Any] | None = None,
    recent_feedback: list[dict[str, Any]] | None = None,
    session_continuity: dict[str, Any] | None = None,
    strategy_reselection: dict[str, Any] | None = None,
    audit_summary: dict[str, Any] | None = None,
) -> EntropyAdjustmentLoop:
    records = records or []
    latest = records[-1] if records else {}
    feedback_summary = feedback_summary or {}
    recent_feedback = recent_feedback or []
    session_continuity = session_continuity or {}
    strategy_reselection = strategy_reselection or {}
    audit_summary = audit_summary or {}

    entropy_score = _safe_int(latest.get("entropy_score"))
    risk_level = str(latest.get("risk_level") or "low")
    feedback_mode = str(latest.get("feedback_adaptation_mode") or "")
    dialogue_stage = str(session_continuity.get("dialogue_stage") or "")
    recent_tags = _recent_tags(recent_feedback)
    latest_route = str(audit_summary.get("latest_decision_route") or latest.get("orchestration_route") or "")
    average_helpful_score = feedback_summary.get("average_helpful_score")
    negative_count = int(feedback_summary.get("negative_count") or 0)

    action = "continue_current_strategy"
    priority = "low"
    next_reply_mode = "supportive"
    question_policy = "one_optional_question"
    memory_policy = "use_recent_context"
    human_followup_policy = "none"
    constraints: list[str] = ["Do not expose entropy or backend labels to the user."]
    preferred_moves: list[str] = ["Reflect the latest user meaning before giving advice."]
    reasons: list[str] = []

    if risk_level == "critical" or dialogue_stage == "safety_priority":
        action = "safety_first"
        priority = "critical"
        next_reply_mode = "safety_grounding"
        question_policy = "ask_only_safety_check"
        memory_policy = "prioritize_current_risk"
        human_followup_policy = "urgent"
        constraints.extend(["Avoid long analysis.", "Give concrete emergency or trusted-person options."])
        preferred_moves.extend(["Validate distress briefly.", "Move to immediate safety and real-world support."])
        reasons.append("critical_risk_or_safety_stage")
    elif risk_level == "high" or latest_route in {"human_support_recommended", "human_support_linkage"}:
        action = "human_followup_watch"
        priority = "high"
        next_reply_mode = "warm_human_linkage"
        question_policy = "one_low_pressure_question"
        memory_policy = "use_recent_context"
        human_followup_policy = "recommended"
        constraints.extend(["Do not sound alarming.", "Explain help-seeking as an option, not a command."])
        preferred_moves.extend(["Normalize seeking support.", "Offer one reachable campus support path."])
        reasons.append("high_risk_or_human_support_route")
    elif strategy_reselection.get("should_reselect"):
        action = "switch_strategy"
        priority = str(strategy_reselection.get("priority") or "medium")
        next_reply_mode = "repair_then_support"
        question_policy = "one_optional_question"
        memory_policy = "avoid_repeating_failed_move"
        constraints.extend(strategy_reselection.get("constraints") or [])
        preferred_moves.extend(["Acknowledge the previous miss.", "Use a different support move."])
        reasons.append(f"strategy_reselection:{strategy_reselection.get('trigger')}")
    elif feedback_mode == "repair_next_turn" or negative_count >= 2 or _score_below_zero(average_helpful_score):
        action = "repair_reply_style"
        priority = "medium"
        next_reply_mode = "repair_conversation"
        question_policy = "low_pressure_or_no_question"
        memory_policy = "avoid_repeating_failed_move"
        constraints.extend(["Do not defend the previous reply.", "Do not ask for many details."])
        preferred_moves.extend(["Say you may not have caught the point.", "Return control to the user."])
        reasons.append("negative_feedback_or_repair_mode")
    elif "repetitive" in recent_tags or "template_reply" in recent_tags or "robotic" in recent_tags:
        action = "increase_contextuality"
        priority = "medium"
        next_reply_mode = "contextual_support"
        question_policy = "one_contextual_question"
        memory_policy = "quote_or_paraphrase_latest_context"
        constraints.extend(["Do not reuse the same opening.", "Do not produce generic comfort."])
        preferred_moves.extend(["Mention the concrete scene the user gave.", "Offer one tailored next step."])
        reasons.append("recent_repetition_feedback")
    elif entropy_score is not None and entropy_score >= 70:
        action = "stabilize_before_problem_solving"
        priority = "medium"
        next_reply_mode = "stabilize"
        question_policy = "one_grounding_choice"
        memory_policy = "use_recent_context"
        constraints.extend(["Do not overload with plans.", "Give only one small action."])
        preferred_moves.extend(["Name the pressure simply.", "Offer one immediate stabilizing step."])
        reasons.append("high_entropy_score")
    elif dialogue_stage in {"boundary_building", "early_trust"}:
        action = "build_trust"
        priority = "medium"
        next_reply_mode = "boundary_respecting"
        question_policy = "no_forced_disclosure"
        memory_policy = "respect_privacy_boundary"
        constraints.extend(["Do not push for details.", "Reassure privacy and choice first."])
        preferred_moves.extend(["State that the user can choose what to share.", "Offer companionship before advice."])
        reasons.append(f"dialogue_stage:{dialogue_stage}")
    else:
        reasons.append("no_strong_adjustment_signal")

    loop = EntropyAdjustmentLoop(
        loop_id=f"{session_id}:adjustment_loop:{len(records)}",
        loop_action=action,
        priority=priority,
        next_reply_mode=next_reply_mode,
        question_policy=question_policy,
        memory_policy=memory_policy,
        human_followup_policy=human_followup_policy,
        rationale=";".join(reasons),
        constraints=_dedupe(constraints),
        preferred_moves=_dedupe(preferred_moves),
        evidence={
            "latest_response_id": latest.get("response_id"),
            "latest_entropy_score": entropy_score,
            "latest_risk_level": risk_level,
            "dialogue_stage": dialogue_stage,
            "latest_decision_route": latest_route,
            "feedback_mode": feedback_mode,
            "negative_feedback_count": negative_count,
            "recent_feedback_tags": sorted(recent_tags),
            "strategy_reselection_trigger": strategy_reselection.get("trigger"),
        },
    )
    return loop


def enrich_student_context_with_adjustment_loop(
    student_context: dict[str, Any],
    adjustment_loop: EntropyAdjustmentLoop | dict[str, Any] | None,
) -> dict[str, Any]:
    if adjustment_loop is None:
        return dict(student_context or {})
    loop = asdict(adjustment_loop) if isinstance(adjustment_loop, EntropyAdjustmentLoop) else dict(adjustment_loop)
    enriched = dict(student_context or {})
    enriched["adjustment_loop"] = {
        "loop_action": loop.get("loop_action"),
        "priority": loop.get("priority"),
        "next_reply_mode": loop.get("next_reply_mode"),
        "question_policy": loop.get("question_policy"),
        "memory_policy": loop.get("memory_policy"),
        "human_followup_policy": loop.get("human_followup_policy"),
        "constraints": loop.get("constraints") or [],
        "preferred_moves": loop.get("preferred_moves") or [],
        "instruction": (
            "Use this hidden loop state to adjust the next reply. Do not mention entropy, loop_action, "
            "risk labels, backend routes, or internal scoring to the user."
        ),
    }
    return enriched


def _recent_tags(recent_feedback: list[dict[str, Any]]) -> set[str]:
    tags: set[str] = set()
    for item in recent_feedback[-5:]:
        tags.update(str(tag).strip() for tag in item.get("tags", []) if str(tag).strip())
    return tags


def _score_below_zero(value: Any) -> bool:
    try:
        return float(value) < 0
    except (TypeError, ValueError):
        return False


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item and item not in result:
            result.append(item)
    return result
