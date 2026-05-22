from __future__ import annotations

from typing import Any

from .schemas import CarePathwayDecision, EntropyTrendWarning, SessionCarePlan


def enrich_student_context_with_care_plan(
    student_context: dict[str, Any] | None,
    care_plan: SessionCarePlan | dict[str, Any] | None,
) -> dict[str, Any]:
    enriched = dict(student_context or {})
    plan = _to_dict(care_plan)
    if not plan:
        return enriched
    enriched["session_care_plan"] = {
        "care_phase": plan.get("care_phase"),
        "priority": plan.get("priority"),
        "primary_goal": plan.get("primary_goal"),
        "user_visible_focus": plan.get("user_visible_focus"),
        "next_actions": plan.get("next_actions") or [],
        "avoid_actions": plan.get("avoid_actions") or [],
        "review_window_hours": plan.get("review_window_hours"),
    }
    return enriched


def build_session_care_plan(
    *,
    session_id: str,
    conversation_memory: dict[str, Any] | None,
    longitudinal_profile: dict[str, Any] | None,
    care_pathway: CarePathwayDecision | dict[str, Any] | None,
    trend_warning: EntropyTrendWarning | dict[str, Any] | None,
    adjustment_loop: dict[str, Any] | None,
    entropy_outcome: dict[str, Any] | None,
    latest_record: dict[str, Any] | None = None,
) -> SessionCarePlan:
    memory = conversation_memory or {}
    profile = longitudinal_profile or {}
    pathway = _to_dict(care_pathway)
    warning = _to_dict(trend_warning)
    loop = adjustment_loop or {}
    outcome = entropy_outcome or {}
    latest_record = latest_record or {}

    priority = _priority(pathway, warning, profile)
    care_phase = _care_phase(pathway, warning, outcome)
    review_window = _review_window(pathway, warning, profile)
    latest_response_id = str(latest_record.get("response_id") or "none")

    return SessionCarePlan(
        plan_id=f"{session_id}:care_plan:{care_phase}:{latest_response_id}",
        session_id=session_id,
        care_phase=care_phase,
        priority=priority,
        review_window_hours=review_window,
        primary_goal=_primary_goal(care_phase, memory, warning),
        user_visible_focus=_user_visible_focus(care_phase, memory, loop),
        backend_focus=_backend_focus(care_phase, pathway, warning, outcome),
        next_actions=_next_actions(care_phase, pathway, warning, loop, memory),
        avoid_actions=_avoid_actions(care_phase, memory, loop),
        success_indicators=_success_indicators(care_phase, warning),
        escalation_conditions=_escalation_conditions(care_phase, warning),
        evidence={
            "memory_topics": memory.get("active_topics") or [],
            "memory_boundaries": memory.get("user_boundaries") or [],
            "preferred_next_move": memory.get("preferred_next_move"),
            "pathway_route": pathway.get("route"),
            "pathway_priority": pathway.get("priority"),
            "trend_warning_level": warning.get("level"),
            "trend_state": warning.get("trend_state"),
            "loop_action": loop.get("loop_action"),
            "entropy_outcome_status": outcome.get("status"),
            "latest_response_id": latest_record.get("response_id"),
            "latest_entropy_score": latest_record.get("entropy_score"),
            "latest_risk_level": latest_record.get("risk_level"),
            "longitudinal_care_level": profile.get("recommended_care_level"),
        },
    )


def _to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dataclass_fields__"):
        return {field: getattr(value, field) for field in value.__dataclass_fields__}
    return {}


def _priority(pathway: dict[str, Any], warning: dict[str, Any], profile: dict[str, Any]) -> str:
    levels = [str(pathway.get("priority") or "low"), str(warning.get("level") or "none")]
    care_level = str(profile.get("recommended_care_level") or "")
    if "critical" in levels or care_level == "urgent":
        return "critical"
    if "high" in levels or care_level == "manual_followup":
        return "high"
    if "medium" in levels or care_level in {"watch_closely", "strategy_repair"}:
        return "medium"
    return "low"


def _care_phase(pathway: dict[str, Any], warning: dict[str, Any], outcome: dict[str, Any]) -> str:
    route = str(pathway.get("route") or "")
    warning_level = str(warning.get("level") or "none")
    status = str(outcome.get("status") or "")
    if route == "urgent_safety" or warning_level == "critical" or status == "crisis_priority":
        return "safety"
    if route == "human_followup_recommended" or warning_level == "high":
        return "human_followup"
    if route == "repair_reply_style" or status == "needs_strategy_repair":
        return "repair"
    if route == "monitor_next_turn" or warning_level == "medium":
        return "monitor"
    return "support"


def _review_window(pathway: dict[str, Any], warning: dict[str, Any], profile: dict[str, Any]) -> int:
    candidates = [
        _safe_int(pathway.get("review_window_hours")),
        _safe_int(warning.get("review_window_hours")),
        _safe_int(profile.get("next_review_hours")),
    ]
    valid = [item for item in candidates if item is not None and item > 0]
    return min(valid) if valid else 72


def _primary_goal(care_phase: str, memory: dict[str, Any], warning: dict[str, Any]) -> str:
    if care_phase == "safety":
        return "first_restore_immediate_safety"
    if care_phase == "human_followup":
        return "connect_user_with_real_world_support"
    if care_phase == "repair":
        return "repair_trust_and_reduce_reply_pressure"
    if care_phase == "monitor":
        return "stabilize_entropy_and_track_next_change"
    if "privacy_reassurance_needed" in (memory.get("user_boundaries") or []):
        return "build_privacy_trust_before_problem_solving"
    if warning.get("trend_state") == "falling":
        return "consolidate_working_support_pattern"
    return "continue_contextual_support"


def _user_visible_focus(care_phase: str, memory: dict[str, Any], loop: dict[str, Any]) -> str:
    if care_phase == "safety":
        return "stabilize immediate safety before ordinary problem solving"
    if care_phase == "human_followup":
        return "offer practical support channels without pressuring the user"
    if care_phase == "repair":
        return "repair trust first, then ask only low-pressure questions"
    if care_phase == "monitor":
        return "keep the next reply calm, concrete, and easy to answer"
    preferred = memory.get("preferred_next_move") or loop.get("next_reply_mode")
    if preferred:
        return f"follow the user's preferred next move: {preferred}"
    return "continue warm, contextual support and preserve conversation continuity"


def _backend_focus(
    care_phase: str,
    pathway: dict[str, Any],
    warning: dict[str, Any],
    outcome: dict[str, Any],
) -> str:
    return (
        f"phase={care_phase}; route={pathway.get('route')}; warning={warning.get('level')}:"
        f"{warning.get('trend_state')}; outcome={outcome.get('status')}"
    )


def _next_actions(
    care_phase: str,
    pathway: dict[str, Any],
    warning: dict[str, Any],
    loop: dict[str, Any],
    memory: dict[str, Any],
) -> list[str]:
    actions = list(pathway.get("backend_actions") or [])
    warning_action = warning.get("recommended_action")
    loop_action = loop.get("loop_action")
    preferred_move = memory.get("preferred_next_move")
    if warning_action and warning_action != "continue_observation":
        actions.append(str(warning_action))
    if loop_action and loop_action != "continue_current_strategy":
        actions.append(f"apply_loop:{loop_action}")
    if preferred_move:
        actions.append(f"use_memory:{preferred_move}")
    if care_phase == "monitor":
        actions.append("compare_next_entropy_score")
    if care_phase == "support":
        actions.append("preserve_contextual_memory")
    return _dedupe(actions) or ["continue_observation"]


def _avoid_actions(care_phase: str, memory: dict[str, Any], loop: dict[str, Any]) -> list[str]:
    avoid = list(memory.get("avoid_next_reply") or [])
    avoid.extend(loop.get("constraints") or [])
    if care_phase in {"safety", "monitor"}:
        avoid.append("do_not_overload_with_long_plan")
    if care_phase in {"repair", "human_followup"}:
        avoid.append("do_not_sound_mechanical_or_alarmist")
    return _dedupe(avoid)


def _success_indicators(care_phase: str, warning: dict[str, Any]) -> list[str]:
    indicators = ["user_reply_is_contextually_connected"]
    if care_phase == "safety":
        indicators.extend(["user_confirms_immediate_safety", "real_world_support_contacted_or_identified"])
    elif care_phase == "human_followup":
        indicators.extend(["user_accepts_or_considers_support_channel", "distress_does_not_escalate"])
    elif care_phase == "repair":
        indicators.extend(["user_continues_conversation", "negative_feedback_reduces"])
    elif care_phase == "monitor":
        indicators.extend(["next_entropy_score_stable_or_lower", "user_can_name_one_small_next_step"])
    else:
        indicators.append("conversation_remains_natural")
    if warning.get("trend_state") == "falling":
        indicators.append("working_strategy_preserved")
    return _dedupe(indicators)


def _escalation_conditions(care_phase: str, warning: dict[str, Any]) -> list[str]:
    conditions = [
        "risk_level_becomes_high_or_critical",
        "entropy_score_rises_by_10_or_more",
        "user_reports_self_harm_or_no_safe_person",
    ]
    if care_phase == "repair":
        conditions.append("user_gives_repeated_negative_feedback")
    if warning.get("level") in {"medium", "high"}:
        conditions.append("trend_warning_persists_next_review")
    return _dedupe(conditions)


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        clean = str(item).strip()
        if clean and clean not in result:
            result.append(clean)
    return result
