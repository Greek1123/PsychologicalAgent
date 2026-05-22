from __future__ import annotations

from typing import Any

from .schemas import SupportPlan


def build_session_tracking_snapshot(
    *,
    session_id: str,
    conversation_memory: dict[str, Any] | None = None,
    longitudinal_profile: dict[str, Any] | None = None,
    intervention_effectiveness: dict[str, Any] | None = None,
    intervention_next_step: dict[str, Any] | None = None,
    reply_quality_report: dict[str, Any] | None = None,
    feedback_summary: dict[str, Any] | None = None,
    latest_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a hidden continuity card for the next backend decision.

    This is intentionally not user-visible. It tells the next request what the
    system has learned across turns: what to keep, what to avoid, and whether
    the next reply should repair, stabilize, or continue.
    """

    memory = conversation_memory or {}
    profile = longitudinal_profile or {}
    effectiveness = intervention_effectiveness or {}
    effectiveness_summary = effectiveness.get("summary") or {}
    next_step = intervention_next_step or {}
    quality_summary = (reply_quality_report or {}).get("summary") or {}
    feedback_summary = feedback_summary or {}
    latest_record = latest_record or {}

    stage = _stage(
        observation_count=int(profile.get("observation_count") or 0),
        next_reply_mode=str(next_step.get("next_reply_mode") or ""),
        care_level=str(profile.get("recommended_care_level") or ""),
        effectiveness_status=str(effectiveness_summary.get("overall_status") or ""),
    )
    dominant_needs = _dominant_needs(memory, profile, latest_record)
    failed_moves = _failed_moves(quality_summary, feedback_summary, next_step)
    stable_support_style = _stable_support_style(memory, effectiveness_summary, next_step)
    privacy_boundary = _privacy_boundary(memory, latest_record)

    return {
        "tracking_id": f"{session_id}:tracking:{profile.get('observation_count') or 0}",
        "session_id": session_id,
        "stage": stage,
        "dominant_needs": dominant_needs,
        "stable_support_style": stable_support_style,
        "failed_moves": failed_moves,
        "privacy_boundary": privacy_boundary,
        "current_risk_track": {
            "risk_course": profile.get("risk_course") or "unknown",
            "recommended_care_level": profile.get("recommended_care_level") or "observe",
            "latest_risk_level": latest_record.get("risk_level"),
            "human_followup_policy": next_step.get("human_followup_policy") or "none",
        },
        "entropy_track": {
            "entropy_course": profile.get("entropy_course") or "baseline",
            "latest_entropy_score": profile.get("latest_entropy_score"),
            "average_entropy": profile.get("average_entropy"),
            "volatility_score": profile.get("volatility_score") or 0,
            "review_window_hours": next_step.get("review_window_hours")
            or profile.get("next_review_hours")
            or 72,
        },
        "next_entry_instruction": _next_entry_instruction(
            stage=stage,
            privacy_boundary=privacy_boundary,
            dominant_needs=dominant_needs,
            failed_moves=failed_moves,
            next_step=next_step,
        ),
        "evidence": {
            "active_topics": memory.get("active_topics") or [],
            "user_boundaries": memory.get("user_boundaries") or [],
            "dominant_states": profile.get("dominant_states") or [],
            "dominant_stress_domains": profile.get("dominant_stress_domains") or [],
            "effectiveness_status": effectiveness_summary.get("overall_status"),
            "reply_quality_needs_review": quality_summary.get("needs_review"),
            "negative_feedback_count": feedback_summary.get("negative_count") or 0,
            "latest_response_id": latest_record.get("response_id"),
        },
    }


def enrich_student_context_with_session_tracking(
    student_context: dict[str, Any],
    tracking_snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    enriched = dict(student_context or {})
    if not tracking_snapshot:
        return enriched
    enriched["session_tracking"] = {
        "stage": tracking_snapshot.get("stage"),
        "dominant_needs": tracking_snapshot.get("dominant_needs") or [],
        "stable_support_style": tracking_snapshot.get("stable_support_style") or [],
        "failed_moves": tracking_snapshot.get("failed_moves") or [],
        "privacy_boundary": tracking_snapshot.get("privacy_boundary"),
        "current_risk_track": tracking_snapshot.get("current_risk_track") or {},
        "entropy_track": tracking_snapshot.get("entropy_track") or {},
        "next_entry_instruction": tracking_snapshot.get("next_entry_instruction"),
        "instruction": (
            "Use this hidden session-tracking snapshot to keep continuity across turns. "
            "Do not expose tracking labels, entropy numbers, risk course, or backend policy names."
        ),
    }
    return enriched


def apply_session_tracking_to_plan(
    plan: SupportPlan,
    *,
    tracking_snapshot: dict[str, Any] | None,
) -> SupportPlan:
    if not tracking_snapshot:
        return plan

    stage = str(tracking_snapshot.get("stage") or "")
    privacy_boundary = str(tracking_snapshot.get("privacy_boundary") or "")
    failed_moves = {str(item) for item in tracking_snapshot.get("failed_moves") or []}
    dominant_needs = {str(item) for item in tracking_snapshot.get("dominant_needs") or []}

    if privacy_boundary == "needs_explicit_reassurance":
        plan.summary = _ZH["privacy_summary"]
        plan.immediate_support = _prepend(plan.immediate_support, _ZH["privacy_support"])[:3]
        plan.follow_up = [_ZH["privacy_follow_up"]]

    if stage == "repair":
        plan.summary = _ZH["repair_summary"]
        plan.immediate_support = _prepend(plan.immediate_support, _ZH["repair_support"])[:2]
        plan.follow_up = [_ZH["repair_follow_up"]]

    if stage == "stabilize":
        plan.immediate_support = _prepend(plan.immediate_support, _ZH["stabilize_support"])[:3]
        plan.self_regulation = _prepend(plan.self_regulation, _ZH["stabilize_self"])[:2]

    if "avoid_generic_advice" in failed_moves:
        plan.immediate_support = _prepend(plan.immediate_support, _ZH["concrete_support"])[:3]
    if "avoid_question_pressure" in failed_moves:
        plan.follow_up = [_ZH["low_pressure_follow_up"]]
    if "sleep_disruption" in dominant_needs:
        plan.self_regulation = _prepend(plan.self_regulation, _ZH["sleep_support"])[:2]

    return _compact_plan(plan)


def _stage(
    *,
    observation_count: int,
    next_reply_mode: str,
    care_level: str,
    effectiveness_status: str,
) -> str:
    if next_reply_mode in {"safety_grounding", "warm_human_linkage"} or care_level in {"urgent", "manual_followup"}:
        return "safety_or_human_followup"
    if next_reply_mode == "repair_conversation" or effectiveness_status == "needs_repair":
        return "repair"
    if next_reply_mode == "stabilize_then_reduce_load" or care_level == "watch_closely":
        return "stabilize"
    if observation_count >= 3:
        return "continuity"
    if observation_count > 0:
        return "early_support"
    return "new_session"


def _dominant_needs(
    memory: dict[str, Any],
    profile: dict[str, Any],
    latest_record: dict[str, Any],
) -> list[str]:
    needs: list[str] = []
    needs.extend(str(item) for item in memory.get("active_topics") or [])
    needs.extend(str(item) for item in profile.get("dominant_stress_domains") or [])
    state = latest_record.get("primary_state")
    if state:
        needs.append(f"state:{state}")
    return _dedupe(needs)[:6]


def _failed_moves(
    quality_summary: dict[str, Any],
    feedback_summary: dict[str, Any],
    next_step: dict[str, Any],
) -> list[str]:
    failed: list[str] = []
    if int(quality_summary.get("needs_review") or 0) > 0:
        failed.append("avoid_generic_advice")
    if int(feedback_summary.get("negative_count") or 0) > 0:
        failed.append("avoid_repeating_rejected_style")
    if str(next_step.get("question_policy") or "") in {"low_pressure_or_no_question", "safety_check_only"}:
        failed.append("avoid_question_pressure")
    if bool(next_step.get("should_pause_advice")):
        failed.append("avoid_long_plans")
    return _dedupe(failed)


def _stable_support_style(
    memory: dict[str, Any],
    effectiveness_summary: dict[str, Any],
    next_step: dict[str, Any],
) -> list[str]:
    style = [
        "reflect_latest_turn_first",
        "use_plain_student_friendly_language",
        "keep_backend_analysis_hidden",
    ]
    preferred_move = memory.get("preferred_next_move")
    if preferred_move:
        style.append(str(preferred_move))
    if str(effectiveness_summary.get("overall_status") or "") in {"effective", "improving"}:
        style.append("keep_working_support_style")
    for item in next_step.get("preferred_moves") or []:
        style.append(str(item))
    return _dedupe(style)[:8]


def _privacy_boundary(memory: dict[str, Any], latest_record: dict[str, Any]) -> str:
    boundaries = {str(item) for item in memory.get("user_boundaries") or []}
    latest_text = str(latest_record.get("input_text") or "")
    if "privacy_reassurance_needed" in boundaries or any(term in latest_text for term in ("保密", "别人知道", "告诉别人")):
        return "needs_explicit_reassurance"
    if "low_disclosure_preferred" in boundaries:
        return "low_disclosure_preferred"
    return "none"


def _next_entry_instruction(
    *,
    stage: str,
    privacy_boundary: str,
    dominant_needs: list[str],
    failed_moves: list[str],
    next_step: dict[str, Any],
) -> str:
    parts = [
        "Start from the user's latest message and connect it to the existing session.",
        "Keep the reply visible as support, not backend analysis.",
    ]
    if privacy_boundary == "needs_explicit_reassurance":
        parts.append("Reassure privacy and user control before asking for details.")
    if stage == "repair":
        parts.append("Briefly repair the miss and return control to the user.")
    if stage == "stabilize":
        parts.append("Stabilize first, then offer only one small next step.")
    if "avoid_question_pressure" in failed_moves:
        parts.append("Do not ask multiple questions.")
    if dominant_needs:
        parts.append(f"Likely focus: {', '.join(dominant_needs[:3])}.")
    mode = next_step.get("next_reply_mode")
    if mode:
        parts.append(f"Hidden next reply mode: {mode}.")
    return " ".join(parts)


def _prepend(items: list[str], item: str) -> list[str]:
    return _dedupe([item, *items])


def _compact_plan(plan: SupportPlan) -> SupportPlan:
    plan.immediate_support = _dedupe(plan.immediate_support)[:3]
    plan.campus_actions = _dedupe(plan.campus_actions)[:2]
    plan.self_regulation = _dedupe(plan.self_regulation)[:2]
    plan.follow_up = _dedupe(plan.follow_up)[:1]
    return plan


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        clean = str(item).strip()
        if clean and clean not in result:
            result.append(clean)
    return result


_ZH = {
    "privacy_summary": "\u4f60\u62c5\u5fc3\u522b\u4eba\u77e5\u9053\uff0c\u8fd9\u4e2a\u62c5\u5fc3\u5f88\u91cd\u8981\uff0c\u6211\u4f1a\u5148\u5c0a\u91cd\u4f60\u7684\u8fb9\u754c\u3002\u4f60\u4e0d\u9700\u8981\u9a6c\u4e0a\u8bb2\u7ec6\u8282\uff0c\u6211\u4eec\u53ef\u4ee5\u5148\u53ea\u5904\u7406\u4f60\u6b64\u523b\u7684\u4e0d\u5b89\u3002",
    "privacy_support": "\u5728\u8fd9\u91cc\uff0c\u4f60\u53ef\u4ee5\u5148\u6309\u81ea\u5df1\u7684\u8282\u594f\u8bf4\uff1b\u5982\u679c\u4e0d\u60f3\u8bf4\u539f\u56e0\uff0c\u53ea\u8bf4\u201c\u6211\u73b0\u5728\u5f88\u6015\u201d\u6216\u201c\u6211\u60f3\u5148\u7f13\u4e00\u4e0b\u201d\u4e5f\u53ef\u4ee5\u3002",
    "privacy_follow_up": "\u4f60\u73b0\u5728\u4e0d\u7528\u8bb2\u9690\u79c1\u7ec6\u8282\uff0c\u53ea\u9700\u8981\u544a\u8bc9\u6211\uff1a\u4f60\u66f4\u60f3\u5148\u5b89\u9759\u4e00\u4f1a\u513f\uff0c\u8fd8\u662f\u60f3\u8981\u4e00\u4e2a\u5f88\u5c0f\u7684\u7f13\u89e3\u529e\u6cd5\uff1f",
    "repair_summary": "\u6211\u521a\u624d\u7684\u56de\u590d\u53ef\u80fd\u6ca1\u6709\u8ddf\u4e0a\u4f60\u771f\u6b63\u5728\u610f\u7684\u90e8\u5206\uff0c\u8fd9\u91cc\u6211\u5148\u628a\u8282\u594f\u653e\u6162\u3002",
    "repair_support": "\u6211\u4f1a\u5148\u63a5\u4f4f\u4f60\u6b64\u523b\u7684\u611f\u53d7\uff0c\u800c\u4e0d\u662f\u6025\u7740\u5206\u6790\u3001\u8ffd\u95ee\u6216\u7ed9\u4e00\u5806\u65b9\u6cd5\u3002",
    "repair_follow_up": "\u5982\u679c\u4f60\u613f\u610f\uff0c\u6211\u4eec\u53ef\u4ee5\u5148\u4ece\u4e00\u53e5\u8bdd\u5f00\u59cb\uff1a\u201c\u73b0\u5728\u6700\u8ba9\u6211\u96be\u53d7\u7684\u662f...\u201d",
    "stabilize_support": "\u5148\u4e0d\u628a\u6240\u6709\u95ee\u9898\u90fd\u642c\u51fa\u6765\uff0c\u6211\u4eec\u53ea\u628a\u6ce8\u610f\u529b\u653e\u5230\u63a5\u4e0b\u6765\u51e0\u5206\u949f\uff0c\u8ba9\u8eab\u4f53\u548c\u60c5\u7eea\u5148\u964d\u4e00\u70b9\u3002",
    "stabilize_self": "\u53ef\u4ee5\u5148\u505a\u4e00\u4e2a\u5f88\u5c0f\u7684\u52a8\u4f5c\uff1a\u628a\u624b\u653e\u5728\u684c\u9762\u6216\u817f\u4e0a\uff0c\u611f\u53d7\u4e00\u4e0b\u63a5\u89e6\u611f\uff0c\u7136\u540e\u6162\u6162\u547c\u51fa\u4e00\u53e3\u6c14\u3002",
    "concrete_support": "\u6211\u4eec\u5148\u4e0d\u8bf4\u7a7a\u6cdb\u7684\u201c\u522b\u60f3\u592a\u591a\u201d\uff0c\u800c\u662f\u627e\u4e00\u4e2a\u80fd\u7acb\u523b\u51cf\u8f7b\u4e00\u70b9\u538b\u529b\u7684\u5c0f\u52a8\u4f5c\u3002",
    "low_pressure_follow_up": "\u4f60\u53ef\u4ee5\u4e0d\u56de\u7b54\u95ee\u9898\uff0c\u53ea\u8981\u544a\u8bc9\u6211\u4e00\u4e2a\u8bcd\u4e5f\u884c\uff1a\u6015\u3001\u70e6\u3001\u7d2f\uff0c\u6216\u8005\u60f3\u5b89\u9759\u3002",
    "sleep_support": "\u5982\u679c\u7761\u4e0d\u7740\u5df2\u7ecf\u6301\u7eed\u5f71\u54cd\u4f60\uff0c\u4eca\u665a\u7684\u76ee\u6807\u5148\u4e0d\u662f\u201c\u7acb\u523b\u7761\u7740\u201d\uff0c\u800c\u662f\u5148\u628a\u8eab\u4f53\u7684\u7d27\u7ef7\u5ea6\u964d\u4e00\u70b9\u3002",
}
