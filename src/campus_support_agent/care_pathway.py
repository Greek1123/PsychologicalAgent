from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .schemas import (
    CarePathwayDecision,
    DynamicAdjustment,
    FeedbackAdaptation,
    LongitudinalStateProfile,
)


def build_care_pathway_decision(
    *,
    session_id: str,
    longitudinal_profile: LongitudinalStateProfile,
    latest_dynamic_adjustment: DynamicAdjustment | dict[str, Any] | None = None,
    latest_feedback_adaptation: FeedbackAdaptation | dict[str, Any] | None = None,
    latest_referral_decision: dict[str, Any] | None = None,
    feedback_summary: dict[str, Any] | None = None,
) -> CarePathwayDecision:
    """Compose long-term state, entropy movement, feedback, and referral into one backend route."""

    dynamic = _to_dict(latest_dynamic_adjustment)
    feedback = _to_dict(latest_feedback_adaptation)
    referral = latest_referral_decision or {}
    feedback_summary = feedback_summary or {}
    care_level = longitudinal_profile.recommended_care_level
    reasons = _base_reasons(longitudinal_profile, dynamic, feedback, referral)

    if care_level == "urgent" or referral.get("urgency") == "urgent":
        return _decision(
            session_id=session_id,
            route="urgent_safety",
            priority="critical",
            user_visible_mode="safety_first_support",
            review_window_hours=1,
            should_notify_human=True,
            should_pause_ai_only_reply=True,
            backend_actions=[
                "activate_crisis_protocol",
                "show_emergency_resources",
                "record_manual_followup_required",
            ],
            rationale=[*reasons, "pathway:critical_or_urgent"],
            evidence=_evidence(longitudinal_profile, dynamic, feedback, referral, feedback_summary),
        )

    if care_level == "manual_followup" or dynamic.get("should_refer") or referral.get("should_refer"):
        return _decision(
            session_id=session_id,
            route="human_followup_recommended",
            priority="high",
            user_visible_mode="support_plus_human_option",
            review_window_hours=min(longitudinal_profile.next_review_hours, 12),
            should_notify_human=True,
            should_pause_ai_only_reply=False,
            backend_actions=[
                "recommend_campus_counseling_channel",
                "keep_reply_low_pressure",
                "track_next_entropy_point",
            ],
            rationale=[*reasons, "pathway:manual_followup"],
            evidence=_evidence(longitudinal_profile, dynamic, feedback, referral, feedback_summary),
        )

    if care_level == "strategy_repair" or feedback.get("mode") == "repair_next_turn":
        return _decision(
            session_id=session_id,
            route="repair_reply_style",
            priority="medium",
            user_visible_mode="acknowledge_and_repair",
            review_window_hours=min(longitudinal_profile.next_review_hours, 24),
            should_notify_human=False,
            should_pause_ai_only_reply=False,
            backend_actions=[
                "avoid_repetition",
                "reduce_question_pressure",
                "collect_bad_case_if_negative",
            ],
            rationale=[*reasons, "pathway:feedback_repair"],
            evidence=_evidence(longitudinal_profile, dynamic, feedback, referral, feedback_summary),
        )

    if care_level == "watch_closely" or dynamic.get("action") in {
        "soften_and_stabilize",
        "escalate_support",
    }:
        return _decision(
            session_id=session_id,
            route="monitor_next_turn",
            priority="medium",
            user_visible_mode="low_pressure_stabilization",
            review_window_hours=min(longitudinal_profile.next_review_hours, 24),
            should_notify_human=False,
            should_pause_ai_only_reply=False,
            backend_actions=[
                "keep_intervention_small",
                "compare_entropy_next_turn",
                "check_support_access_if_score_rises",
            ],
            rationale=[*reasons, "pathway:entropy_watch"],
            evidence=_evidence(longitudinal_profile, dynamic, feedback, referral, feedback_summary),
        )

    return _decision(
        session_id=session_id,
        route="continue_observation",
        priority="low",
        user_visible_mode="normal_supportive_chat",
        review_window_hours=longitudinal_profile.next_review_hours,
        should_notify_human=False,
        should_pause_ai_only_reply=False,
        backend_actions=[
            "continue_context_memory",
            "update_entropy_trace",
            "preserve_working_style",
        ],
        rationale=[*reasons, "pathway:observe"],
        evidence=_evidence(longitudinal_profile, dynamic, feedback, referral, feedback_summary),
    )


def _decision(
    *,
    session_id: str,
    route: str,
    priority: str,
    user_visible_mode: str,
    review_window_hours: int,
    should_notify_human: bool,
    should_pause_ai_only_reply: bool,
    backend_actions: list[str],
    rationale: list[str],
    evidence: dict[str, Any],
) -> CarePathwayDecision:
    return CarePathwayDecision(
        pathway_id=f"{session_id}:care_pathway:{route}",
        session_id=session_id,
        route=route,
        priority=priority,
        user_visible_mode=user_visible_mode,
        review_window_hours=review_window_hours,
        should_notify_human=should_notify_human,
        should_pause_ai_only_reply=should_pause_ai_only_reply,
        backend_actions=_dedupe(backend_actions),
        rationale=_dedupe(rationale),
        evidence=evidence,
    )


def _to_dict(value: DynamicAdjustment | FeedbackAdaptation | dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    return asdict(value)


def _base_reasons(
    profile: LongitudinalStateProfile,
    dynamic: dict[str, Any],
    feedback: dict[str, Any],
    referral: dict[str, Any],
) -> list[str]:
    reasons = [
        f"care_level:{profile.recommended_care_level}",
        f"entropy_course:{profile.entropy_course}",
        f"risk_course:{profile.risk_course}",
        f"engagement:{profile.engagement_signal}",
    ]
    if dynamic.get("action"):
        reasons.append(f"dynamic_action:{dynamic['action']}")
    if feedback.get("mode"):
        reasons.append(f"feedback_mode:{feedback['mode']}")
    if referral.get("urgency"):
        reasons.append(f"referral_urgency:{referral['urgency']}")
    return reasons


def _evidence(
    profile: LongitudinalStateProfile,
    dynamic: dict[str, Any],
    feedback: dict[str, Any],
    referral: dict[str, Any],
    feedback_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "observation_count": profile.observation_count,
        "latest_entropy_score": profile.latest_entropy_score,
        "peak_entropy_score": profile.peak_entropy_score,
        "dominant_states": profile.dominant_states,
        "dominant_stress_domains": profile.dominant_stress_domains,
        "dynamic_adjustment_id": dynamic.get("adjustment_id"),
        "feedback_adaptation_id": feedback.get("adaptation_id"),
        "referral_should_refer": bool(referral.get("should_refer")),
        "feedback_negative_count": int(feedback_summary.get("negative_count") or 0),
    }


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item not in result:
            result.append(item)
    return result
