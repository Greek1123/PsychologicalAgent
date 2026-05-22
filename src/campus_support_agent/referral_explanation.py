from __future__ import annotations

from typing import Any

from .schemas import (
    DynamicAdjustment,
    EntropyReductionGoal,
    GoalAttainmentEvaluation,
    PsychologicalEntropy,
    ReferralDecision,
    ReferralExplanation,
    RiskAssessment,
    StateProfile,
)


def build_referral_explanation(
    *,
    risk: RiskAssessment,
    entropy: PsychologicalEntropy,
    state_profile: StateProfile,
    referral_decision: ReferralDecision | None,
    dynamic_adjustment: DynamicAdjustment | None = None,
    reduction_goal: EntropyReductionGoal | None = None,
    latest_goal_attainment: GoalAttainmentEvaluation | dict[str, Any] | None = None,
    session_continuity: dict[str, Any] | None = None,
) -> ReferralExplanation:
    urgency = (referral_decision.urgency if referral_decision else "none") or "none"
    should_refer = bool(referral_decision.should_refer) if referral_decision else False
    trigger_reasons = _trigger_reasons(
        risk=risk,
        entropy=entropy,
        state_profile=state_profile,
        referral_decision=referral_decision,
        dynamic_adjustment=dynamic_adjustment,
        reduction_goal=reduction_goal,
        latest_goal_attainment=latest_goal_attainment,
        session_continuity=session_continuity,
    )
    referral_level = _referral_level(risk, entropy, urgency, should_refer, trigger_reasons)
    channel = _recommended_channel(referral_level, state_profile, urgency)

    return ReferralExplanation(
        explanation_id=f"referral_explain:{risk.level}:{entropy.score}:{urgency}",
        should_escalate=referral_level != "none",
        referral_level=referral_level,
        recommended_channel=channel,
        user_visible_reason=_user_visible_reason(referral_level, channel),
        backend_reason=_backend_reason(referral_level, trigger_reasons),
        urgency=urgency,
        trigger_reasons=trigger_reasons,
        protective_notes=_protective_notes(referral_level),
        evidence={
            "risk_level": risk.level,
            "risk_score": risk.score,
            "risk_reason": risk.reason,
            "trigger_terms": risk.trigger_terms,
            "entropy_score": entropy.score,
            "entropy_level": entropy.level,
            "balance_state": entropy.balance_state,
            "dominant_drivers": entropy.dominant_drivers,
            "primary_state": state_profile.primary_state,
            "risk_signals": state_profile.risk_signals,
            "body_signals": state_profile.body_signals,
            "dynamic_action": getattr(dynamic_adjustment, "action", None),
            "reduction_goal_priority": getattr(reduction_goal, "priority", None),
            "goal_attainment_status": _goal_status(latest_goal_attainment),
            "dialogue_stage": (session_continuity or {}).get("dialogue_stage"),
        },
    )


def _trigger_reasons(
    *,
    risk: RiskAssessment,
    entropy: PsychologicalEntropy,
    state_profile: StateProfile,
    referral_decision: ReferralDecision | None,
    dynamic_adjustment: DynamicAdjustment | None,
    reduction_goal: EntropyReductionGoal | None,
    latest_goal_attainment: GoalAttainmentEvaluation | dict[str, Any] | None,
    session_continuity: dict[str, Any] | None,
) -> list[str]:
    reasons: list[str] = []
    if risk.level == "critical":
        reasons.append("critical_risk_signal")
    elif risk.level == "high":
        reasons.append("high_risk_signal")
    if risk.trigger_terms:
        reasons.append("explicit_risk_terms")
    if entropy.score >= 75:
        reasons.append("very_high_entropy")
    elif entropy.score >= 65:
        reasons.append("high_entropy")
    if entropy.balance_state == "fragile":
        reasons.append("fragile_balance_state")
    if dynamic_adjustment and dynamic_adjustment.should_refer:
        reasons.append("dynamic_adjustment_recommends_referral")
    if referral_decision and referral_decision.should_refer:
        reasons.extend(referral_decision.reasons or ["referral_decision_positive"])
    if reduction_goal and reduction_goal.priority in {"critical", "high"}:
        reasons.append("high_priority_reduction_goal")
    if _goal_status(latest_goal_attainment) in {"worsened", "ineffective"}:
        reasons.append("recent_goal_failed_or_worsened")
    if (session_continuity or {}).get("dialogue_stage") in {"safety_priority", "human_followup_watch"}:
        reasons.append("session_stage_requires_human_watch")
    if state_profile.body_signals and entropy.score >= 55:
        reasons.append("body_functioning_affected")
    return list(dict.fromkeys(reasons))


def _referral_level(
    risk: RiskAssessment,
    entropy: PsychologicalEntropy,
    urgency: str,
    should_refer: bool,
    trigger_reasons: list[str],
) -> str:
    if risk.level == "critical" or urgency == "urgent" or "critical_risk_signal" in trigger_reasons:
        return "emergency"
    if risk.level == "high" or urgency == "recommended":
        return "professional_followup"
    if should_refer or entropy.score >= 70 or "recent_goal_failed_or_worsened" in trigger_reasons:
        return "campus_support_watch"
    if entropy.score >= 60 or "body_functioning_affected" in trigger_reasons:
        return "soft_checkin"
    return "none"


def _recommended_channel(referral_level: str, state_profile: StateProfile, urgency: str) -> str:
    if referral_level == "emergency":
        return "emergency_contact_or_crisis_hotline"
    if referral_level == "professional_followup":
        return "campus_counseling_center"
    if referral_level == "campus_support_watch":
        if "academic" in state_profile.stress_domains:
            return "counselor_or_academic_advisor"
        return "trusted_person_or_campus_counselor"
    if referral_level == "soft_checkin":
        return "trusted_peer_or_self_monitoring_checkin"
    return "none"


def _user_visible_reason(referral_level: str, channel: str) -> str:
    if referral_level == "emergency":
        return "你现在的安全比继续分析更重要，建议马上联系现实中的紧急支持。"
    if referral_level == "professional_followup":
        return "这个状态已经不适合只靠自己硬撑，可以让学校心理中心或专业支持靠近一点。"
    if referral_level == "campus_support_watch":
        return "如果这种状态持续，建议把现实里的支持拉近一点，比如辅导员、心理老师或可信任的人。"
    if referral_level == "soft_checkin":
        return "可以先找一个可信任的人简单说一声，让自己不要完全一个人扛着。"
    return "当前暂不需要转介，继续观察状态变化。"


def _backend_reason(referral_level: str, trigger_reasons: list[str]) -> str:
    if referral_level == "none":
        return "未达到转介阈值，保持普通支持和动态观察。"
    return "触发转介解释层：" + "、".join(trigger_reasons or ["manual_review_recommended"])


def _protective_notes(referral_level: str) -> list[str]:
    if referral_level == "emergency":
        return ["避免让用户独处", "优先现实安全和紧急联系人", "AI 不继续做深度探索"]
    if referral_level == "professional_followup":
        return ["说明转介是增加支持而非否定用户", "保持用户自主选择感", "避免制造羞耻感"]
    if referral_level == "campus_support_watch":
        return ["使用温和建议", "强调可以先从可信任的人开始", "继续观察熵值变化"]
    if referral_level == "soft_checkin":
        return ["不夸大风险", "鼓励轻量求助", "保留继续聊天入口"]
    return ["继续普通支持", "不向用户展示后台风险标签"]


def _goal_status(goal_attainment: GoalAttainmentEvaluation | dict[str, Any] | None) -> str | None:
    if goal_attainment is None:
        return None
    if isinstance(goal_attainment, dict):
        return goal_attainment.get("status")
    return goal_attainment.status
