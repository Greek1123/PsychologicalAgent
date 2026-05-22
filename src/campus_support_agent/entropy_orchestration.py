from __future__ import annotations

from typing import Any

from .dialogue_memory import ConversationMemory, build_conversation_memory
from .schemas import (
    DynamicAdjustment,
    EntropyOrchestration,
    FeedbackAdaptation,
    InterventionStrategy,
    PsychologicalEntropy,
    ReferralDecision,
    RiskAssessment,
    RiskLevel,
    StateProfile,
)


def build_entropy_orchestration(
    *,
    text: str,
    conversation_history: list[dict[str, Any]] | None,
    risk: RiskAssessment,
    entropy: PsychologicalEntropy,
    state_profile: StateProfile,
    intervention_strategy: InterventionStrategy,
    dynamic_adjustment: DynamicAdjustment,
    feedback_adaptation: FeedbackAdaptation,
    referral_decision: ReferralDecision,
) -> EntropyOrchestration:
    """Create a single next-turn control decision for entropy reduction."""

    memory = build_conversation_memory(conversation_history, current_text=text)
    route = _route(
        risk=risk,
        dynamic_adjustment=dynamic_adjustment,
        feedback_adaptation=feedback_adaptation,
        referral_decision=referral_decision,
        memory=memory,
    )
    response_constraints = _response_constraints(
        route=route,
        memory=memory,
        dynamic_adjustment=dynamic_adjustment,
        feedback_adaptation=feedback_adaptation,
        state_profile=state_profile,
    )
    hidden_actions = _hidden_actions(
        route=route,
        risk=risk,
        dynamic_adjustment=dynamic_adjustment,
        feedback_adaptation=feedback_adaptation,
        referral_decision=referral_decision,
    )
    user_visible_goal = _user_visible_goal(route, memory, state_profile)
    next_focus = _next_focus(route, memory, dynamic_adjustment, intervention_strategy)
    reasons = _reasons(
        route=route,
        risk=risk,
        entropy=entropy,
        state_profile=state_profile,
        dynamic_adjustment=dynamic_adjustment,
        feedback_adaptation=feedback_adaptation,
        memory=memory,
    )

    return EntropyOrchestration(
        orchestration_id=f"{route}:{dynamic_adjustment.action}:{intervention_strategy.strategy_id}",
        route=route,
        user_visible_goal=user_visible_goal,
        next_focus=next_focus,
        response_constraints=response_constraints,
        hidden_actions=hidden_actions,
        memory_topics=memory.active_topics,
        boundary_flags=memory.user_boundaries,
        reasons=reasons,
    )


def _route(
    *,
    risk: RiskAssessment,
    dynamic_adjustment: DynamicAdjustment,
    feedback_adaptation: FeedbackAdaptation,
    referral_decision: ReferralDecision,
    memory: ConversationMemory,
) -> str:
    if risk.level == RiskLevel.CRITICAL or dynamic_adjustment.action == "urgent_referral":
        return "safety_first"
    if referral_decision.should_refer or dynamic_adjustment.should_refer:
        return "human_support_linkage"
    if feedback_adaptation.mode == "repair_next_turn":
        return "repair_conversation"
    if dynamic_adjustment.action in {"soften_and_stabilize", "escalate_support"}:
        return "stabilize_and_reduce_load"
    if memory.user_boundaries:
        return "boundary_respecting_support"
    if dynamic_adjustment.action == "maintain_and_consolidate":
        return "consolidate_progress"
    return "continue_support"


def _response_constraints(
    *,
    route: str,
    memory: ConversationMemory,
    dynamic_adjustment: DynamicAdjustment,
    feedback_adaptation: FeedbackAdaptation,
    state_profile: StateProfile,
) -> list[str]:
    constraints = [
        "承接最近上下文，不要像新会话一样重新开场",
        "不要把心理熵、风险分数、内部评估术语直接说给用户",
        "不要重复上一轮原句",
    ]
    if route == "safety_first":
        constraints.extend(["优先确认安全", "引导联系现实支持", "避免普通安慰和复杂分析"])
    if route == "human_support_linkage":
        constraints.extend(["语气保持温和", "说明现实支持是加一层保护而不是否定用户"])
    if route == "repair_conversation":
        constraints.extend(["先承认可能没接住", "降低提问压力", "换一种更贴近用户处境的表达"])
    if route == "stabilize_and_reduce_load":
        constraints.extend(["把建议缩小到一个动作", "不要一次给完整计划"])
    if route == "boundary_respecting_support" or memory.user_boundaries:
        constraints.extend(["不要像新会话一样重新开场", "尊重用户不想细说", "允许只说感受或暂时沉默"])
    if feedback_adaptation.question_pressure == "low" or state_profile.boundary_flags:
        constraints.append("最多问一个可选问题")
    if feedback_adaptation.detail_level == "more_concrete":
        constraints.append("提供一个具体、低负担的小步骤")
    if dynamic_adjustment.intensity_shift == "increase":
        constraints.append("降低用户负荷，但提高系统观察和支持强度")
    return _dedupe(constraints)


def _hidden_actions(
    *,
    route: str,
    risk: RiskAssessment,
    dynamic_adjustment: DynamicAdjustment,
    feedback_adaptation: FeedbackAdaptation,
    referral_decision: ReferralDecision,
) -> list[str]:
    actions = ["记录本轮熵值、风险和策略变化"]
    if route in {"safety_first", "human_support_linkage"} or risk.needs_human_followup:
        actions.append("标记人工关注或转介候选")
    if dynamic_adjustment.review_window_hours <= 12:
        actions.append(f"{dynamic_adjustment.review_window_hours} 小时内复查状态")
    else:
        actions.append(f"{dynamic_adjustment.review_window_hours} 小时内常规观察")
    if feedback_adaptation.should_collect_bad_case:
        actions.append("将本轮纳入后续坏例回收候选")
    if feedback_adaptation.should_avoid_repetition:
        actions.append("下一轮避免复用相同开场和建议")
    if referral_decision.should_refer:
        actions.append(f"建议转介渠道：{referral_decision.recommended_channel or '学校心理支持资源'}")
    return _dedupe(actions)


def _user_visible_goal(route: str, memory: ConversationMemory, state_profile: StateProfile) -> str:
    if route == "safety_first":
        return "先保证用户现实安全，并把支持从线上对话连接到现实帮助。"
    if route == "human_support_linkage":
        return "让用户知道不需要独自硬撑，同时温和连接校园支持资源。"
    if route == "repair_conversation":
        return "修复上一轮没接住的感觉，重新回到用户真正表达的处境。"
    if route == "stabilize_and_reduce_load":
        return "先降低当下负荷，把问题缩小成用户能完成的一小步。"
    if route == "boundary_respecting_support":
        return "尊重用户边界，只围绕感受和当下支持继续。"
    if route == "consolidate_progress":
        return "保留已经有效的小动作，避免新增压力。"
    if memory.active_topics:
        return f"围绕{memory.active_topics[0]}继续提供低负担支持。"
    return state_profile.recommended_focus or "稳定陪伴并逐步澄清当前困扰。"


def _next_focus(
    route: str,
    memory: ConversationMemory,
    dynamic_adjustment: DynamicAdjustment,
    intervention_strategy: InterventionStrategy,
) -> str:
    if route == "repair_conversation":
        return "acknowledge_and_repair"
    if route == "boundary_respecting_support":
        return "privacy_and_low_pressure_presence"
    if route == "safety_first":
        return "immediate_safety_check"
    if memory.active_topics:
        return memory.continuity_focus
    return dynamic_adjustment.next_focus or intervention_strategy.next_step


def _reasons(
    *,
    route: str,
    risk: RiskAssessment,
    entropy: PsychologicalEntropy,
    state_profile: StateProfile,
    dynamic_adjustment: DynamicAdjustment,
    feedback_adaptation: FeedbackAdaptation,
    memory: ConversationMemory,
) -> list[str]:
    reasons = [
        f"route:{route}",
        f"risk:{risk.level}",
        f"entropy_score:{entropy.score}",
        f"entropy_trend:{dynamic_adjustment.trend_direction}:{dynamic_adjustment.trend_delta}",
        f"state:{state_profile.primary_state}",
        f"dynamic_action:{dynamic_adjustment.action}",
        f"feedback_mode:{feedback_adaptation.mode}",
    ]
    reasons.extend(f"topic:{topic}" for topic in memory.active_topics)
    reasons.extend(f"boundary:{index + 1}" for index, _ in enumerate(memory.user_boundaries))
    return _dedupe(reasons)


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item and item not in result:
            result.append(item)
    return result
