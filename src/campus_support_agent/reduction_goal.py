from __future__ import annotations

from typing import Any

from .schemas import (
    DynamicAdjustment,
    EntropyOrchestration,
    EntropyReductionGoal,
    PsychologicalEntropy,
    ReferralDecision,
    RiskAssessment,
    StateProfile,
)


def build_entropy_reduction_goal(
    *,
    text: str,
    risk: RiskAssessment,
    entropy: PsychologicalEntropy,
    state_profile: StateProfile,
    dynamic_adjustment: DynamicAdjustment,
    entropy_orchestration: EntropyOrchestration,
    referral_decision: ReferralDecision | None = None,
    session_continuity: dict[str, Any] | None = None,
) -> EntropyReductionGoal:
    stage = str((session_continuity or {}).get("dialogue_stage") or "")
    active_driver = _select_active_driver(entropy, state_profile, risk, stage)
    priority = _priority(risk, entropy, dynamic_adjustment, stage)
    target_delta = _target_delta(priority, entropy.score)
    micro_intervention = _micro_intervention(
        active_driver=active_driver,
        stage=stage,
        state_profile=state_profile,
        orchestration=entropy_orchestration,
        referral_decision=referral_decision,
    )
    reduction_goal = _reduction_goal_text(active_driver, stage, state_profile)

    return EntropyReductionGoal(
        goal_id=f"goal:{entropy_orchestration.orchestration_id}",
        active_driver=active_driver,
        reduction_goal=reduction_goal,
        target_entropy_delta=target_delta,
        micro_intervention=micro_intervention,
        review_condition=_review_condition(priority, dynamic_adjustment, stage),
        priority=priority,
        success_signal=_success_signal(active_driver, stage),
        avoid=_avoid_for_goal(active_driver, stage),
        evidence={
            "risk_level": risk.level,
            "risk_score": risk.score,
            "entropy_score": entropy.score,
            "entropy_level": entropy.level,
            "balance_state": entropy.balance_state,
            "dominant_drivers": entropy.dominant_drivers,
            "state": state_profile.primary_state,
            "stress_domains": state_profile.stress_domains,
            "dialogue_stage": stage,
            "dynamic_action": dynamic_adjustment.action,
            "orchestration_route": entropy_orchestration.route,
            "input_preview": text[:80],
        },
    )


def _select_active_driver(
    entropy: PsychologicalEntropy,
    state_profile: StateProfile,
    risk: RiskAssessment,
    stage: str,
) -> str:
    if risk.level in {"high", "critical"} or stage == "safety_priority":
        return "risk_pressure"
    if state_profile.boundary_flags or stage == "boundary_building":
        return "privacy_boundary_tension"
    if "sleep" in state_profile.stress_domains or state_profile.body_signals:
        return "physiological_imbalance"
    if "academic" in state_profile.stress_domains or any("认知负荷" in item for item in entropy.dominant_drivers):
        return "cognitive_load"
    if "social" in state_profile.stress_domains or "dorm" in state_profile.primary_state:
        return "social_support_tension"
    if entropy.dimensions.emotion_intensity >= 60 or entropy.dimensions.emotional_volatility >= 60:
        return "emotion_intensity"
    return "general_uncertainty"


def _priority(
    risk: RiskAssessment,
    entropy: PsychologicalEntropy,
    dynamic_adjustment: DynamicAdjustment,
    stage: str,
) -> str:
    if risk.level == "critical" or stage == "safety_priority":
        return "critical"
    if risk.level == "high" or dynamic_adjustment.should_refer:
        return "high"
    if entropy.score >= 65 or stage in {"deteriorating_watch", "high_entropy_stabilization"}:
        return "medium_high"
    if entropy.score >= 40:
        return "medium"
    return "low"


def _target_delta(priority: str, entropy_score: int) -> int:
    if priority == "critical":
        return -4
    if priority == "high":
        return -6
    if priority == "medium_high":
        return -8 if entropy_score >= 70 else -6
    if priority == "medium":
        return -5
    return -3


def _micro_intervention(
    *,
    active_driver: str,
    stage: str,
    state_profile: StateProfile,
    orchestration: EntropyOrchestration,
    referral_decision: ReferralDecision | None,
) -> str:
    if stage == "safety_priority" or active_driver == "risk_pressure":
        return "先确认现实安全，并引导联系现实支持。"
    if referral_decision and referral_decision.should_refer:
        return "温和提示可联系辅导员、心理中心或可信任同伴。"
    if active_driver == "privacy_boundary_tension":
        return "先保证边界，不追问细节，只允许用户低暴露表达。"
    if active_driver == "physiological_imbalance":
        return "先做一个身体稳定动作，再处理问题本身。"
    if active_driver == "cognitive_load":
        return "把任务缩小到十到十五分钟内的一步。"
    if active_driver == "social_support_tension":
        return "先帮助用户离开触发点或建立短暂边界。"
    if state_profile.weak_input_detected or state_profile.noisy_input_detected:
        return "先复述可能含义，请用户用最省力方式确认。"
    return orchestration.user_visible_goal or "先承接情绪，再只推进一个小步骤。"


def _reduction_goal_text(active_driver: str, stage: str, state_profile: StateProfile) -> str:
    if stage == "safety_priority":
        return "把危机风险从对话层转入现实安全支持。"
    mapping = {
        "risk_pressure": "降低风险压力，优先建立现实支持连接。",
        "privacy_boundary_tension": "建立隐私安全感，降低用户被追问和暴露的压力。",
        "physiological_imbalance": "先稳定睡眠、饮食或身体紧绷带来的熵增。",
        "cognitive_load": "降低灾难化推演和任务堆积造成的认知负荷。",
        "social_support_tension": "降低宿舍、人际或支持关系带来的持续触发。",
        "emotion_intensity": "先降低情绪强度，让用户从崩溃边缘回到可对话状态。",
        "general_uncertainty": "降低模糊不确定感，帮助用户找到一个可表达入口。",
    }
    return mapping.get(active_driver, state_profile.recommended_focus or "降低当前主要心理熵驱动。")


def _review_condition(priority: str, dynamic_adjustment: DynamicAdjustment, stage: str) -> str:
    if priority == "critical":
        return "本轮立即复核安全状态。"
    if priority == "high":
        return "下一轮或 12 小时内复核是否已连接现实支持。"
    if dynamic_adjustment.trend_direction == "up" or stage == "deteriorating_watch":
        return "下一轮复核心理熵是否继续上升。"
    if dynamic_adjustment.trend_direction == "down":
        return "下一轮确认有效动作是否可以保留。"
    return "24 到 72 小时内观察熵值、睡眠和表达意愿变化。"


def _success_signal(active_driver: str, stage: str) -> str:
    if stage == "safety_priority":
        return "用户确认当前安全，或已经联系现实支持。"
    mapping = {
        "privacy_boundary_tension": "用户愿意以低暴露方式继续，或明确表示被尊重。",
        "physiological_imbalance": "睡眠、饮食或身体紧绷有轻微缓解。",
        "cognitive_load": "用户能说出一个小步骤，而不是继续整体灾难化。",
        "social_support_tension": "用户能暂时离开触发点，或形成一个边界动作。",
        "emotion_intensity": "用户情绪强度下降，能用词描述当前感受。",
        "risk_pressure": "用户愿意联系现实支持或接受转介建议。",
    }
    return mapping.get(active_driver, "用户能继续表达且心理熵没有继续升高。")


def _avoid_for_goal(active_driver: str, stage: str) -> list[str]:
    avoid = ["不要把心理熵等后台术语直接说给用户", "不要连续追问多个问题"]
    if active_driver == "privacy_boundary_tension" or stage == "boundary_building":
        avoid.extend(["不要追问具体人名、地点或细节", "不要要求用户完整复述经过"])
    if active_driver == "cognitive_load":
        avoid.append("不要一次给完整学习计划")
    if active_driver == "physiological_imbalance":
        avoid.append("不要只讲道理而忽略身体状态")
    if stage in {"deteriorating_watch", "high_entropy_stabilization", "safety_priority"}:
        avoid.append("不要切换到无关闲聊")
    return list(dict.fromkeys(avoid))
