from __future__ import annotations

from typing import Any

from .schemas import StrategyReselectionDecision


FAILURE_STATUSES = {"worsened", "ineffective", "not_achieved"}
SOFT_FAILURE_STATUSES = {"holding", "partially_achieved"}


def build_strategy_reselection_decision(
    *,
    goal_attainment_timeline: list[dict[str, Any]],
    session_continuity: dict[str, Any] | None = None,
    latest_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    recent = goal_attainment_timeline[-3:]
    latest = recent[-1] if recent else {}
    stage = str((session_continuity or {}).get("dialogue_stage") or "")
    latest_goal = (latest_record or {}).get("reduction_goal") or {}
    latest_driver = latest.get("driver") or latest_goal.get("active_driver")
    statuses = [str(item.get("status") or "") for item in recent]

    trigger = _trigger(statuses, stage, latest_record)
    should_reselect = trigger != "none"
    recommended_strategy, recommended_driver = _recommend_strategy(
        trigger=trigger,
        latest_driver=latest_driver,
        stage=stage,
        latest_record=latest_record,
    )
    priority = _priority(trigger, stage, latest_record)
    decision = StrategyReselectionDecision(
        decision_id=f"strategy_reselect:{(latest_record or {}).get('response_id') or 'none'}:{len(goal_attainment_timeline)}",
        should_reselect=should_reselect,
        trigger=trigger,
        from_driver=latest_driver,
        recommended_strategy=recommended_strategy,
        recommended_goal_driver=recommended_driver,
        priority=priority,
        rationale=_rationale(trigger, latest_driver, recommended_strategy),
        constraints=_constraints(trigger, stage, recommended_strategy),
        evidence={
            "recent_statuses": statuses,
            "latest_completion_score": latest.get("completion_score"),
            "latest_entropy_delta": latest.get("entropy_delta"),
            "latest_risk_shift": latest.get("risk_shift"),
            "dialogue_stage": stage,
            "latest_risk_level": (latest_record or {}).get("risk_level"),
            "latest_entropy_score": (latest_record or {}).get("entropy_score"),
        },
    )
    return {
        "decision_id": decision.decision_id,
        "should_reselect": decision.should_reselect,
        "trigger": decision.trigger,
        "from_driver": decision.from_driver,
        "recommended_strategy": decision.recommended_strategy,
        "recommended_goal_driver": decision.recommended_goal_driver,
        "priority": decision.priority,
        "rationale": decision.rationale,
        "constraints": decision.constraints,
        "evidence": decision.evidence,
    }


def enrich_student_context_with_strategy_reselection(
    student_context: dict[str, Any],
    strategy_reselection: dict[str, Any] | None,
) -> dict[str, Any]:
    if not strategy_reselection or not strategy_reselection.get("should_reselect"):
        return dict(student_context or {})
    enriched = dict(student_context or {})
    enriched["strategy_reselection"] = {
        "recommended_strategy": strategy_reselection.get("recommended_strategy"),
        "recommended_goal_driver": strategy_reselection.get("recommended_goal_driver"),
        "trigger": strategy_reselection.get("trigger"),
        "priority": strategy_reselection.get("priority"),
        "constraints": strategy_reselection.get("constraints") or [],
        "instruction": (
            "The previous strategy may not be working. In the next reply, change the support move "
            "according to recommended_strategy and constraints. Do not repeat the same intervention."
        ),
    }
    return enriched


def _trigger(statuses: list[str], stage: str, latest_record: dict[str, Any] | None) -> str:
    if stage == "safety_priority" or (latest_record or {}).get("risk_level") == "critical":
        return "safety_priority"
    if statuses and statuses[-1] == "worsened":
        return "latest_worsened"
    if statuses and statuses[-1] == "ineffective":
        return "latest_ineffective"
    if len(statuses) >= 2 and all(status in FAILURE_STATUSES for status in statuses[-2:]):
        return "repeated_failure"
    if len(statuses) >= 3 and all(status in SOFT_FAILURE_STATUSES for status in statuses[-3:]):
        return "stuck_without_progress"
    return "none"


def _recommend_strategy(
    *,
    trigger: str,
    latest_driver: str | None,
    stage: str,
    latest_record: dict[str, Any] | None,
) -> tuple[str, str]:
    if trigger == "safety_priority":
        return "safety_first_human_linkage", "risk_pressure"
    if trigger in {"latest_worsened", "repeated_failure"}:
        if latest_driver == "cognitive_load":
            return "reduce_task_scope_and_validate_emotion", "emotion_intensity"
        if latest_driver == "physiological_imbalance":
            return "switch_to_body_stabilization_then_human_support", "risk_pressure"
        if latest_driver == "privacy_boundary_tension" or stage == "boundary_building":
            return "repair_trust_and_stop_detail_questions", "privacy_boundary_tension"
        return "lower_pressure_and_reassess_driver", "emotion_intensity"
    if trigger == "latest_ineffective":
        return "repair_response_style_before_advice", latest_driver or "general_uncertainty"
    if trigger == "stuck_without_progress":
        if latest_driver == "social_support_tension":
            return "move_from_reflection_to_boundary_action", "social_support_tension"
        return "change_micro_intervention_format", latest_driver or "general_uncertainty"
    return "maintain_current_strategy", latest_driver or (latest_record or {}).get("reduction_goal_active_driver") or "general_uncertainty"


def _priority(trigger: str, stage: str, latest_record: dict[str, Any] | None) -> str:
    if trigger == "safety_priority":
        return "critical"
    if trigger in {"latest_worsened", "repeated_failure"}:
        return "high"
    if trigger in {"latest_ineffective", "stuck_without_progress"}:
        return "medium"
    risk_level = (latest_record or {}).get("risk_level")
    if risk_level == "high":
        return "high"
    return "low"


def _rationale(trigger: str, latest_driver: str | None, recommended_strategy: str) -> str:
    if trigger == "none":
        return "当前目标完成度没有显示明显失败，暂不需要重选策略。"
    if trigger == "safety_priority":
        return "安全风险优先级高于普通熵减目标，需要切换到现实支持连接。"
    if trigger == "latest_worsened":
        return f"上一轮围绕 {latest_driver or 'unknown'} 的目标后状态恶化，需要更换策略为 {recommended_strategy}。"
    if trigger == "repeated_failure":
        return f"连续目标未达成，说明当前微干预可能不匹配，需要切换策略为 {recommended_strategy}。"
    if trigger == "latest_ineffective":
        return "用户反馈或目标评估显示上一轮可能无效，下一轮先修复回复方式再给建议。"
    if trigger == "stuck_without_progress":
        return "多轮维持但没有改善，需要改变干预形式，避免原地打转。"
    return "根据目标完成度切换下一轮策略。"


def _constraints(trigger: str, stage: str, recommended_strategy: str) -> list[str]:
    constraints = ["不要重复上一轮干预", "不要直接展示后台策略名称"]
    if trigger in {"latest_worsened", "repeated_failure", "safety_priority"}:
        constraints.append("降低提问压力")
        constraints.append("优先稳定情绪或现实安全")
    if "repair" in recommended_strategy:
        constraints.append("先承认刚才可能没有接住用户")
    if stage == "boundary_building":
        constraints.append("不要追问隐私细节")
    if recommended_strategy == "change_micro_intervention_format":
        constraints.append("把建议换成选择题或一个动作，不要继续长篇解释")
    return list(dict.fromkeys(constraints))
