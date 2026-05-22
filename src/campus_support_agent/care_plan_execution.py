from __future__ import annotations

from typing import Any

from .schemas import SessionCarePlan, SupportPlan


def apply_session_care_plan_to_plan(
    plan: SupportPlan,
    *,
    session_care_plan: SessionCarePlan | dict[str, Any] | None,
) -> SupportPlan:
    """Use the hidden care plan as the final backend guardrail for the visible reply."""

    if not session_care_plan:
        return plan
    care_plan = _to_dict(session_care_plan)
    phase = str(care_plan.get("care_phase") or "support")
    primary_goal = str(care_plan.get("primary_goal") or "")
    user_focus = str(care_plan.get("user_visible_focus") or "")
    next_actions = {str(item) for item in care_plan.get("next_actions") or []}
    avoid_actions = {str(item) for item in care_plan.get("avoid_actions") or []}

    if phase == "safety":
        return _safety_phase(plan)
    if phase == "human_followup":
        return _human_followup_phase(plan)
    if phase == "repair":
        return _repair_phase(plan, primary_goal=primary_goal, user_focus=user_focus)
    if phase == "monitor":
        return _monitor_phase(plan, next_actions=next_actions, avoid_actions=avoid_actions)
    return _support_phase(plan, primary_goal=primary_goal, user_focus=user_focus)


def _safety_phase(plan: SupportPlan) -> SupportPlan:
    plan.summary = "我先把重点放在你的安全上。现在不用急着解释完整发生了什么，我们先确认你不是一个人硬扛。"
    plan.immediate_support = [
        "如果你此刻有伤害自己或失控的冲动，请先离开危险物品，去到有人在的地方，或立刻联系身边可信的人。",
        *plan.immediate_support,
    ][:3]
    plan.campus_actions = [
        "如果风险正在发生，请优先联系学校心理中心、辅导员、宿管或当地紧急求助渠道。",
        *plan.campus_actions,
    ][:3]
    plan.follow_up = ["你可以只回我一句：现在身边有没有一个能联系到的人？"]
    return _compact_plan(plan, max_follow_up=1)


def _human_followup_phase(plan: SupportPlan) -> SupportPlan:
    plan.summary = "这件事已经不适合只靠你一个人撑着了。找现实里的支持不是麻烦别人，而是给自己多一层保护。"
    plan.immediate_support = [
        "今晚先不用把所有问题讲清楚，可以先选一个最容易联系的人，说一句“我现在状态不太好，能不能陪我一下”。",
        *plan.immediate_support,
    ][:3]
    plan.campus_actions = [
        "如果这种状态持续影响睡眠、学习或安全，建议尽快联系辅导员或学校心理中心做一次线下支持。",
        *plan.campus_actions,
    ][:3]
    plan.follow_up = ["如果你愿意，我们可以先一起选一个最不费力的求助对象。"]
    return _compact_plan(plan, max_follow_up=1)


def _repair_phase(plan: SupportPlan, *, primary_goal: str, user_focus: str) -> SupportPlan:
    del primary_goal, user_focus
    plan.summary = "刚才如果我没有接住你的意思，我们先把节奏放慢。你不需要证明自己为什么难受，我会先按你现在的感受来。"
    plan.immediate_support = [
        "你可以不细说原因，也可以只告诉我现在更需要“被陪着”还是“一个很小的建议”。",
        *plan.immediate_support,
    ][:2]
    plan.follow_up = ["不用展开也可以，你回“陪着”或“建议”就行。"]
    return _compact_plan(plan, max_follow_up=1)


def _monitor_phase(
    plan: SupportPlan,
    *,
    next_actions: set[str],
    avoid_actions: set[str],
) -> SupportPlan:
    del next_actions
    plan.summary = "我听到你现在的负担还在，我们先不把问题扩大，只处理接下来一小段时间。"
    plan.immediate_support = [
        "先选一个最小动作：喝几口水、坐到床边、把手机放下两分钟，或者只把下一件事写成一句话。",
        *plan.immediate_support,
    ][:2]
    if "do_not_overload_with_long_plan" in avoid_actions:
        plan.campus_actions = plan.campus_actions[:1]
        plan.self_regulation = plan.self_regulation[:1]
    plan.follow_up = ["你现在更想先缓一下，还是想把今晚最烦的一件事拆小一点？"]
    return _compact_plan(plan, max_follow_up=1)


def _support_phase(plan: SupportPlan, *, primary_goal: str, user_focus: str) -> SupportPlan:
    if primary_goal == "build_privacy_trust_before_problem_solving":
        plan.summary = "你担心别人知道，这一点很重要。你可以先不用讲细节，我们先把边界和安全感放在前面。"
        plan.immediate_support = [
            "在这个对话里，我会尽量按你愿意透露的程度来回应；你不用说姓名、宿舍号或具体对象，也可以只讲感受。",
            *plan.immediate_support,
        ][:2]
        plan.follow_up = ["如果你不想展开，也完全可以先停在这里；如果愿意，只说“我现在最怕的是被知道”还是“我现在最烦的是宿舍氛围”就可以。"]
        return _compact_plan(plan, max_follow_up=1)

    if user_focus:
        plan.self_regulation = [user_focus, *plan.self_regulation][:2]
    if not plan.summary.strip():
        plan.summary = "我会先顺着你现在说到的部分陪你梳理，不急着下结论。"
    plan.follow_up = plan.follow_up[:1] or ["你可以继续说一点点，我会跟着你的节奏来。"]
    return _compact_plan(plan, max_follow_up=1)


def _compact_plan(plan: SupportPlan, *, max_follow_up: int) -> SupportPlan:
    plan.immediate_support = _dedupe(plan.immediate_support)[:3]
    plan.campus_actions = _dedupe(plan.campus_actions)[:2]
    plan.self_regulation = _dedupe(plan.self_regulation)[:2]
    plan.follow_up = _dedupe(plan.follow_up)[:max_follow_up]
    return plan


def _to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dataclass_fields__"):
        return {field: getattr(value, field) for field in value.__dataclass_fields__}
    return {}


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        clean = str(item).strip()
        if clean and clean not in result:
            result.append(clean)
    return result
