from __future__ import annotations

from typing import Any

from .schemas import SupportPlan


def build_strategy_version_decision(
    *,
    session_id: str,
    session_tracking: dict[str, Any] | None = None,
    strategy_reselection: dict[str, Any] | None = None,
    intervention_effectiveness: dict[str, Any] | None = None,
    intervention_next_step: dict[str, Any] | None = None,
    latest_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tracking = session_tracking or {}
    reselection = strategy_reselection or {}
    effectiveness = intervention_effectiveness or {}
    summary = effectiveness.get("summary") or {}
    next_step = intervention_next_step or {}
    latest_record = latest_record or {}

    current_family = _strategy_family(latest_record, tracking, next_step)
    decision = _decision(
        tracking_stage=str(tracking.get("stage") or ""),
        reselection_trigger=str(reselection.get("trigger") or "none"),
        should_reselect=bool(reselection.get("should_reselect")),
        effectiveness_status=str(summary.get("overall_status") or ""),
        next_reply_mode=str(next_step.get("next_reply_mode") or ""),
    )
    target_family = _target_family(
        decision=decision,
        current_family=current_family,
        reselection=reselection,
        next_step=next_step,
        tracking=tracking,
    )
    version = {
        "version_id": f"{session_id}:strategy:{target_family}:{latest_record.get('response_id') or 'new'}",
        "session_id": session_id,
        "decision": decision,
        "current_strategy_family": current_family,
        "target_strategy_family": target_family,
        "should_switch_strategy": decision in {"repair", "revise", "stabilize", "escalate"},
        "reply_contract": _reply_contract(target_family, decision, tracking),
        "avoid_contract": _avoid_contract(target_family, tracking, next_step),
        "review_after_turns": _review_after_turns(decision),
        "evidence": {
            "tracking_stage": tracking.get("stage"),
            "reselection_trigger": reselection.get("trigger"),
            "reselection_strategy": reselection.get("recommended_strategy"),
            "effectiveness_status": summary.get("overall_status"),
            "latest_effectiveness_score": summary.get("latest_effectiveness_score"),
            "next_reply_mode": next_step.get("next_reply_mode"),
            "latest_response_id": latest_record.get("response_id"),
        },
    }
    version["instruction"] = _instruction(version)
    return version


def enrich_student_context_with_strategy_version(
    student_context: dict[str, Any],
    strategy_version: dict[str, Any] | None,
) -> dict[str, Any]:
    enriched = dict(student_context or {})
    if not strategy_version:
        return enriched
    enriched["strategy_version"] = {
        "decision": strategy_version.get("decision"),
        "target_strategy_family": strategy_version.get("target_strategy_family"),
        "should_switch_strategy": strategy_version.get("should_switch_strategy"),
        "reply_contract": strategy_version.get("reply_contract") or [],
        "avoid_contract": strategy_version.get("avoid_contract") or [],
        "review_after_turns": strategy_version.get("review_after_turns"),
        "instruction": (
            "Use this hidden strategy-version decision to keep or switch the intervention style. "
            "Do not expose strategy version, contracts, or backend labels to the user."
        ),
    }
    return enriched


def apply_strategy_version_to_plan(
    plan: SupportPlan,
    *,
    strategy_version: dict[str, Any] | None,
) -> SupportPlan:
    if not strategy_version:
        return plan

    decision = str(strategy_version.get("decision") or "")
    family = str(strategy_version.get("target_strategy_family") or "")
    avoid = {str(item) for item in strategy_version.get("avoid_contract") or []}

    if decision == "repair" or family == "trust_repair":
        plan.summary = _ZH["repair_summary"]
        plan.immediate_support = _prepend(plan.immediate_support, _ZH["repair_support"])[:3]
        plan.follow_up = [_ZH["repair_follow_up"]]
    elif decision == "stabilize" or family == "stabilization":
        plan.summary = _ZH["stabilize_summary"]
        plan.immediate_support = _prepend(plan.immediate_support, _ZH["stabilize_support"])[:3]
        plan.self_regulation = _prepend(plan.self_regulation, _ZH["stabilize_self"])[:2]
        plan.follow_up = [_ZH["stabilize_follow_up"]]
    elif decision == "escalate" or family == "human_linkage":
        plan.summary = _ZH["human_summary"]
        plan.campus_actions = _prepend(plan.campus_actions, _ZH["human_action"])[:2]
        plan.follow_up = [_ZH["human_follow_up"]]
    elif decision == "revise":
        plan.immediate_support = _prepend(plan.immediate_support, _ZH["revise_support"])[:3]
        plan.follow_up = [_ZH["revise_follow_up"]]

    if "avoid_long_plans" in avoid:
        plan.campus_actions = plan.campus_actions[:1]
        plan.self_regulation = plan.self_regulation[:1]
    if "avoid_question_pressure" in avoid:
        plan.follow_up = [_ZH["low_pressure_follow_up"]]

    return _compact_plan(plan)


def _strategy_family(
    latest_record: dict[str, Any],
    tracking: dict[str, Any],
    next_step: dict[str, Any],
) -> str:
    route = str(latest_record.get("orchestration_route") or "")
    mode = str(next_step.get("next_reply_mode") or "")
    stage = str(tracking.get("stage") or "")
    if route in {"safety_first", "human_support_linkage"} or mode in {"safety_grounding", "warm_human_linkage"}:
        return "human_linkage"
    if route == "repair_conversation" or mode == "repair_conversation" or stage == "repair":
        return "trust_repair"
    if route == "stabilize_and_reduce_load" or mode == "stabilize_then_reduce_load" or stage == "stabilize":
        return "stabilization"
    if route == "boundary_respecting_support":
        return "boundary_support"
    if route == "explore_and_clarify":
        return "gentle_exploration"
    return "supportive_continuity"


def _decision(
    *,
    tracking_stage: str,
    reselection_trigger: str,
    should_reselect: bool,
    effectiveness_status: str,
    next_reply_mode: str,
) -> str:
    if next_reply_mode in {"safety_grounding", "warm_human_linkage"} or effectiveness_status in {
        "crisis_priority",
        "needs_human_followup",
    }:
        return "escalate"
    if tracking_stage == "repair" or effectiveness_status == "needs_repair" or reselection_trigger == "latest_ineffective":
        return "repair"
    if next_reply_mode == "stabilize_then_reduce_load" or effectiveness_status == "deteriorating":
        return "stabilize"
    if should_reselect or reselection_trigger not in {"", "none"}:
        return "revise"
    return "continue"


def _target_family(
    *,
    decision: str,
    current_family: str,
    reselection: dict[str, Any],
    next_step: dict[str, Any],
    tracking: dict[str, Any],
) -> str:
    recommended = str(reselection.get("recommended_strategy") or "")
    mode = str(next_step.get("next_reply_mode") or "")
    privacy = str(tracking.get("privacy_boundary") or "")
    if decision == "escalate":
        return "human_linkage"
    if decision == "repair" or "repair" in recommended:
        return "trust_repair"
    if decision == "stabilize" or mode == "stabilize_then_reduce_load":
        return "stabilization"
    if privacy in {"needs_explicit_reassurance", "low_disclosure_preferred"}:
        return "boundary_support"
    if "boundary" in recommended:
        return "boundary_support"
    if "body" in recommended:
        return "stabilization"
    if decision == "revise":
        return "gentle_exploration" if current_family != "gentle_exploration" else "supportive_continuity"
    return current_family


def _reply_contract(target_family: str, decision: str, tracking: dict[str, Any]) -> list[str]:
    contract = ["acknowledge_latest_user_turn", "keep_language_plain", "hide_backend_analysis"]
    if target_family == "trust_repair":
        contract.extend(["briefly_repair_miss", "return_control_to_user"])
    elif target_family == "stabilization":
        contract.extend(["reduce_immediate_arousal", "offer_one_small_step"])
    elif target_family == "human_linkage":
        contract.extend(["name_real_world_support", "avoid_making_user_feel_abandoned"])
    elif target_family == "boundary_support":
        contract.extend(["reassure_privacy_boundary", "avoid_detail_pressure"])
    elif target_family == "gentle_exploration":
        contract.append("ask_at_most_one_contextual_question")
    if decision == "continue" and str(tracking.get("stage") or "") == "continuity":
        contract.append("connect_to_previous_turn_without_repeating")
    return _dedupe(contract)


def _avoid_contract(
    target_family: str,
    tracking: dict[str, Any],
    next_step: dict[str, Any],
) -> list[str]:
    avoid = ["avoid_template_comfort", "avoid_role_drift", "avoid_exposing_entropy_or_risk_labels"]
    failed = {str(item) for item in tracking.get("failed_moves") or []}
    if "avoid_question_pressure" in failed or target_family in {"trust_repair", "boundary_support"}:
        avoid.append("avoid_question_pressure")
    if bool(next_step.get("should_pause_advice")) or target_family in {"stabilization", "human_linkage"}:
        avoid.append("avoid_long_plans")
    if target_family == "trust_repair":
        avoid.append("avoid_defending_previous_reply")
    return _dedupe(avoid)


def _review_after_turns(decision: str) -> int:
    if decision == "escalate":
        return 1
    if decision in {"repair", "stabilize", "revise"}:
        return 2
    return 3


def _instruction(version: dict[str, Any]) -> str:
    return (
        f"Strategy decision={version['decision']}; target_family={version['target_strategy_family']}. "
        "Shape the next reply by the reply_contract and avoid_contract. Keep all labels hidden."
    )


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
    "repair_summary": "\u6211\u5148\u4e0d\u6025\u7740\u63a8\u8fdb\u65b0\u5efa\u8bae\uff0c\u56e0\u4e3a\u521a\u624d\u5982\u679c\u6ca1\u6709\u63a5\u4f4f\u4f60\uff0c\u5148\u628a\u8fd9\u4e00\u70b9\u4fee\u56de\u6765\u66f4\u91cd\u8981\u3002",
    "repair_support": "\u4f60\u53ef\u4ee5\u4e0d\u6309\u6211\u7684\u95ee\u9898\u6765\u56de\u7b54\uff0c\u6211\u4f1a\u5148\u8ddf\u7740\u4f60\u6b64\u523b\u6700\u660e\u663e\u7684\u611f\u53d7\u8d70\u3002",
    "repair_follow_up": "\u6211\u4eec\u5148\u4e0d\u8ffd\u95ee\u7ec6\u8282\uff0c\u4f60\u53ea\u8981\u56de\u6211\u4e00\u4e2a\u8bcd\u4e5f\u53ef\u4ee5\uff1a\u70e6\u3001\u6015\u3001\u59d4\u5c48\uff0c\u6216\u8005\u4e0d\u60f3\u8bf4\u3002",
    "stabilize_summary": "\u73b0\u5728\u5148\u4e0d\u628a\u6240\u6709\u4e8b\u60c5\u90fd\u62c6\u5f00\uff0c\u6211\u4eec\u5148\u628a\u5f53\u4e0b\u8fd9\u4e00\u6ce2\u538b\u529b\u964d\u4e0b\u6765\u4e00\u70b9\u3002",
    "stabilize_support": "\u5148\u505c\u5728\u6700\u8fd1\u7684\u4e00\u5c0f\u6b65\uff1a\u4e0d\u9700\u8981\u7acb\u523b\u60f3\u660e\u767d\u5168\u90e8\u539f\u56e0\uff0c\u53ea\u9700\u8981\u8ba9\u81ea\u5df1\u5148\u4e0d\u88ab\u538b\u529b\u5e26\u7740\u8dd1\u3002",
    "stabilize_self": "\u8bd5\u7740\u628a\u811a\u8e29\u5b9e\u5730\u9762\uff0c\u6162\u6162\u547c\u51fa\u4e00\u53e3\u6c14\uff0c\u7136\u540e\u770b\u770b\u5468\u56f4\u4e09\u4e2a\u4f60\u80fd\u770b\u5230\u7684\u4e1c\u897f\u3002",
    "stabilize_follow_up": "\u5982\u679c\u53ea\u9009\u4e00\u4ef6\u4e8b\u5148\u653e\u4e0b\u6216\u5148\u5904\u7406\uff0c\u4f60\u89c9\u5f97\u6700\u50cf\u662f\u54ea\u4e00\u4ef6\uff1f",
    "human_summary": "\u8fd9\u91cc\u53ef\u80fd\u9700\u8981\u4e00\u4e2a\u771f\u5b9e\u7684\u652f\u6491\u70b9\u548c\u4f60\u4e00\u8d77\u627f\u62c5\uff0c\u800c\u4e0d\u662f\u8ba9\u4f60\u4e00\u4e2a\u4eba\u786c\u6491\u3002",
    "human_action": "\u5982\u679c\u4f60\u5728\u5b66\u6821\uff0c\u53ef\u4ee5\u5148\u627e\u4e00\u4e2a\u76f8\u5bf9\u5bb9\u6613\u5f00\u53e3\u7684\u4eba\uff1a\u670b\u53cb\u3001\u8f85\u5bfc\u5458\u3001\u5bbf\u7ba1\u6216\u5fc3\u7406\u4e2d\u5fc3\uff0c\u4e0d\u5fc5\u4e00\u6b21\u8bb2\u5b8c\u3002",
    "human_follow_up": "\u5982\u679c\u4f60\u613f\u610f\uff0c\u6211\u53ef\u4ee5\u5e2e\u4f60\u628a\u6c42\u52a9\u5f00\u573a\u767d\u5199\u6210\u5f88\u77ed\u7684\u4e00\u53e5\u8bdd\u3002",
    "revise_support": "\u6211\u6362\u4e00\u79cd\u66f4\u5b9e\u9645\u7684\u65b9\u5f0f\u966a\u4f60\u770b\u8fd9\u4ef6\u4e8b\uff1a\u5148\u4e0d\u505a\u5927\u9053\u7406\uff0c\u53ea\u627e\u5f53\u4e0b\u6700\u5361\u4f60\u7684\u90a3\u4e00\u70b9\u3002",
    "revise_follow_up": "\u4f60\u73b0\u5728\u66f4\u50cf\u662f\u88ab\u54ea\u4e2a\u90e8\u5206\u5361\u4f4f\uff1a\u60c5\u7eea\u592a\u6ee1\u3001\u4e8b\u60c5\u592a\u591a\uff0c\u8fd8\u662f\u4e0d\u77e5\u9053\u627e\u8c01\u8bf4\uff1f",
    "low_pressure_follow_up": "\u4f60\u4e0d\u7528\u9a6c\u4e0a\u56de\u7b54\u95ee\u9898\uff0c\u53ea\u8981\u544a\u8bc9\u6211\u4f60\u60f3\u201c\u5148\u7f13\u7f13\u201d\u8fd8\u662f\u201c\u542c\u4e00\u4e2a\u529e\u6cd5\u201d\u5c31\u884c\u3002",
}
