from __future__ import annotations

from typing import Any

from .schemas import SupportPlan


def build_intervention_next_step(
    *,
    session_id: str | None,
    intervention_effectiveness: dict[str, Any] | None = None,
    care_plan: dict[str, Any] | None = None,
    reply_quality_summary: dict[str, Any] | None = None,
    trend_warning: dict[str, Any] | None = None,
    latest_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    effectiveness = intervention_effectiveness or {}
    summary = effectiveness.get("summary") or {}
    latest = effectiveness.get("latest_effectiveness") or {}
    care_plan = care_plan or {}
    reply_quality_summary = reply_quality_summary or {}
    trend_warning = trend_warning or {}
    latest_record = latest_record or {}

    status = str(summary.get("overall_status") or latest.get("status") or "baseline")
    care_phase = str(care_plan.get("care_phase") or "support")
    warning_level = str(trend_warning.get("level") or "none")
    quality_signal = str(summary.get("reply_quality_signal") or _reply_quality_signal(reply_quality_summary))
    feedback_signal = str(summary.get("feedback_signal") or "none")

    decision = _base_decision(session_id=session_id, status=status)
    _apply_care_phase(decision, care_phase)
    _apply_warning(decision, warning_level)
    _apply_quality_signal(decision, quality_signal)
    _apply_feedback_signal(decision, feedback_signal)
    _apply_latest_record(decision, latest_record)
    decision["constraints"] = _dedupe(decision["constraints"])
    decision["preferred_moves"] = _dedupe(decision["preferred_moves"])
    decision["avoid_moves"] = _dedupe(decision["avoid_moves"])
    decision["evidence"] = {
        "overall_status": status,
        "latest_effectiveness_score": summary.get("latest_effectiveness_score"),
        "latest_recommended_next_action": summary.get("latest_recommended_next_action"),
        "care_phase": care_phase,
        "warning_level": warning_level,
        "quality_signal": quality_signal,
        "feedback_signal": feedback_signal,
        "latest_response_id": latest_record.get("response_id"),
    }
    return decision


def enrich_student_context_with_intervention_next_step(
    student_context: dict[str, Any],
    next_step: dict[str, Any] | None,
) -> dict[str, Any]:
    enriched = dict(student_context or {})
    if not next_step:
        return enriched
    enriched["intervention_next_step"] = {
        "next_reply_mode": next_step.get("next_reply_mode"),
        "question_policy": next_step.get("question_policy"),
        "human_followup_policy": next_step.get("human_followup_policy"),
        "should_collect_bad_case": next_step.get("should_collect_bad_case"),
        "should_pause_advice": next_step.get("should_pause_advice"),
        "preferred_moves": next_step.get("preferred_moves") or [],
        "avoid_moves": next_step.get("avoid_moves") or [],
        "constraints": next_step.get("constraints") or [],
        "instruction": (
            "Use this hidden next-step decision to shape the next reply. Do not expose status, "
            "scores, care phase, backend policies, or internal labels to the user."
        ),
    }
    return enriched


def apply_intervention_next_step_to_plan(
    plan: SupportPlan,
    *,
    next_step: dict[str, Any] | None,
) -> SupportPlan:
    if not next_step:
        return plan
    mode = str(next_step.get("next_reply_mode") or "")
    question_policy = str(next_step.get("question_policy") or "")
    human_policy = str(next_step.get("human_followup_policy") or "")
    should_pause_advice = bool(next_step.get("should_pause_advice"))

    if mode == "safety_grounding":
        plan.summary = _ZH["safety_summary"]
        plan.immediate_support = _prepend(plan.immediate_support, _ZH["safety_immediate"])[:2]
        plan.campus_actions = _prepend(plan.campus_actions, _ZH["safety_campus"])[:2]
        plan.self_regulation = plan.self_regulation[:1]
        plan.follow_up = [_ZH["safety_follow_up"]]
        return plan

    if mode == "warm_human_linkage" or human_policy in {"recommended", "urgent"}:
        plan.summary = _ZH["human_summary"]
        plan.campus_actions = _prepend(plan.campus_actions, _ZH["human_campus"])[:2]
        plan.follow_up = [_ZH["human_follow_up"]]

    if mode == "repair_conversation":
        plan.summary = _ZH["repair_summary"]
        plan.immediate_support = _prepend(plan.immediate_support, _ZH["repair_immediate"])[:2]
        plan.follow_up = [_ZH["repair_follow_up"]]

    if mode == "stabilize_then_reduce_load":
        plan.summary = _ZH["stabilize_summary"]
        plan.immediate_support = _prepend(plan.immediate_support, _ZH["stabilize_immediate"])[:2]
        plan.self_regulation = _prepend(plan.self_regulation, _ZH["stabilize_self"])[:1]
        plan.campus_actions = plan.campus_actions[:1]
        plan.follow_up = [_ZH["stabilize_follow_up"]]

    if mode == "consolidate_progress":
        plan.follow_up = _prepend(plan.follow_up, _ZH["consolidate_follow_up"])[:1]

    if should_pause_advice:
        plan.self_regulation = plan.self_regulation[:1]
        plan.campus_actions = plan.campus_actions[:1]

    if question_policy in {"low_pressure_or_no_question", "safety_check_only"}:
        plan.follow_up = [_ZH["low_pressure_follow_up"]]
    elif question_policy in {"one_contextual_question", "one_low_pressure_question", "one_grounding_choice"}:
        plan.follow_up = _prepend(plan.follow_up, _ZH["one_question_follow_up"])[:1]

    return _compact_plan(plan)


def _base_decision(*, session_id: str | None, status: str) -> dict[str, Any]:
    decision = {
        "decision_id": f"{session_id or 'anonymous'}:next_step:{status}",
        "status": status,
        "priority": "low",
        "next_reply_mode": "supportive_contextual",
        "question_policy": "one_optional_question",
        "human_followup_policy": "none",
        "should_collect_bad_case": False,
        "should_pause_advice": False,
        "review_window_hours": 72,
        "preferred_moves": ["Reflect the user's latest concrete situation before giving suggestions."],
        "avoid_moves": ["Do not expose backend analysis or entropy labels."],
        "constraints": ["Keep the visible reply warm, concrete, and non-technical."],
    }
    if status == "crisis_priority":
        decision.update(
            {
                "priority": "critical",
                "next_reply_mode": "safety_grounding",
                "question_policy": "safety_check_only",
                "human_followup_policy": "urgent",
                "should_pause_advice": True,
                "review_window_hours": 1,
            }
        )
        decision["preferred_moves"].extend(["Focus on immediate safety.", "Offer real-world emergency support paths."])
        decision["avoid_moves"].extend(["Do not debate causes.", "Do not give long self-help plans."])
    elif status == "needs_human_followup":
        decision.update(
            {
                "priority": "high",
                "next_reply_mode": "warm_human_linkage",
                "question_policy": "one_low_pressure_question",
                "human_followup_policy": "recommended",
                "review_window_hours": 12,
            }
        )
        decision["preferred_moves"].extend(["Normalize asking a real person for support.", "Offer one reachable campus path."])
    elif status == "deteriorating":
        decision.update(
            {
                "priority": "high",
                "next_reply_mode": "stabilize_then_reduce_load",
                "question_policy": "one_grounding_choice",
                "review_window_hours": 6,
            }
        )
        decision["preferred_moves"].extend(["Reduce pressure first.", "Give one tiny next step only."])
        decision["avoid_moves"].append("Do not add a full plan.")
    elif status == "needs_repair":
        decision.update(
            {
                "priority": "medium",
                "next_reply_mode": "repair_conversation",
                "question_policy": "low_pressure_or_no_question",
                "should_collect_bad_case": True,
                "review_window_hours": 24,
            }
        )
        decision["preferred_moves"].extend(["Acknowledge the previous miss briefly.", "Return control to the user."])
        decision["avoid_moves"].extend(["Do not defend the previous reply.", "Do not ask for many details."])
    elif status in {"effective", "improving"}:
        decision.update(
            {
                "priority": "low",
                "next_reply_mode": "consolidate_progress",
                "question_policy": "one_optional_question",
                "review_window_hours": 72,
            }
        )
        decision["preferred_moves"].append("Notice the small improvement without over-celebrating.")
    elif status == "baseline":
        decision.update(
            {
                "next_reply_mode": "collect_context_gently",
                "question_policy": "one_contextual_question",
            }
        )
    return decision


def _apply_care_phase(decision: dict[str, Any], care_phase: str) -> None:
    if care_phase == "safety":
        decision["priority"] = "critical"
        decision["next_reply_mode"] = "safety_grounding"
        decision["human_followup_policy"] = "urgent"
        decision["should_pause_advice"] = True
    elif care_phase == "human_followup":
        if decision["priority"] not in {"critical"}:
            decision["priority"] = "high"
        decision["human_followup_policy"] = "recommended"
        decision["preferred_moves"].append("Frame human support as a practical option, not a failure.")
    elif care_phase == "repair":
        decision["should_collect_bad_case"] = True
        decision["next_reply_mode"] = "repair_conversation"
        decision["question_policy"] = "low_pressure_or_no_question"
    elif care_phase == "monitor":
        decision["review_window_hours"] = min(int(decision["review_window_hours"]), 24)


def _apply_warning(decision: dict[str, Any], warning_level: str) -> None:
    if warning_level == "critical":
        decision["priority"] = "critical"
        decision["human_followup_policy"] = "urgent"
        decision["should_pause_advice"] = True
        decision["review_window_hours"] = 1
    elif warning_level == "high":
        if decision["priority"] not in {"critical"}:
            decision["priority"] = "high"
        decision["review_window_hours"] = min(int(decision["review_window_hours"]), 6)
    elif warning_level == "medium":
        if decision["priority"] == "low":
            decision["priority"] = "medium"
        decision["review_window_hours"] = min(int(decision["review_window_hours"]), 24)


def _apply_quality_signal(decision: dict[str, Any], quality_signal: str) -> None:
    if quality_signal == "poor":
        decision["should_collect_bad_case"] = True
        decision["next_reply_mode"] = "repair_conversation"
        decision["question_policy"] = "low_pressure_or_no_question"
        decision["preferred_moves"].append("Use a fresh, context-specific opening.")
        decision["avoid_moves"].append("Do not reuse template comfort.")
    elif quality_signal == "mixed":
        decision["preferred_moves"].append("Make the next reply more concrete than the previous one.")


def _apply_feedback_signal(decision: dict[str, Any], feedback_signal: str) -> None:
    if feedback_signal == "negative":
        decision["should_collect_bad_case"] = True
        decision["next_reply_mode"] = "repair_conversation"
        decision["avoid_moves"].append("Do not repeat the move that received negative feedback.")
    elif feedback_signal == "positive":
        decision["preferred_moves"].append("Keep the working support style stable.")


def _apply_latest_record(decision: dict[str, Any], latest_record: dict[str, Any]) -> None:
    if latest_record.get("adjustment_loop_action") == "safety_first":
        decision["priority"] = "critical"
        decision["human_followup_policy"] = "urgent"
    if latest_record.get("feedback_adaptation_mode") == "repair_next_turn":
        decision["should_collect_bad_case"] = True
        decision["next_reply_mode"] = "repair_conversation"


def _reply_quality_signal(summary: dict[str, Any]) -> str:
    total = int(summary.get("total_replies") or 0)
    needs_review = int(summary.get("needs_review") or 0)
    if total <= 0:
        return "unknown"
    ratio = needs_review / total
    if ratio >= 0.4:
        return "poor"
    if ratio >= 0.15:
        return "mixed"
    return "clean"


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item and item not in result:
            result.append(item)
    return result


def _prepend(items: list[str], item: str) -> list[str]:
    return _dedupe([item, *items])


def _compact_plan(plan: SupportPlan) -> SupportPlan:
    plan.immediate_support = _dedupe(plan.immediate_support)[:3]
    plan.campus_actions = _dedupe(plan.campus_actions)[:2]
    plan.self_regulation = _dedupe(plan.self_regulation)[:2]
    plan.follow_up = _dedupe(plan.follow_up)[:1]
    return plan


_ZH = {
    "safety_summary": "\u6211\u4eec\u5148\u4e0d\u6025\u7740\u5206\u6790\u539f\u56e0\uff0c\u73b0\u5728\u6700\u91cd\u8981\u7684\u662f\u8ba9\u4f60\u5148\u5b89\u5168\u4e00\u70b9\u3001\u522b\u4e00\u4e2a\u4eba\u786c\u625b\u3002",
    "safety_immediate": "\u5982\u679c\u4f60\u6b64\u523b\u6709\u4f24\u5bb3\u81ea\u5df1\u6216\u522b\u4eba\u7684\u51b2\u52a8\uff0c\u8bf7\u5148\u8054\u7cfb\u8eab\u8fb9\u53ef\u4fe1\u4efb\u7684\u4eba\uff0c\u6216\u76f4\u63a5\u62e8\u6253\u5f53\u5730\u7d27\u6025\u7535\u8bdd\u3002",
    "safety_campus": "\u5982\u679c\u4f60\u5728\u5b66\u6821\uff0c\u53ef\u4ee5\u5c3d\u5feb\u8054\u7cfb\u8f85\u5bfc\u5458\u3001\u5bbf\u7ba1\u6216\u5b66\u6821\u5fc3\u7406\u652f\u6301\u8d44\u6e90\uff0c\u5148\u8ba9\u4e00\u4e2a\u771f\u5b9e\u7684\u4eba\u77e5\u9053\u4f60\u73b0\u5728\u4e0d\u592a\u597d\u3002",
    "safety_follow_up": "\u4f60\u73b0\u5728\u53ea\u9700\u8981\u56de\u6211\u4e00\u4ef6\u4e8b\uff1a\u4f60\u6b64\u523b\u662f\u5426\u8fd8\u5b89\u5168\uff1f",
    "human_summary": "\u8fd9\u4ef6\u4e8b\u5df2\u7ecf\u4e0d\u53ea\u662f\u666e\u901a\u7684\u5fc3\u60c5\u4e0d\u597d\uff0c\u66f4\u50cf\u662f\u9700\u8981\u591a\u4e00\u4e2a\u73b0\u5b9e\u4e2d\u7684\u652f\u6491\u70b9\u3002",
    "human_campus": "\u4f60\u53ef\u4ee5\u5148\u627e\u4e00\u4e2a\u76f8\u5bf9\u5bb9\u6613\u5f00\u53e3\u7684\u4eba\uff0c\u6bd4\u5982\u670b\u53cb\u3001\u8f85\u5bfc\u5458\u6216\u5fc3\u7406\u4e2d\u5fc3\uff0c\u4e0d\u5fc5\u4e00\u6b21\u8bb2\u5b8c\u5168\u90e8\u3002",
    "human_follow_up": "\u5982\u679c\u4f60\u613f\u610f\uff0c\u6211\u4eec\u53ef\u4ee5\u5148\u60f3\u4e00\u53e5\u5f88\u77ed\u7684\u6c42\u52a9\u5f00\u573a\u767d\u3002",
    "repair_summary": "\u6211\u521a\u624d\u53ef\u80fd\u6ca1\u6709\u771f\u6b63\u63a5\u4f4f\u4f60\u7684\u70b9\uff0c\u8fd9\u91cc\u5148\u4e0d\u6025\u7740\u7ed9\u5efa\u8bae\u3002",
    "repair_immediate": "\u4f60\u4e0d\u9700\u8981\u628a\u4e8b\u60c5\u8bb2\u5f97\u5f88\u5b8c\u6574\uff0c\u6211\u4eec\u53ef\u4ee5\u53ea\u5148\u505c\u5728\u4f60\u73b0\u5728\u6700\u96be\u53d7\u7684\u90a3\u4e00\u70b9\u3002",
    "repair_follow_up": "\u4f60\u53ef\u4ee5\u53ea\u56de\u6211\uff1a\u4f60\u73b0\u5728\u66f4\u9700\u8981\u6211\u966a\u7740\uff0c\u8fd8\u662f\u7ed9\u4e00\u4e2a\u5f88\u5c0f\u7684\u529e\u6cd5\uff1f",
    "stabilize_summary": "\u4f60\u73b0\u5728\u7684\u538b\u529b\u50cf\u662f\u5df2\u7ecf\u5806\u5230\u5f88\u6ee1\uff0c\u5148\u7a33\u4f4f\u5f53\u4e0b\uff0c\u6bd4\u7acb\u523b\u89e3\u51b3\u5168\u90e8\u95ee\u9898\u66f4\u91cd\u8981\u3002",
    "stabilize_immediate": "\u5148\u628a\u63a5\u4e0b\u6765\u7684\u4e8b\u7f29\u5230\u4e00\u4e2a\u5f88\u5c0f\u7684\u52a8\u4f5c\uff1a\u559d\u51e0\u53e3\u6c34\u3001\u5750\u7a33\uff0c\u7136\u540e\u53ea\u5904\u7406\u6700\u8fd1\u7684\u4e00\u4ef6\u4e8b\u3002",
    "stabilize_self": "\u8bd5\u7740\u505a\u4e00\u8f6e\u6162\u547c\u5438\uff1a\u5438\u6c14 4 \u62cd\uff0c\u505c 2 \u62cd\uff0c\u547c\u6c14 6 \u62cd\uff0c\u5148\u505a\u4e09\u6b21\u5c31\u591f\u3002",
    "stabilize_follow_up": "\u5982\u679c\u53ea\u9009\u4e00\u4ef6\u4e8b\u5148\u5904\u7406\uff0c\u4f60\u89c9\u5f97\u662f\u7761\u7720\u3001\u8003\u8bd5\uff0c\u8fd8\u662f\u5bbf\u820d\u91cc\u7684\u538b\u529b\uff1f",
    "consolidate_follow_up": "\u521a\u624d\u8fd9\u4e2a\u65b9\u5411\u5982\u679c\u5bf9\u4f60\u7a0d\u5fae\u6709\u7528\uff0c\u6211\u4eec\u53ef\u4ee5\u5148\u6cbf\u7740\u5b83\u8d70\uff0c\u4e0d\u6025\u7740\u6362\u592a\u591a\u65b9\u6cd5\u3002",
    "low_pressure_follow_up": "\u4f60\u4e0d\u60f3\u8bf4\u4e5f\u6ca1\u5173\u7cfb\uff0c\u6211\u4f1a\u5c0a\u91cd\u4f60\u7684\u8282\u594f\u3002",
    "one_question_follow_up": "\u5982\u679c\u613f\u610f\uff0c\u4f60\u53ea\u9700\u8981\u9009\u4e00\u4e2a\u6700\u8d34\u8fd1\u7684\u8bcd\uff1a\u5bb3\u6015\u3001\u59d4\u5c48\u3001\u70e6\uff0c\u8fd8\u662f\u7d2f\uff1f",
}
