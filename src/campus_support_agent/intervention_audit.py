from __future__ import annotations

from typing import Any

from .schemas import InterventionAuditLog


def build_intervention_audit_log(record: dict[str, Any]) -> dict[str, Any]:
    risk = _risk_snapshot(record)
    entropy = _entropy_snapshot(record)
    goal = record.get("reduction_goal") or {}
    strategy = _strategy_snapshot(record)
    referral = record.get("referral_explanation") or {}
    route = _decision_route(record, referral)
    trace = _backend_trace(record, route, goal, referral)

    audit = InterventionAuditLog(
        audit_id=f"audit:{record.get('response_id') or 'unknown'}",
        response_id=record.get("response_id"),
        session_id=record.get("session_id"),
        created_at=record.get("created_at"),
        decision_route=route,
        risk_snapshot=risk,
        entropy_snapshot=entropy,
        selected_goal=goal,
        strategy_snapshot=strategy,
        referral_snapshot=referral,
        explanation=_explanation(route, risk, entropy, goal, referral),
        backend_trace=trace,
        user_visible_summary=_user_visible_summary(record),
        evidence={
            "input_text": record.get("input_text"),
            "local_policy_name": record.get("local_policy_name"),
            "dynamic_action": record.get("dynamic_action"),
            "feedback_adaptation_mode": record.get("feedback_adaptation_mode"),
            "orchestration_route": record.get("orchestration_route"),
            "reduction_goal_active_driver": record.get("reduction_goal_active_driver"),
            "referral_explanation_level": record.get("referral_explanation_level"),
            "adjustment_loop_action": record.get("adjustment_loop_action"),
            "adjustment_loop_priority": record.get("adjustment_loop_priority"),
        },
    )
    return {
        "audit_id": audit.audit_id,
        "response_id": audit.response_id,
        "session_id": audit.session_id,
        "created_at": audit.created_at,
        "decision_route": audit.decision_route,
        "risk_snapshot": audit.risk_snapshot,
        "entropy_snapshot": audit.entropy_snapshot,
        "selected_goal": audit.selected_goal,
        "strategy_snapshot": audit.strategy_snapshot,
        "referral_snapshot": audit.referral_snapshot,
        "explanation": audit.explanation,
        "backend_trace": audit.backend_trace,
        "user_visible_summary": audit.user_visible_summary,
        "evidence": audit.evidence,
    }


def build_intervention_audit_timeline(records: list[dict[str, Any]], *, limit: int = 20) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    return [build_intervention_audit_log(record) for record in records[-limit:]]


def summarize_intervention_audits(audit_logs: list[dict[str, Any]]) -> dict[str, Any]:
    route_counts: dict[str, int] = {}
    referral_counts: dict[str, int] = {}
    goal_driver_counts: dict[str, int] = {}
    for item in audit_logs:
        route = str(item.get("decision_route") or "unknown")
        route_counts[route] = route_counts.get(route, 0) + 1
        referral = (item.get("referral_snapshot") or {}).get("referral_level") or "none"
        referral_counts[referral] = referral_counts.get(referral, 0) + 1
        driver = (item.get("selected_goal") or {}).get("active_driver") or "unknown"
        goal_driver_counts[driver] = goal_driver_counts.get(driver, 0) + 1
    latest = audit_logs[-1] if audit_logs else None
    return {
        "total_audits": len(audit_logs),
        "decision_routes": route_counts,
        "referral_levels": referral_counts,
        "goal_drivers": goal_driver_counts,
        "latest_decision_route": latest.get("decision_route") if latest else None,
        "latest_explanation": latest.get("explanation") if latest else None,
    }


def _risk_snapshot(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "level": record.get("risk_level"),
        "score": record.get("risk_score"),
        "referral_should_refer": record.get("referral_should_refer"),
        "referral_urgency": record.get("referral_urgency"),
    }


def _entropy_snapshot(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "score": record.get("entropy_score"),
        "level": record.get("entropy_level"),
        "balance_state": record.get("balance_state"),
        "target_delta": record.get("reduction_goal_target_delta"),
    }


def _strategy_snapshot(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "strategy_id": record.get("strategy_id"),
        "strategy_priority": record.get("strategy_priority"),
        "response_mode": record.get("response_mode"),
        "dynamic_action": record.get("dynamic_action"),
        "feedback_adaptation_mode": record.get("feedback_adaptation_mode"),
        "orchestration_route": record.get("orchestration_route"),
        "local_policy_name": record.get("local_policy_name"),
        "adjustment_loop_action": record.get("adjustment_loop_action"),
        "adjustment_loop_priority": record.get("adjustment_loop_priority"),
    }


def _decision_route(record: dict[str, Any], referral: dict[str, Any]) -> str:
    referral_level = referral.get("referral_level") or record.get("referral_explanation_level")
    if referral_level == "emergency":
        return "emergency_referral"
    if referral_level in {"professional_followup", "campus_support_watch"}:
        return "human_support_recommended"
    if record.get("orchestration_route"):
        return str(record["orchestration_route"])
    if record.get("adjustment_loop_action"):
        return f"loop:{record['adjustment_loop_action']}"
    if record.get("dynamic_action"):
        return f"dynamic:{record['dynamic_action']}"
    if record.get("local_policy_name"):
        return f"local_policy:{record['local_policy_name']}"
    return "general_support"


def _backend_trace(
    record: dict[str, Any],
    route: str,
    goal: dict[str, Any],
    referral: dict[str, Any],
) -> list[str]:
    trace = [
        f"risk={record.get('risk_level')}:{record.get('risk_score')}",
        f"entropy={record.get('entropy_score')}:{record.get('balance_state')}",
        f"route={route}",
    ]
    if goal:
        trace.append(f"goal={goal.get('active_driver')}:{goal.get('priority')}")
    if record.get("dynamic_action"):
        trace.append(f"dynamic_action={record.get('dynamic_action')}")
    if record.get("feedback_adaptation_mode"):
        trace.append(f"feedback_mode={record.get('feedback_adaptation_mode')}")
    if record.get("adjustment_loop_action"):
        trace.append(f"loop_action={record.get('adjustment_loop_action')}:{record.get('adjustment_loop_priority')}")
    if referral:
        trace.append(f"referral={referral.get('referral_level')}:{referral.get('recommended_channel')}")
    return trace


def _explanation(
    route: str,
    risk: dict[str, Any],
    entropy: dict[str, Any],
    goal: dict[str, Any],
    referral: dict[str, Any],
) -> str:
    if referral.get("should_escalate"):
        return (
            f"Escalation was selected because risk={risk.get('level')} and entropy={entropy.get('score')}; "
            f"referral_level={referral.get('referral_level')} under route={route}."
        )
    if goal:
        return (
            f"The selected reduction goal targets driver={goal.get('active_driver')} with goal="
            f"{goal.get('reduction_goal')}; route={route}."
        )
    return f"General support route={route}; no escalation or explicit reduction goal was selected."


def _user_visible_summary(record: dict[str, Any]) -> str | None:
    reply = str(record.get("reply_text") or "").strip()
    if not reply:
        return None
    return reply[:160]
