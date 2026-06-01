from __future__ import annotations

from collections import Counter
from typing import Any


def build_processing_consistency_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    timeline = build_processing_consistency_timeline(records)
    return {
        "summary": summarize_processing_consistency(timeline),
        "timeline": timeline,
    }


def build_processing_consistency_timeline(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        summary = record.get("processing_summary") or {}
        route = str(summary.get("route") or record.get("processing_route") or "")
        safety_priority = str(summary.get("safety_priority") or record.get("processing_safety_priority") or "")
        next_action = str(summary.get("next_backend_action") or record.get("processing_next_backend_action") or "")
        risk_level = str(summary.get("risk_level") or record.get("risk_level") or "")
        reply_source = str(summary.get("reply_source") or record.get("processing_reply_source") or "")
        should_refer = bool(summary.get("should_refer") or record.get("referral_should_refer"))
        referral_urgency = str(summary.get("referral_urgency") or record.get("referral_urgency") or "")
        issues = _detect_consistency_issues(
            has_summary=bool(summary),
            route=route,
            safety_priority=safety_priority,
            next_action=next_action,
            risk_level=risk_level,
            reply_source=reply_source,
            should_refer=should_refer,
            referral_urgency=referral_urgency,
        )
        timeline.append(
            {
                "turn_index": index,
                "response_id": record.get("response_id"),
                "created_at": record.get("created_at"),
                "risk_level": risk_level or None,
                "route": route or None,
                "safety_priority": safety_priority or None,
                "next_backend_action": next_action or None,
                "reply_source": reply_source or None,
                "should_refer": should_refer,
                "referral_urgency": referral_urgency or None,
                "consistent": not issues,
                "issues": issues,
            }
        )
    return timeline


def summarize_processing_consistency(timeline: list[dict[str, Any]]) -> dict[str, Any]:
    issue_counter: Counter[str] = Counter()
    inconsistent_turns = 0
    for item in timeline:
        issues = [str(issue) for issue in item.get("issues") or []]
        if issues:
            inconsistent_turns += 1
        issue_counter.update(issues)
    latest = timeline[-1] if timeline else {}
    latest_issues = [str(issue) for issue in latest.get("issues") or []]
    return {
        "total_turns": len(timeline),
        "consistent_turns": max(len(timeline) - inconsistent_turns, 0),
        "inconsistent_turns": inconsistent_turns,
        "status": "ok" if inconsistent_turns == 0 else "needs_review",
        "latest_consistent": not latest_issues if timeline else True,
        "latest_issues": latest_issues,
        "issue_counts": dict(issue_counter),
        "top_issues": [
            {"issue": issue, "count": count}
            for issue, count in issue_counter.most_common(8)
        ],
    }


def _detect_consistency_issues(
    *,
    has_summary: bool,
    route: str,
    safety_priority: str,
    next_action: str,
    risk_level: str,
    reply_source: str,
    should_refer: bool,
    referral_urgency: str,
) -> list[str]:
    issues: list[str] = []
    if not has_summary:
        issues.append("missing_processing_summary")

    if risk_level in {"high", "critical"} and route != "crisis_safety":
        issues.append("high_risk_not_crisis_route")
    if risk_level == "critical" and safety_priority != "urgent":
        issues.append("critical_risk_not_urgent")
    if route == "crisis_safety" and safety_priority != "urgent":
        issues.append("crisis_route_not_urgent")
    if safety_priority == "urgent" and next_action != "activate_urgent_handoff":
        issues.append("urgent_without_handoff_action")
    if route == "crisis_safety" and reply_source and reply_source != "crisis_template":
        issues.append("crisis_route_non_crisis_reply_source")
    if should_refer and next_action == "continue_supportive_monitoring":
        issues.append("referral_marked_but_monitoring_only")
    if referral_urgency == "urgent" and safety_priority != "urgent":
        issues.append("urgent_referral_not_urgent_priority")
    if route == "local_policy" and safety_priority == "urgent":
        issues.append("local_policy_marked_urgent")

    return _dedupe(issues)


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item not in result:
            result.append(item)
    return result
