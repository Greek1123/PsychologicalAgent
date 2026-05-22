from __future__ import annotations

from collections import Counter
from typing import Any


GUARDRAIL_PATCH_ISSUES = {
    "empty_reply",
    "mojibake_or_encoding_artifact",
    "backend_terms_exposed",
    "assistant_role_drift",
    "numeric_pattern_continuation",
}

BEHAVIOR_SFT_ISSUES = {
    "too_short_for_distress",
    "generic_advice_under_distress",
    "repeated_previous_reply",
    "pushes_after_privacy_boundary",
}

DPO_ISSUES = {
    "pushes_after_privacy_boundary",
    "generic_advice_under_distress",
    "repeated_previous_reply",
}

SAFETY_REVIEW_RISK_LEVELS = {"high", "critical"}


def build_quality_refinement_plan(
    bad_cases: list[dict[str, Any]],
    *,
    max_examples_per_bucket: int = 8,
) -> dict[str, Any]:
    buckets = {
        "guardrail_patch": [],
        "behavior_sft": [],
        "preference_dpo": [],
        "safety_review": [],
        "manual_review": [],
    }
    issue_counts: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()

    for case in bad_cases:
        issues = _issues(case)
        issue_counts.update(issues)
        routes = _routes_for_case(case, issues)
        route_counts.update(routes)
        compact_case = _compact_case(case, issues, routes)
        for route in routes:
            if len(buckets[route]) < max_examples_per_bucket:
                buckets[route].append(compact_case)

    return {
        "total_bad_cases": len(bad_cases),
        "issue_counts": dict(issue_counts),
        "route_counts": dict(route_counts),
        "recommended_order": _recommended_order(route_counts),
        "buckets": buckets,
        "next_actions": _next_actions(route_counts),
    }


def _routes_for_case(case: dict[str, Any], issues: list[str]) -> list[str]:
    routes: list[str] = []
    issue_set = set(issues)
    risk_level = str((case.get("risk") or {}).get("level") or "").lower()

    if risk_level in SAFETY_REVIEW_RISK_LEVELS:
        routes.append("safety_review")
    if issue_set & GUARDRAIL_PATCH_ISSUES:
        routes.append("guardrail_patch")
    if issue_set & BEHAVIOR_SFT_ISSUES:
        routes.append("behavior_sft")
    if issue_set & DPO_ISSUES:
        routes.append("preference_dpo")
    if not routes:
        routes.append("manual_review")
    return _dedupe(routes)


def _recommended_order(route_counts: Counter[str]) -> list[str]:
    fixed_priority = [
        "safety_review",
        "guardrail_patch",
        "behavior_sft",
        "preference_dpo",
        "manual_review",
    ]
    return [
        route
        for route in fixed_priority
        if route_counts.get(route, 0) > 0
    ]


def _next_actions(route_counts: Counter[str]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if route_counts.get("safety_review", 0):
        actions.append(
            {
                "route": "safety_review",
                "priority": "critical",
                "action": "Review high-risk conversations before using them for training.",
            }
        )
    if route_counts.get("guardrail_patch", 0):
        actions.append(
            {
                "route": "guardrail_patch",
                "priority": "high",
                "action": "Patch backend final-reply guardrails before collecting more model samples.",
            }
        )
    if route_counts.get("behavior_sft", 0):
        actions.append(
            {
                "route": "behavior_sft",
                "priority": "medium",
                "action": "Rewrite chosen replies and convert reviewed cases into behavior SFT rows.",
            }
        )
    if route_counts.get("preference_dpo", 0):
        actions.append(
            {
                "route": "preference_dpo",
                "priority": "medium",
                "action": "Keep rejected replies and add stronger chosen replies for DPO/KTO preference tuning.",
            }
        )
    if route_counts.get("manual_review", 0):
        actions.append(
            {
                "route": "manual_review",
                "priority": "low",
                "action": "Inspect unclassified samples and decide whether to add new quality rules.",
            }
        )
    return actions


def _compact_case(case: dict[str, Any], issues: list[str], routes: list[str]) -> dict[str, Any]:
    return {
        "id": case.get("id"),
        "response_id": case.get("response_id"),
        "session_id": case.get("session_id"),
        "input_text": case.get("input_text"),
        "assistant_reply_preview": str(case.get("assistant_reply") or "")[:220],
        "issues": issues,
        "routes": routes,
        "quality_score": (case.get("quality_review") or {}).get("quality_score"),
        "risk_level": (case.get("risk") or {}).get("level"),
        "entropy_score": (case.get("entropy") or {}).get("score"),
    }


def _issues(case: dict[str, Any]) -> list[str]:
    review = case.get("quality_review") or {}
    feedback = case.get("feedback") or {}
    failure = case.get("failure_review") or {}
    return _dedupe(
        [
            *[str(item) for item in review.get("issues") or []],
            *[str(item) for item in feedback.get("tags") or []],
            *[str(item) for item in failure.get("suspected_problem") or []],
        ]
    )


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item and item not in result:
            result.append(item)
    return result
