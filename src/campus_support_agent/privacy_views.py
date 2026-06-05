from __future__ import annotations

from copy import deepcopy
from typing import Any

from .privacy_redaction import redact_private_identifiers_in_value


SUPPORTED_VIEW_ROLES = {"student", "counselor", "research", "admin"}


def normalize_view_role(role: str | None) -> str:
    normalized = (role or "student").strip().lower()
    if normalized not in SUPPORTED_VIEW_ROLES:
        raise ValueError(f"Unsupported view role: {role}")
    return normalized


def project_support_response_for_role(response: dict[str, Any], role: str) -> dict[str, Any]:
    role = normalize_view_role(role)
    if role == "admin":
        return deepcopy(response)
    if role == "student":
        return _student_response_view(response)
    if role == "research":
        return _research_response_view(response)
    return _counselor_response_view(response)


def project_session_analysis_for_role(analysis: dict[str, Any], role: str) -> dict[str, Any]:
    role = normalize_view_role(role)
    if role == "admin":
        return deepcopy(analysis)
    if role == "student":
        return _student_analysis_view(analysis)
    if role == "research":
        return _research_analysis_view(analysis)
    return _counselor_analysis_view(analysis)


def project_care_queue_for_role(queue: dict[str, Any], role: str) -> dict[str, Any]:
    role = normalize_view_role(role)
    if role == "admin":
        return deepcopy(queue)
    projected = deepcopy(queue)
    items = projected.get("items") or []
    if not isinstance(items, list):
        projected["items"] = []
        return projected

    redacted_items = []
    total_summary: dict[str, int] = {}
    total_redactions = 0
    for item in items:
        item_view = _research_care_queue_item(item) if role == "research" else _counselor_care_queue_item(item)
        redacted_items.append(item_view)
        summary = item_view.get("privacy_redaction") or {}
        total_redactions += int(summary.get("total_redactions") or 0)
        for category, count in (summary.get("categories") or {}).items():
            total_summary[category] = total_summary.get(category, 0) + int(count)
    projected["items"] = redacted_items
    projected["privacy_redaction"] = {
        "total_redactions": total_redactions,
        "categories": dict(sorted(total_summary.items())),
    }
    projected["role"] = role
    return projected


def _student_response_view(response: dict[str, Any]) -> dict[str, Any]:
    safety = response.get("safety") or {}
    entropy = response.get("entropy") or {}
    entropy_reduction = response.get("entropy_reduction") or {}
    risk = response.get("risk") or {}
    reply_text, redaction_summary = redact_private_identifiers_in_value(response.get("reply_text") or "")
    return {
        "response_id": response.get("response_id"),
        "reply_text": reply_text,
        "risk": {
            "level": risk.get("level"),
            "needs_human_followup": bool(risk.get("needs_human_followup")),
        },
        "entropy": {
            "balance_state": entropy.get("balance_state"),
            "trend": entropy.get("trend"),
        },
        "entropy_reduction": {
            "target_state": entropy_reduction.get("target_state"),
            "core_actions": entropy_reduction.get("core_actions") or [],
            "review_window_hours": entropy_reduction.get("review_window_hours"),
        },
        "campus_resources": response.get("campus_resources") or [],
        "safety": {
            "emergency_notice": safety.get("emergency_notice"),
            "human_referral": safety.get("human_referral"),
        },
        "session": response.get("session"),
        "privacy_redaction": redaction_summary.as_dict(),
    }


def _student_analysis_view(analysis: dict[str, Any]) -> dict[str, Any]:
    latest_reply_text, redaction_summary = redact_private_identifiers_in_value(analysis.get("latest_reply_text") or "")
    return {
        "session_id": analysis.get("session_id"),
        "total_responses": analysis.get("total_responses"),
        "latest_reply_text": latest_reply_text,
        "trend_warning": _pick(
            analysis.get("trend_warning") or {},
            ["level", "trend_state", "review_window_hours", "user_visible_mode"],
        ),
        "session_care_plan": _pick(
            analysis.get("session_care_plan") or {},
            ["care_phase", "user_visible_focus", "next_actions", "success_indicators"],
        ),
        "entropy_reduction_outcome": _pick(
            analysis.get("entropy_reduction_outcome") or {},
            ["status", "summary", "next_action"],
        ),
        "privacy_redaction": redaction_summary.as_dict(),
    }


def _counselor_response_view(response: dict[str, Any]) -> dict[str, Any]:
    projected = _remove_sensitive_backend_fields(response)
    redacted, summary = redact_private_identifiers_in_value(projected)
    redacted["privacy_redaction"] = summary.as_dict()
    return redacted


def _counselor_analysis_view(analysis: dict[str, Any]) -> dict[str, Any]:
    projected = _remove_sensitive_backend_fields(analysis)
    redacted, summary = redact_private_identifiers_in_value(projected)
    redacted["privacy_redaction"] = summary.as_dict()
    return redacted


def _research_response_view(response: dict[str, Any]) -> dict[str, Any]:
    projected = _remove_sensitive_backend_fields(response)
    for key in ("input_text", "transcript", "reply_text", "safety", "campus_resources"):
        projected.pop(key, None)
    projected["text_redacted"] = True
    redacted, summary = redact_private_identifiers_in_value(projected)
    redacted["privacy_redaction"] = summary.as_dict()
    return redacted


def _research_analysis_view(analysis: dict[str, Any]) -> dict[str, Any]:
    projected = _remove_sensitive_backend_fields(analysis)
    for key in (
        "latest_reply_text",
        "conversation_memory",
        "referral_events",
        "intervention_feedback",
        "human_interventions",
        "latest_human_intervention",
    ):
        projected.pop(key, None)
    projected["text_redacted"] = True
    redacted, summary = redact_private_identifiers_in_value(projected)
    redacted["privacy_redaction"] = summary.as_dict()
    return redacted


def _counselor_care_queue_item(item: dict[str, Any]) -> dict[str, Any]:
    redacted, summary = redact_private_identifiers_in_value(item)
    redacted["privacy_redaction"] = summary.as_dict()
    return redacted


def _research_care_queue_item(item: dict[str, Any]) -> dict[str, Any]:
    projected = deepcopy(item)
    intervention = ((projected.get("evidence") or {}).get("human_intervention") or {})
    if isinstance(intervention, dict):
        intervention.pop("note", None)
        intervention.pop("next_action", None)
    redacted, summary = redact_private_identifiers_in_value(projected)
    redacted["privacy_redaction"] = summary.as_dict()
    return redacted


def _remove_sensitive_backend_fields(value: Any) -> Any:
    if isinstance(value, list):
        return [_remove_sensitive_backend_fields(item) for item in value]
    if not isinstance(value, dict):
        return deepcopy(value)

    sensitive_keys = {
        "hidden_clinical_goal",
        "backend_reason",
        "backend_actions",
        "system_flags",
        "student_context",
        "conversation_history",
    }
    return {
        key: _remove_sensitive_backend_fields(item)
        for key, item in value.items()
        if key not in sensitive_keys
    }


def _pick(source: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: source.get(key) for key in keys if key in source}
