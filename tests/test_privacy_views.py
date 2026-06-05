from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.privacy_views import (
    project_session_analysis_for_role,
    project_support_response_for_role,
)
from campus_support_agent.privacy_redaction import redact_private_identifiers


def test_student_response_view_hides_backend_fields() -> None:
    response = {
        "response_id": "support-1",
        "input_text": "raw student text",
        "reply_text": "visible reply",
        "risk": {"level": "medium", "score": 45, "needs_human_followup": True},
        "entropy": {"score": 55, "balance_state": "fragile", "trend": {"delta": 5}},
        "entropy_reduction": {
            "target_state": "stable",
            "core_actions": ["one small step"],
            "review_window_hours": 24,
        },
        "intervention_strategy": {"hidden_clinical_goal": "backend only"},
        "referral_explanation": {"backend_reason": "internal reason"},
        "care_pathway": {"backend_actions": ["notify_staff"]},
        "system_flags": {"reasons": ["internal_flag"]},
    }

    projected = project_support_response_for_role(response, "student")

    assert projected["reply_text"] == "visible reply"
    assert projected["risk"] == {"level": "medium", "needs_human_followup": True}
    assert "input_text" not in projected
    assert "intervention_strategy" not in projected
    assert "referral_explanation" not in projected
    assert "system_flags" not in projected


def test_research_view_redacts_text_and_direct_intervention_notes() -> None:
    analysis = {
        "session_id": "session-1",
        "latest_reply_text": "student-visible text",
        "risk_levels": {"high": 1},
        "human_interventions": [{"note": "called student", "handler_id": "counselor-1"}],
        "latest_human_intervention": {"note": "called student"},
        "conversation_memory": {"topics": ["private"]},
        "strategy_layer_summary": {"should_consider_human_followup": True},
    }

    projected = project_session_analysis_for_role(analysis, "research")

    assert projected["text_redacted"] is True
    assert projected["risk_levels"] == {"high": 1}
    assert "latest_reply_text" not in projected
    assert "human_interventions" not in projected
    assert "latest_human_intervention" not in projected
    assert "conversation_memory" not in projected


def test_private_identifier_redaction_masks_common_contact_fields() -> None:
    redacted, summary = redact_private_identifiers(
        "我的手机号是13812345678，邮箱是me@example.com，微信abc12345，QQ 1234567，学号 2024012345。"
    )

    assert "13812345678" not in redacted
    assert "me@example.com" not in redacted
    assert "abc12345" not in redacted
    assert "1234567" not in redacted
    assert "2024012345" not in redacted
    assert "[手机号]" in redacted
    assert "[邮箱]" in redacted
    assert summary.total_redactions == 5
    assert summary.categories["phone"] == 1
    assert summary.categories["email"] == 1


def test_counselor_view_keeps_clinical_context_but_masks_direct_identifiers() -> None:
    response = {
        "response_id": "support-2",
        "input_text": "我叫小张，电话13812345678，邮箱me@example.com，我最近睡不着。",
        "reply_text": "先不用在这里重复电话13812345678，我们先处理睡眠压力。",
        "risk": {"level": "medium"},
        "state_profile": {"primary_state": "sleep_pressure"},
        "system_flags": {"manual_review": True},
        "student_context": {"student_name": "小张"},
    }

    projected = project_support_response_for_role(response, "counselor")

    assert projected["risk"]["level"] == "medium"
    assert projected["state_profile"]["primary_state"] == "sleep_pressure"
    assert "system_flags" not in projected
    assert "student_context" not in projected
    assert "13812345678" not in projected["input_text"]
    assert "me@example.com" not in projected["input_text"]
    assert "[手机号]" in projected["input_text"]
    assert projected["privacy_redaction"]["total_redactions"] >= 2


def test_admin_view_keeps_raw_private_identifiers_for_authorized_audit() -> None:
    response = {
        "response_id": "support-3",
        "input_text": "电话13812345678",
        "reply_text": "已收到。",
        "system_flags": {"manual_review": True},
    }

    projected = project_support_response_for_role(response, "admin")

    assert projected["input_text"] == "电话13812345678"
    assert projected["system_flags"]["manual_review"] is True
    assert "privacy_redaction" not in projected
