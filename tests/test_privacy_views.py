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
