from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for path in (SCRIPTS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from campus_support_agent.storage import SQLiteSessionStore
from export_care_queue_snapshot import export_care_queue_snapshot


NOW = datetime(2026, 6, 6, 14, 0, 0, tzinfo=timezone.utc)


def test_export_care_queue_snapshot_writes_redacted_json_and_markdown(tmp_path: Path) -> None:
    db_path = tmp_path / "agent.db"
    _seed_queue_store(db_path)

    result = export_care_queue_snapshot(
        db_path=db_path,
        out_dir=tmp_path / "exports",
        role="counselor",
        include_low_priority=True,
        timestamp=NOW,
    )

    json_path = Path(result["json_path"])
    markdown_path = Path(result["markdown_path"])
    assert json_path.exists()
    assert markdown_path.exists()
    snapshot = json.loads(json_path.read_text(encoding="utf-8"))
    serialized = json.dumps(snapshot, ensure_ascii=False)

    assert result["total_items"] == 1
    assert snapshot["role"] == "counselor"
    assert snapshot["queue"]["items"][0]["session_id"] == "session-care"
    assert "13812345678" not in serialized
    assert snapshot["queue"]["privacy_redaction"]["categories"]["phone"] >= 1
    assert "| 1 | `session-care`" in markdown_path.read_text(encoding="utf-8")


def test_export_care_queue_snapshot_research_view_removes_intervention_note(tmp_path: Path) -> None:
    db_path = tmp_path / "agent.db"
    _seed_queue_store(db_path)

    result = export_care_queue_snapshot(
        db_path=db_path,
        out_dir=tmp_path / "exports",
        role="research",
        include_low_priority=True,
        timestamp=NOW,
    )

    snapshot = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
    intervention = snapshot["queue"]["items"][0]["evidence"]["human_intervention"]
    assert "note" not in intervention
    assert "next_action" not in intervention
    assert snapshot["queue"]["role"] == "research"


def _seed_queue_store(db_path: Path) -> None:
    store = SQLiteSessionStore(str(db_path), max_messages=6)
    response = {
        "response_id": "resp-care",
        "reply_text": "I will help you reduce tonight's pressure first.",
        "risk": {"level": "high", "score": 70, "should_refer": True, "manual_referral_recommended": True},
        "entropy": {"score": 32, "level": 3, "balance_state": "fragile"},
        "state_profile": {"primary_state": "sleep_pressure", "intensity": "high", "recommended_focus": "stabilize"},
        "intervention_strategy": {"strategy_id": "safety_first", "priority": "high", "response_mode": "grounding"},
        "dynamic_adjustment": {"stability_state": "fragile", "action": "human_followup_watch", "should_refer": True},
        "feedback_adaptation": {"mode": "none", "question_pressure": "low"},
        "entropy_orchestration": {"route": "safety", "next_focus": "support", "user_visible_goal": "stabilize tonight"},
        "reduction_goal": {"active_driver": "sleep", "reduction_goal": "lower arousal", "priority": "high"},
        "referral_explanation": {"referral_level": "recommended", "recommended_channel": "counseling", "should_escalate": True, "urgency": "urgent"},
        "adjustment_loop": {"loop_action": "monitor", "priority": "high", "next_reply_mode": "support", "question_policy": "low"},
        "local_policy": {"policy_name": "manual_followup", "policy_stage": "escalation_watch"},
        "referral_decision": {"should_refer": True, "urgency": "urgent"},
        "processing_summary": {"route": "local_policy", "safety_priority": "human_followup", "reply_source": "mock", "next_backend_action": "queue_human_followup"},
    }
    store.store_support_response(
        session_id="session-care",
        response_id="resp-care",
        source="text",
        input_text="My phone is 13812345678 and I cannot sleep.",
        transcript=None,
        student_context={},
        conversation_history=[],
        response_payload=response,
    )
    store.append_human_intervention(
        session_id="session-care",
        response_id="resp-care",
        status="acknowledged",
        handler_id="counselor-001",
        note="Student phone 13812345678 needs follow-up tonight.",
        next_action="call 13812345678",
        tags=["manual_followup"],
    )
