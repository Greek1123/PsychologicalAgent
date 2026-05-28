from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.demo_workspace import (
    DEMO_SCENARIOS,
    build_demo_workspace_report,
    summarize_demo_turn,
)


def test_demo_scenarios_cover_low_and_human_followup_cases() -> None:
    assert len(DEMO_SCENARIOS) >= 6
    assert any(not scenario.get("human_status") for scenario in DEMO_SCENARIOS)
    assert any(scenario.get("human_status") == "escalated" for scenario in DEMO_SCENARIOS)


def test_summarize_demo_turn_extracts_backend_state() -> None:
    response = {
        "response_id": "support-1",
        "reply_text": "reply",
        "risk": {"level": "high", "score": 80},
        "entropy": {"score": 72, "balance_state": "fragile"},
        "intervention_strategy": {"strategy_id": "safety_first"},
        "dynamic_adjustment": {"action": "urgent_referral"},
        "referral_decision": {"should_refer": True, "urgency": "urgent"},
    }

    turn = summarize_demo_turn(response, "user", 1)

    assert turn["response_id"] == "support-1"
    assert turn["risk_level"] == "high"
    assert turn["entropy_score"] == 72
    assert turn["strategy_id"] == "safety_first"
    assert turn["should_refer"] is True


def test_build_demo_workspace_report_renders_summary() -> None:
    report = build_demo_workspace_report(
        scenario_results=[
            {
                "title": "危险地点与危机优先",
                "session_id": "demo-danger",
                "human_intervention": {"status": "escalated", "handler_id": "counselor"},
                "turns": [
                    {
                        "turn_index": 1,
                        "user_text": "我在楼顶",
                        "reply_text": "先离开危险地点。",
                        "risk_level": "critical",
                        "risk_score": 100,
                        "entropy_score": 90,
                        "balance_state": "crisis",
                        "strategy_id": "safety_first",
                        "should_refer": True,
                        "referral_urgency": "urgent",
                    }
                ],
            }
        ],
        care_queue={
            "total_items": 1,
            "priority_counts": {"critical": 1},
            "route_counts": {"urgent_safety": 1},
            "outcome_counts": {"crisis_priority": 1},
        },
        generated_at="2026-05-28T12:00:00",
    )

    assert "# 演示工作台样例报告" in report
    assert "危险地点与危机优先" in report
    assert "critical" in report
    assert "Care Queue 摘要" in report
