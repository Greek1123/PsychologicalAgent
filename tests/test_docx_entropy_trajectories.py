from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_docx_entropy_trajectories import summarize_all, summarize_case


def test_summarize_case_reports_entropy_delta_and_strategy_sequence() -> None:
    case = {
        "turns": [
            {"entropy_score": 40, "risk_level": "low", "strategy_id": "supportive_listening", "dynamic_state": "stable"},
            {"entropy_score": 52, "risk_level": "medium", "strategy_id": "grounding_small_step", "dynamic_state": "escalating_entropy"},
        ]
    }

    summary = summarize_case(case)

    assert summary["start_score"] == 40
    assert summary["end_score"] == 52
    assert summary["delta"] == 12
    assert summary["peak_score"] == 52
    assert summary["risk_counts"] == {"low": 1, "medium": 1}
    assert summary["strategy_sequence"] == ["supportive_listening", "grounding_small_step"]


def test_summarize_all_groups_entropy_by_category() -> None:
    cases = [
        {
            "turns": [
                {
                    "category": "学业任务与科研压力",
                    "entropy_score": 30,
                    "risk_level": "low",
                    "balance_state": "stable",
                    "strategy_id": "supportive_listening",
                },
                {
                    "category": "学业任务与科研压力",
                    "entropy_score": 50,
                    "risk_level": "medium",
                    "balance_state": "strained",
                    "strategy_id": "grounding_small_step",
                },
            ]
        }
    ]

    summary = summarize_all(cases)

    assert summary["cases"] == 1
    assert summary["turns"] == 2
    assert summary["average_entropy"] == 40
    assert summary["risk_counts"] == {"low": 1, "medium": 1}
    assert summary["by_category"]["学业任务与科研压力"]["max_entropy"] == 50
