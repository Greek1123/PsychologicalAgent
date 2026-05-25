from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_docx_backend_experiment import categorize_case, summarize_results


def test_categorize_case_detects_hidden_crisis() -> None:
    case = {"title": "把账号密码交给朋友", "observation_points": []}

    assert categorize_case(case) == "安全危机与隐性高危"


def test_summarize_results_counts_categories_and_flags() -> None:
    results = [
        {
            "case_id": "01",
            "title": "考试压力",
            "category": "学业任务与科研压力",
            "turns": [
                {"comparison": {"score": 80, "flags": []}, "turn_index": 1, "user": "我怕考试。"},
                {"comparison": {"score": 60, "flags": ["weak_action_specificity"]}, "turn_index": 2, "user": "怎么办？"},
            ],
        }
    ]

    summary = summarize_results(results)

    assert summary["cases"] == 1
    assert summary["turns"] == 2
    assert summary["average_score"] == 70
    assert summary["flag_counts"] == {"weak_action_specificity": 1}
    assert summary["by_category"]["学业任务与科研压力"]["average_score"] == 70
    assert len(summary["low_score_examples"]) == 1
