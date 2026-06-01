from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.acceptance_report import (
    ReportArtifacts,
    build_acceptance_report,
    summarize_delivery_artifacts,
    summarize_docx_eval,
    summarize_manual_check,
)
from campus_support_agent.config import Settings


def test_summarize_docx_eval_counts_scores_flags_and_low_examples(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    rows = [
        {
            "case_id": "01",
            "turns": [
                {"comparison": {"score": 80, "flags": []}},
                {"comparison": {"score": 60, "flags": ["weak_action_specificity"]}},
            ],
        }
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")

    summary = summarize_docx_eval(path)

    assert summary["cases"] == 1
    assert summary["turns"] == 2
    assert summary["average_score"] == 70
    assert summary["flag_counts"] == {"weak_action_specificity": 1}
    assert summary["low_score_count"] == 1


def test_summarize_manual_check_counts_statuses(tmp_path: Path) -> None:
    path = tmp_path / "manual.json"
    path.write_text(
        json.dumps(
            [
                {"scenario_id": "a", "check_status": "PASS"},
                {"scenario_id": "a", "check_status": "WARN"},
                {"scenario_id": "b", "check_status": "PASS"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = summarize_manual_check(path)

    assert summary["scenarios"] == 2
    assert summary["turns"] == 3
    assert summary["pass"] == 2
    assert summary["warn"] == 1


def test_acceptance_report_renders_key_sections(tmp_path: Path) -> None:
    kb_path = tmp_path / "campus_knowledge.json"
    kb_path.write_text("[]", encoding="utf-8")
    settings = Settings(
        llm_provider="mock",
        stt_provider="mock",
        database_path=str(tmp_path / "agent.db"),
        log_file_path=str(tmp_path / "app.log"),
        campus_kb_path=str(kb_path),
    )

    report = build_acceptance_report(
        artifacts=ReportArtifacts(
            docx_eval_jsonl=None,
            manual_check_json=None,
            test_summary="278 passed",
            frontend_client=tmp_path / "campusSupportApi.ts",
            frontend_example=tmp_path / "StudentChatExample.jsx",
            demo_playbook=tmp_path / "demo_acceptance_playbook.md",
            demo_checklist=tmp_path / "demo_acceptance_checklist.md",
        ),
        settings=settings,
        generated_at="2026-05-28T12:00:00",
    )

    assert "# 系统验收报告" in report
    assert "## 当前系统层级" in report
    assert "## 关键验收指标" in report
    assert "## 前端与演示交付物" in report
    assert "TypeScript API client" in report
    assert "演示验收手册" in report
    assert "278 passed" in report
    assert "部署自检：ready" in report


def test_summarize_delivery_artifacts_reports_ready_state(tmp_path: Path) -> None:
    frontend_client = tmp_path / "campusSupportApi.ts"
    frontend_example = tmp_path / "StudentChatExample.jsx"
    demo_playbook = tmp_path / "demo_acceptance_playbook.md"
    demo_checklist = tmp_path / "demo_acceptance_checklist.md"
    for path in [frontend_client, frontend_example, demo_playbook, demo_checklist]:
        path.write_text("ok", encoding="utf-8")

    summary = summarize_delivery_artifacts(
        ReportArtifacts(
            docx_eval_jsonl=None,
            manual_check_json=None,
            test_summary="not run",
            frontend_client=frontend_client,
            frontend_example=frontend_example,
            demo_playbook=demo_playbook,
            demo_checklist=demo_checklist,
        )
    )

    assert summary["frontend_ready"] == "ready"
    assert summary["demo_ready"] == "ready"
    assert all(item["status"] in {"present", "missing"} for item in summary["items"])
