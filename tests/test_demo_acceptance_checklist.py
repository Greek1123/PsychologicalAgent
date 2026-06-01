from __future__ import annotations

from pathlib import Path

from scripts.generate_demo_acceptance_checklist import build_checklist


ROOT = Path(__file__).resolve().parents[1]


def test_demo_acceptance_checklist_contains_core_steps() -> None:
    checklist = build_checklist(ROOT)

    for text in [
        "演示验收清单",
        "scripts\\smoke_frontend_handoff.py",
        "考试焦虑",
        "隐私威胁",
        "危险地点",
        "care queue",
        "frontend_handoff/campusSupportApi.ts",
    ]:
        assert text in checklist


def test_demo_acceptance_playbook_exists_and_mentions_boundaries() -> None:
    playbook = (ROOT / "docs" / "demo_acceptance_playbook.md").read_text(encoding="utf-8")

    assert "学生端不应默认展示" in playbook
    assert "system_flags" in playbook
    assert "危机场景必须优先安全" in playbook
