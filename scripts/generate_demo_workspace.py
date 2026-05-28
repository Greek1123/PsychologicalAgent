from __future__ import annotations

import argparse
import os
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a demo workspace report by running representative sessions.")
    parser.add_argument("--out", default=str(ROOT / "docs" / "demo_workspace_report.md"))
    parser.add_argument("--database-path", default=str(ROOT / "tmp_test_artifacts" / "demo_workspace.db"))
    args = parser.parse_args()

    database_path = Path(args.database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if database_path.exists():
        database_path.unlink()

    os.environ["LLM_PROVIDER"] = "mock"
    os.environ["STT_PROVIDER"] = "mock"
    os.environ["DATABASE_PATH"] = str(database_path)

    from campus_support_agent import main as app_main

    app_main.get_settings.cache_clear()
    app_main.get_agent.cache_clear()
    app_main.get_session_store.cache_clear()

    scenario_results: list[dict[str, object]] = []
    for scenario in DEMO_SCENARIOS:
        session_id = f"demo-{scenario['scenario_id']}"
        turns: list[dict[str, object]] = []
        latest_response_id = None
        for turn_index, text in enumerate(scenario["texts"], start=1):
            response = app_main.support_text(
                {
                    "session_id": session_id,
                    "text": text,
                    "student_context": {"demo": True, "scenario_id": scenario["scenario_id"]},
                    "conversation_history": [],
                }
            )
            latest_response_id = response.get("response_id")
            turns.append(summarize_demo_turn(response, str(text), turn_index))

        human_intervention = None
        if scenario.get("human_status"):
            human_intervention = app_main.append_session_human_intervention(
                session_id,
                {
                    "response_id": latest_response_id,
                    "status": scenario["human_status"],
                    "handler_id": scenario.get("handler_id"),
                    "note": "演示数据：已由人工工作台接手。",
                    "next_action": "demo_followup",
                    "tags": ["demo", "manual_followup"],
                },
            )["human_intervention"]

        scenario_results.append(
            {
                "scenario_id": scenario["scenario_id"],
                "title": scenario["title"],
                "session_id": session_id,
                "turns": turns,
                "human_intervention": human_intervention,
            }
        )

    care_queue = app_main.get_care_queue(limit=50, include_low_priority=True, include_resolved=True)
    report = build_demo_workspace_report(scenario_results=scenario_results, care_queue=care_queue)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
