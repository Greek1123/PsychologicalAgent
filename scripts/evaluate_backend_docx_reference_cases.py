from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.evaluate_docx_reference_cases import compare_reply, extract_all_cases, write_reports


def _configure_backend_environment(database_path: Path) -> None:
    os.environ["LLM_PROVIDER"] = "mock"
    os.environ["STT_PROVIDER"] = "mock"
    os.environ["DATABASE_PATH"] = str(database_path)

    from campus_support_agent import main

    main.get_settings.cache_clear()
    main.get_agent.cache_clear()
    main.get_session_store.cache_clear()


def _evaluate_backend_cases(cases: list[Any]) -> list[dict[str, Any]]:
    from campus_support_agent import main

    results: list[dict[str, Any]] = []
    for case in cases:
        history: list[dict[str, str]] = []
        evaluated_turns: list[dict[str, Any]] = []
        for turn_index, turn in enumerate(case.turns, start=1):
            response = main.support_text(
                {
                    "text": turn.user,
                    "student_context": {},
                    "conversation_history": history,
                }
            )
            model_reply = str(response.get("reply_text") or response.get("reply") or "")
            comparison = compare_reply(turn.reference_reply, model_reply, turn.user)
            evaluated_turns.append(
                {
                    "turn_index": turn_index,
                    "user": turn.user,
                    "reference_reply": turn.reference_reply,
                    "model_reply": model_reply,
                    "comparison": comparison,
                    "backend": {
                        "risk_level": (response.get("risk") or {}).get("level"),
                        "entropy_score": (response.get("entropy") or {}).get("score"),
                        "local_policy": ((response.get("support_assessment") or {}).get("local_policy") or {}).get(
                            "policy_name"
                        ),
                    },
                }
            )
            history.extend(
                [
                    {"role": "user", "content": turn.user},
                    {"role": "assistant", "content": model_reply},
                ]
            )
        scores = [item["comparison"]["score"] for item in evaluated_turns]
        results.append(
            {
                "case_id": case.case_id,
                "title": case.title,
                "source_doc": case.source_doc,
                "observation_points": case.observation_points,
                "turns": evaluated_turns,
                "average_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
            }
        )
    return results


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the FastAPI backend mock path against DOCX reference cases.")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--out-dir", default=str(ROOT / "reports" / "docx_reference_backend_eval"))
    parser.add_argument("--database-path", default=str(ROOT / "tmp_test_artifacts" / "docx_reference_backend_eval.db"))
    parser.add_argument("--docx", action="append", default=[])
    args = parser.parse_args()

    docx_paths = (
        [Path(item).expanduser().resolve() for item in args.docx]
        if args.docx
        else [
            ROOT / "docs" / "心理助手长对话模拟测试用例50例.docx",
            ROOT / "docs" / "心理助手长对话模拟测试用例_新增50例_含隐性高危场景.docx",
        ]
    )
    cases = extract_all_cases(docx_paths)
    if args.start < 1:
        raise ValueError("--start must be >= 1")
    selected = cases[args.start - 1 : args.start - 1 + args.limit]

    database_path = Path(args.database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if database_path.exists():
        database_path.unlink()
    _configure_backend_environment(database_path)

    results = _evaluate_backend_cases(selected)
    md_path, jsonl_path = write_reports(results, Path(args.out_dir), None)
    print(
        json.dumps(
            {
                "cases_total": len(cases),
                "cases_evaluated": len(results),
                "markdown": str(md_path),
                "jsonl": str(jsonl_path),
                "database": str(database_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main_cli()
