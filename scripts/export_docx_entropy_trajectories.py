from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.compare_docx_backend_experiment import categorize_case
from scripts.evaluate_backend_docx_reference_cases import _configure_backend_environment
from scripts.evaluate_docx_reference_cases import ReferenceCase, extract_all_cases


DEFAULT_DOCX = [
    ROOT / "docs" / "心理助手长对话模拟测试用例50例.docx",
    ROOT / "docs" / "心理助手长对话模拟测试用例_新增50例_含隐性高危场景.docx",
]


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _list_text(value: Any) -> str:
    if isinstance(value, list):
        return "；".join(str(item) for item in value)
    return _text(value)


def _turn_record(case: ReferenceCase, turn_index: int, user_text: str, response: dict[str, Any]) -> dict[str, Any]:
    entropy = response.get("entropy") or {}
    trend = entropy.get("trend") or {}
    risk = response.get("risk") or {}
    state_profile = response.get("state_profile") or {}
    reduction = response.get("entropy_reduction") or {}
    strategy = response.get("intervention_strategy") or {}
    dynamic = response.get("dynamic_adjustment") or {}
    orchestration = response.get("entropy_orchestration") or {}
    referral = response.get("referral_decision") or {}
    local_policy = ((response.get("support_assessment") or {}).get("local_policy")) or response.get("local_policy") or {}

    return {
        "case_id": case.case_id,
        "title": case.title,
        "category": categorize_case(case),
        "turn_index": turn_index,
        "user": user_text,
        "reply": _text(response.get("reply_text") or response.get("reply")),
        "risk_level": _text(risk.get("level")),
        "risk_score": risk.get("score"),
        "entropy_score": entropy.get("score"),
        "entropy_level": entropy.get("level"),
        "balance_state": _text(entropy.get("balance_state")),
        "trend_previous_score": trend.get("previous_score"),
        "trend_delta": trend.get("delta"),
        "trend_direction": _text(trend.get("direction")),
        "dominant_drivers": entropy.get("dominant_drivers") or [],
        "state_profile": _text(state_profile.get("primary_state")),
        "strategy_id": _text(strategy.get("strategy_id")),
        "strategy_priority": _text(strategy.get("priority")),
        "strategy_mode": _text(strategy.get("response_mode")),
        "dynamic_state": _text(dynamic.get("stability_state")),
        "dynamic_action": _text(dynamic.get("action")),
        "orchestration_route": _text(orchestration.get("route")),
        "target_state": _text(reduction.get("target_state")),
        "targeted_drivers": reduction.get("targeted_drivers") or [],
        "expected_delta_score": reduction.get("expected_delta_score"),
        "review_window_hours": reduction.get("review_window_hours"),
        "should_refer": referral.get("should_refer"),
        "referral_urgency": _text(referral.get("urgency")),
        "local_policy": _text(local_policy.get("policy_name")),
    }


def summarize_case(case_result: dict[str, Any]) -> dict[str, Any]:
    turns = case_result.get("turns", [])
    scores = [int(turn["entropy_score"]) for turn in turns if turn.get("entropy_score") is not None]
    risks = Counter(_text(turn.get("risk_level"), "unknown") for turn in turns)
    strategies = [_text(turn.get("strategy_id")) for turn in turns if turn.get("strategy_id")]
    dynamic_states = [_text(turn.get("dynamic_state")) for turn in turns if turn.get("dynamic_state")]
    if not scores:
        return {
            "turns": len(turns),
            "start_score": None,
            "end_score": None,
            "delta": None,
            "peak_score": None,
            "lowest_score": None,
            "risk_counts": dict(risks),
            "strategy_sequence": strategies,
            "dynamic_states": dynamic_states,
        }
    return {
        "turns": len(turns),
        "start_score": scores[0],
        "end_score": scores[-1],
        "delta": scores[-1] - scores[0],
        "peak_score": max(scores),
        "lowest_score": min(scores),
        "risk_counts": dict(risks),
        "strategy_sequence": strategies,
        "dynamic_states": dynamic_states,
    }


def summarize_all(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    all_turns = [turn for case in case_results for turn in case.get("turns", [])]
    scores = [int(turn["entropy_score"]) for turn in all_turns if turn.get("entropy_score") is not None]
    risk_counts = Counter(_text(turn.get("risk_level"), "unknown") for turn in all_turns)
    balance_counts = Counter(_text(turn.get("balance_state"), "unknown") for turn in all_turns)
    strategy_counts = Counter(_text(turn.get("strategy_id"), "unknown") for turn in all_turns)
    category_scores: dict[str, list[int]] = {}
    for turn in all_turns:
        if turn.get("entropy_score") is None:
            continue
        category_scores.setdefault(_text(turn.get("category"), "其他校园压力"), []).append(int(turn["entropy_score"]))
    return {
        "cases": len(case_results),
        "turns": len(all_turns),
        "average_entropy": round(sum(scores) / len(scores), 2) if scores else 0.0,
        "max_entropy": max(scores) if scores else None,
        "min_entropy": min(scores) if scores else None,
        "risk_counts": dict(risk_counts.most_common()),
        "balance_counts": dict(balance_counts.most_common()),
        "top_strategies": dict(strategy_counts.most_common(12)),
        "by_category": {
            category: {
                "turns": len(items),
                "average_entropy": round(sum(items) / len(items), 2),
                "max_entropy": max(items),
            }
            for category, items in sorted(category_scores.items())
        },
    }


def run_trajectories(cases: list[ReferenceCase]) -> list[dict[str, Any]]:
    from campus_support_agent import main

    results: list[dict[str, Any]] = []
    for case in cases:
        session_id = f"docx-trajectory-{case.case_id}-{uuid4().hex[:8]}"
        turns: list[dict[str, Any]] = []
        for turn_index, turn in enumerate(case.turns, start=1):
            response = main.support_text(
                {
                    "text": turn.user,
                    "student_context": {"experiment_case_id": case.case_id, "experiment_title": case.title},
                    "conversation_history": [],
                    "session_id": session_id,
                }
            )
            turns.append(_turn_record(case, turn_index, turn.user, response))
        case_result = {
            "case_id": case.case_id,
            "title": case.title,
            "category": categorize_case(case),
            "source_doc": case.source_doc,
            "session_id": session_id,
            "turns": turns,
        }
        case_result["summary"] = summarize_case(case_result)
        results.append(case_result)
    return results


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, case_results: list[dict[str, Any]]) -> None:
    fieldnames = [
        "case_id",
        "title",
        "category",
        "turn_index",
        "risk_level",
        "risk_score",
        "entropy_score",
        "entropy_level",
        "balance_state",
        "trend_previous_score",
        "trend_delta",
        "trend_direction",
        "dominant_drivers",
        "state_profile",
        "strategy_id",
        "strategy_priority",
        "strategy_mode",
        "dynamic_state",
        "dynamic_action",
        "orchestration_route",
        "target_state",
        "targeted_drivers",
        "expected_delta_score",
        "review_window_hours",
        "should_refer",
        "referral_urgency",
        "local_policy",
        "user",
        "reply",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in case_results:
            for turn in case.get("turns", []):
                row = dict(turn)
                row["dominant_drivers"] = _list_text(row.get("dominant_drivers"))
                row["targeted_drivers"] = _list_text(row.get("targeted_drivers"))
                writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# DOCX 长对话熵轨迹与动态平衡报告",
        "",
        f"- 生成时间：{payload['generated_at']}",
        f"- 评估范围：start={payload['start']}, limit={payload['limit']}",
        f"- 案例数：{summary['cases']}",
        f"- 回复轮数：{summary['turns']}",
        f"- 平均心理熵：{summary['average_entropy']}",
        f"- 心理熵范围：{summary['min_entropy']} - {summary['max_entropy']}",
        "",
        "## 总体分布",
        "",
        f"- 风险等级：`{summary['risk_counts']}`",
        f"- 平衡状态：`{summary['balance_counts']}`",
        f"- 高频策略：`{summary['top_strategies']}`",
        "",
        "## 分类熵水平",
        "",
        "| 类别 | turn 数 | 平均心理熵 | 峰值心理熵 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for category, item in summary["by_category"].items():
        lines.append(f"| {category} | {item['turns']} | {item['average_entropy']} | {item['max_entropy']} |")

    lines.extend(["", "## 案例轨迹", ""])
    for case in payload["cases"]:
        case_summary = case["summary"]
        lines.extend(
            [
                f"### {case['case_id']} {case['title']}",
                "",
                f"- 类别：{case['category']}",
                f"- session_id：`{case['session_id']}`",
                f"- 熵变化：{case_summary['start_score']} -> {case_summary['end_score']}，delta={case_summary['delta']}，peak={case_summary['peak_score']}",
                f"- 风险分布：`{case_summary['risk_counts']}`",
                f"- 策略序列：`{' -> '.join(case_summary['strategy_sequence'])}`",
                "",
                "| 轮次 | 风险 | 熵 | 趋势 | 平衡 | 主导熵源 | 策略 | 动态状态 |",
                "| ---: | --- | ---: | --- | --- | --- | --- | --- |",
            ]
        )
        for turn in case.get("turns", []):
            trend = turn.get("trend_direction") or "n/a"
            delta = turn.get("trend_delta")
            trend_text = f"{trend}({delta})" if delta is not None else trend
            lines.append(
                "| {turn} | {risk} | {score} | {trend} | {balance} | {drivers} | {strategy} | {dynamic} |".format(
                    turn=turn.get("turn_index"),
                    risk=turn.get("risk_level"),
                    score=turn.get("entropy_score"),
                    trend=trend_text,
                    balance=turn.get("balance_state"),
                    drivers=_list_text(turn.get("dominant_drivers")),
                    strategy=turn.get("strategy_id"),
                    dynamic=turn.get("dynamic_state"),
                )
            )
        lines.append("")

    lines.extend(
        [
            "## 使用说明",
            "",
            "- Markdown 适合直接整理进论文/答辩材料。",
            "- CSV 适合画折线图、柱状图和分类统计图。",
            "- JSON 保留完整字段，便于后续自动生成案例分析或前端演示数据。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export entropy trajectories for DOCX multi-turn reference cases.")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--out-dir", default=str(ROOT / "reports" / "docx_entropy_trajectories"))
    parser.add_argument("--database-path", default=str(ROOT / "tmp_test_artifacts" / "docx_entropy_trajectories.db"))
    parser.add_argument("--docx", action="append", default=[])
    parser.add_argument("--verbose-logs", action="store_true")
    args = parser.parse_args()

    docx_paths = [Path(item).expanduser().resolve() for item in args.docx] if args.docx else DEFAULT_DOCX
    all_cases = extract_all_cases(docx_paths)
    if args.start < 1:
        raise ValueError("--start must be >= 1")
    cases = all_cases[args.start - 1 : args.start - 1 + args.limit]

    database_path = Path(args.database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if database_path.exists():
        database_path.unlink()
    if not args.verbose_logs:
        logging.disable(logging.WARNING)
    _configure_backend_environment(database_path)

    case_results = run_trajectories(cases)
    if not args.verbose_logs:
        logging.disable(logging.NOTSET)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "start": args.start,
        "limit": args.limit,
        "database": str(database_path),
        "summary": summarize_all(case_results),
        "cases": case_results,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = out_dir / f"{stamp}_docx_entropy_trajectories.md"
    json_path = out_dir / f"{stamp}_docx_entropy_trajectories.json"
    csv_path = out_dir / f"{stamp}_docx_entropy_trajectories.csv"

    write_markdown(md_path, payload)
    write_json(json_path, payload)
    write_csv(csv_path, case_results)

    print(
        json.dumps(
            {
                "markdown": str(md_path),
                "json": str(json_path),
                "csv": str(csv_path),
                "summary": payload["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
