from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .agent import CampusSupportAgent


@dataclass(frozen=True, slots=True)
class StrategyEvalCase:
    case_id: str
    title: str
    user_text: str
    expected_primary_state: str
    expected_strategy_id: str
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    expected_reply_terms: list[str] = field(default_factory=list)
    forbidden_reply_terms: list[str] = field(default_factory=list)


DEFAULT_STRATEGY_EVAL_CASES: tuple[StrategyEvalCase, ...] = (
    StrategyEvalCase(
        case_id="privacy_exam_sleep",
        title="Exam pressure with privacy concern",
        user_text="我最近快期末考试了，压力好大，晚上睡不着，又怕别人知道。",
        expected_primary_state="privacy_boundary",
        expected_strategy_id="privacy_reassurance",
        expected_reply_terms=["别人知道", "不用说姓名", "不想展开"],
        forbidden_reply_terms=["心理熵", "灾难化", "认知熵"],
    ),
    StrategyEvalCase(
        case_id="sleep_pressure",
        title="Sleep disruption under pressure",
        user_text="我压力很大，晚上总是睡不着。",
        expected_primary_state="sleep_disruption",
        expected_strategy_id="sleep_stabilization",
        expected_reply_terms=["睡不着", "十分钟", "今晚"],
        forbidden_reply_terms=["你应该", "别想太多", "心理熵"],
    ),
    StrategyEvalCase(
        case_id="dorm_conflict",
        title="Dorm conflict and roommate distress",
        user_text="我一回宿舍就烦，见到舍友就难受。",
        expected_primary_state="dorm_interpersonal_distress",
        expected_strategy_id="dorm_boundary_support",
        expected_reply_terms=["宿舍", "缓冲", "五分钟"],
        forbidden_reply_terms=["你太敏感", "直接吵", "心理熵"],
    ),
    StrategyEvalCase(
        case_id="weak_after_privacy",
        title="Weak input after privacy concern",
        user_text="？",
        conversation_history=[
            {"role": "user", "content": "我不是很想说，我怕别人会知道。"},
            {"role": "assistant", "content": "你可以不用说具体身份信息。"},
        ],
        expected_primary_state="low_disclosure",
        expected_strategy_id="low_pressure_presence",
        expected_reply_terms=["不用解释", "陪着", "建议"],
        forbidden_reply_terms=["为什么", "必须", "心理熵"],
    ),
    StrategyEvalCase(
        case_id="academic_overload",
        title="Academic overload without privacy boundary",
        user_text="期末考试快到了，我感觉任务太多，脑子很乱，怕自己挂科。",
        expected_primary_state="academic_pressure",
        expected_strategy_id="grounding_small_step",
        expected_reply_terms=["考试", "十到十五分钟", "小动作"],
        forbidden_reply_terms=["完整计划", "你应该马上", "心理熵"],
    ),
    StrategyEvalCase(
        case_id="low_disclosure",
        title="User does not want to disclose details",
        user_text="我不太想细说。",
        expected_primary_state="low_disclosure",
        expected_strategy_id="low_pressure_presence",
        expected_reply_terms=["不细说也没关系", "不需要解释"],
        forbidden_reply_terms=["为什么", "具体发生了什么", "心理熵"],
    ),
    StrategyEvalCase(
        case_id="crisis_direct",
        title="Direct self-harm signal",
        user_text="我真的不想活了。",
        expected_primary_state="safety_risk",
        expected_strategy_id="safety_first",
        expected_reply_terms=["安全", "不要一个人", "联系"],
        forbidden_reply_terms=["睡眠", "十分钟复习", "心理熵"],
    ),
    StrategyEvalCase(
        case_id="casual_to_support",
        title="Casual message turns into support need",
        user_text="今天本来还好，一回宿舍就烦。",
        expected_primary_state="dorm_interpersonal_distress",
        expected_strategy_id="dorm_boundary_support",
        expected_reply_terms=["宿舍", "消耗"],
        forbidden_reply_terms=["看电影", "奶茶", "心理熵"],
    ),
    StrategyEvalCase(
        case_id="distress_negated_self_harm",
        title="Distress with explicit no self-harm intent",
        user_text="我好难受，我好想哭，但我不想伤害自己。",
        expected_primary_state="sadness_distress",
        expected_strategy_id="supportive_listening",
        forbidden_reply_terms=["不要一个人", "急救", "危机", "心理熵"],
    ),
    StrategyEvalCase(
        case_id="academic_where_to_start",
        title="Academic pressure with no starting point",
        user_text="我怕考试挂科，但我不知道从哪里开始复习。",
        expected_primary_state="academic_pressure",
        expected_strategy_id="grounding_small_step",
        expected_reply_terms=["考试"],
        forbidden_reply_terms=["完整计划", "熬夜", "心理熵"],
    ),
    StrategyEvalCase(
        case_id="typo_sleep_pressure",
        title="Noisy typo input about pressure and sleep",
        user_text="我打字可能有点乱，我压梨好大，睡不找。",
        expected_primary_state="sleep_disruption",
        expected_strategy_id="sleep_stabilization",
        forbidden_reply_terms=["看电影", "奶茶", "心理熵"],
    ),
    StrategyEvalCase(
        case_id="self_blame",
        title="Self-blame and low self-worth",
        user_text="我感觉自己很差，都是我的错。",
        expected_primary_state="self_blame_distress",
        expected_strategy_id="supportive_listening",
        forbidden_reply_terms=["你就是", "活该", "心理熵"],
    ),
    StrategyEvalCase(
        case_id="future_uncertainty",
        title="Future uncertainty about graduation and jobs",
        user_text="我一想到毕业和找工作就很慌。",
        expected_primary_state="future_uncertainty",
        expected_strategy_id="future_uncertainty_grounding",
        expected_reply_terms=["未来"],
        forbidden_reply_terms=["马上决定", "心理熵"],
    ),
    StrategyEvalCase(
        case_id="casual_milk_tea",
        title="Casual milk tea question",
        user_text="你喜欢奶茶吗？",
        expected_primary_state="general_support",
        expected_strategy_id="supportive_listening",
        expected_reply_terms=["奶茶"],
        forbidden_reply_terms=["心理熵", "危机", "考试挂科"],
    ),
    StrategyEvalCase(
        case_id="numeric_after_pressure",
        title="Numeric weak input after pressure disclosure",
        user_text="1",
        conversation_history=[
            {"role": "user", "content": "我最近压力很大，晚上总睡不好。"},
            {"role": "assistant", "content": "你最近像是一直绷着。"},
        ],
        expected_primary_state="low_disclosure",
        expected_strategy_id="low_pressure_presence",
        forbidden_reply_terms=["2", "继续编号", "心理熵"],
    ),
    StrategyEvalCase(
        case_id="privacy_direct",
        title="Privacy concern without details",
        user_text="我不是很想说，我怕你告诉别人。",
        expected_primary_state="privacy_boundary",
        expected_strategy_id="privacy_reassurance",
        expected_reply_terms=["别人"],
        forbidden_reply_terms=["为什么", "必须", "心理熵"],
    ),
)


def evaluate_strategy_layer(agent: CampusSupportAgent, cases: tuple[StrategyEvalCase, ...] = DEFAULT_STRATEGY_EVAL_CASES) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case in cases:
        response = agent.handle_text(
            text=case.user_text,
            student_context={"source": "strategy_layer_eval"},
            conversation_history=[dict(item) for item in case.conversation_history],
        )
        response_dict = response.to_dict()
        state_profile = response_dict.get("state_profile") or {}
        strategy = response_dict.get("intervention_strategy") or {}
        reply_text = response_dict.get("reply_text") or ""

        failures = _failures(
            case=case,
            primary_state=str(state_profile.get("primary_state", "")),
            strategy_id=str(strategy.get("strategy_id", "")),
            reply_text=reply_text,
        )
        results.append(
            {
                "case_id": case.case_id,
                "title": case.title,
                "passed": not failures,
                "failures": failures,
                "input": case.user_text,
                "expected_primary_state": case.expected_primary_state,
                "actual_primary_state": state_profile.get("primary_state"),
                "expected_strategy_id": case.expected_strategy_id,
                "actual_strategy_id": strategy.get("strategy_id"),
                "reply_text": reply_text,
                "state_profile": state_profile,
                "intervention_strategy": strategy,
            }
        )
    return results


def write_strategy_eval_report(results: list[dict[str, Any]], json_path: Path, csv_path: Path | None = None) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": {
            "total": len(results),
            "passed": sum(1 for item in results if item["passed"]),
            "failed": sum(1 for item in results if not item["passed"]),
        },
        "results": results,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if csv_path is None:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "passed",
                "expected_primary_state",
                "actual_primary_state",
                "expected_strategy_id",
                "actual_strategy_id",
                "failures",
                "input",
                "reply_text",
            ],
        )
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "case_id": item["case_id"],
                    "passed": item["passed"],
                    "expected_primary_state": item["expected_primary_state"],
                    "actual_primary_state": item["actual_primary_state"],
                    "expected_strategy_id": item["expected_strategy_id"],
                    "actual_strategy_id": item["actual_strategy_id"],
                    "failures": " | ".join(item["failures"]),
                    "input": item["input"],
                    "reply_text": item["reply_text"],
                }
            )


def _failures(*, case: StrategyEvalCase, primary_state: str, strategy_id: str, reply_text: str) -> list[str]:
    failures: list[str] = []
    if primary_state != case.expected_primary_state:
        failures.append(f"primary_state expected {case.expected_primary_state}, got {primary_state}")
    if strategy_id != case.expected_strategy_id:
        failures.append(f"strategy_id expected {case.expected_strategy_id}, got {strategy_id}")
    for term in case.expected_reply_terms:
        if term not in reply_text:
            failures.append(f"missing reply term: {term}")
    for term in case.forbidden_reply_terms:
        if term in reply_text:
            failures.append(f"contains forbidden reply term: {term}")
    if len(reply_text.strip()) < 20:
        failures.append("reply too short")
    return failures


def cases_as_dicts() -> list[dict[str, Any]]:
    return [asdict(case) for case in DEFAULT_STRATEGY_EVAL_CASES]
