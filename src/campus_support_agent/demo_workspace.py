from __future__ import annotations

from datetime import datetime
from typing import Any


DEMO_SCENARIOS: list[dict[str, Any]] = [
    {
        "scenario_id": "exam_sleep",
        "title": "期末复习焦虑与睡眠失衡",
        "texts": [
            "明天早上考试，我现在完全睡不着，越想越慌。",
            "我怕明天脑子一片空白。",
        ],
    },
    {
        "scenario_id": "dorm_boundary",
        "title": "宿舍边界与沟通压力",
        "texts": [
            "室友每天晚上外放视频，我提醒过一次，她好像不太高兴，我现在也不敢说了。",
            "我怕再说会被孤立。",
        ],
    },
    {
        "scenario_id": "crush_uncertainty",
        "title": "暗恋不确定与关系风险",
        "texts": [
            "我喜欢一个同学很久了，但他回复消息忽冷忽热，我不知道是不是我想太多。",
            "如果他拒绝，我怕以后连朋友都做不成。",
        ],
    },
    {
        "scenario_id": "privacy_blackmail",
        "title": "隐私威胁与安全转介",
        "texts": [
            "有人拿我的私密照片威胁我，说如果我不听他的就发到班群里。",
            "我不敢告诉别人，怕所有人都知道，也怕他继续逼我。",
        ],
        "human_status": "acknowledged",
        "handler_id": "counselor-demo-001",
    },
    {
        "scenario_id": "stalking_safety",
        "title": "被尾随后的安全安排",
        "texts": [
            "最近有个人总是在我下课后跟着我到宿舍附近，但我又怕是自己想多了。",
            "我今晚还有晚课，下课要走那条路。",
        ],
        "human_status": "in_progress",
        "handler_id": "counselor-demo-002",
    },
    {
        "scenario_id": "dangerous_place",
        "title": "危险地点与危机优先",
        "texts": [
            "我现在一个人在天台边上，真的不想活了，也不想再回去了。",
        ],
        "human_status": "escalated",
        "handler_id": "crisis-demo-001",
    },
]


def summarize_demo_turn(response: dict[str, Any], user_text: str, turn_index: int) -> dict[str, Any]:
    risk = response.get("risk") or {}
    entropy = response.get("entropy") or {}
    strategy = response.get("intervention_strategy") or {}
    referral = response.get("referral_decision") or {}
    dynamic = response.get("dynamic_adjustment") or {}
    return {
        "turn_index": turn_index,
        "user_text": user_text,
        "reply_text": response.get("reply_text") or "",
        "risk_level": risk.get("level"),
        "risk_score": risk.get("score"),
        "entropy_score": entropy.get("score"),
        "balance_state": entropy.get("balance_state"),
        "strategy_id": strategy.get("strategy_id"),
        "dynamic_action": dynamic.get("action"),
        "should_refer": bool(referral.get("should_refer")),
        "referral_urgency": referral.get("urgency"),
        "response_id": response.get("response_id"),
    }


def build_demo_workspace_report(
    *,
    scenario_results: list[dict[str, Any]],
    care_queue: dict[str, Any],
    generated_at: str | None = None,
) -> str:
    generated_at = generated_at or datetime.now().isoformat(timespec="seconds")
    lines = [
        "# 演示工作台样例报告",
        "",
        f"- 生成时间：{generated_at}",
        f"- 演示场景数：{len(scenario_results)}",
        f"- care queue 当前条目：{care_queue.get('total_items', 0)}",
        "",
        "## 场景总览",
        "",
        "| 场景 | 轮次 | 最高风险 | 最高心理熵 | 是否转介 | 人工状态 |",
        "| --- | ---: | --- | ---: | --- | --- |",
    ]
    for scenario in scenario_results:
        turns = scenario.get("turns") or []
        highest_entropy = max((int(turn.get("entropy_score") or 0) for turn in turns), default=0)
        risk_order = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        highest_risk = max((str(turn.get("risk_level") or "low") for turn in turns), key=lambda item: risk_order.get(item, 0))
        should_refer = any(bool(turn.get("should_refer")) for turn in turns)
        human = scenario.get("human_intervention") or {}
        lines.append(
            f"| {scenario['title']} | {len(turns)} | {highest_risk} | {highest_entropy} | "
            f"{'yes' if should_refer else 'no'} | {human.get('status', '-')} |"
        )

    lines.extend(
        [
            "",
            "## 逐场景样例",
            "",
        ]
    )
    for scenario in scenario_results:
        lines.extend([f"### {scenario['title']}", "", f"- session_id：`{scenario['session_id']}`"])
        human = scenario.get("human_intervention")
        if human:
            lines.append(f"- 人工处理：`{human.get('status')}`，处理人：`{human.get('handler_id')}`")
        for turn in scenario.get("turns") or []:
            reply = _compact_text(str(turn.get("reply_text") or ""), limit=120)
            lines.extend(
                [
                    "",
                    f"**第 {turn['turn_index']} 轮用户输入**：{turn['user_text']}",
                    "",
                    f"**系统回复摘录**：{reply}",
                    "",
                    (
                        f"- risk={turn.get('risk_level')}({turn.get('risk_score')}), "
                        f"entropy={turn.get('entropy_score')}, balance={turn.get('balance_state')}, "
                        f"strategy={turn.get('strategy_id')}, refer={turn.get('should_refer')}:{turn.get('referral_urgency')}"
                    ),
                ]
            )
        lines.append("")

    lines.extend(
        [
            "## Care Queue 摘要",
            "",
            f"- priority_counts：`{care_queue.get('priority_counts', {})}`",
            f"- route_counts：`{care_queue.get('route_counts', {})}`",
            f"- outcome_counts：`{care_queue.get('outcome_counts', {})}`",
            "",
            "## 使用建议",
            "",
            "- 学生端演示：选择低/中风险场景展示自然回复和低压力行动。",
            "- 咨询师端演示：选择隐私威胁、被尾随、危险地点场景展示 care queue 和人工状态。",
            "- 研究端演示：展示风险、心理熵、策略、转介和动态调整字段如何随轮次变化。",
            "",
        ]
    )
    return "\n".join(lines)


def _compact_text(text: str, *, limit: int) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1] + "..."
