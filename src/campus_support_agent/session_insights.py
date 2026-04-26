from __future__ import annotations

from typing import Any


_POLICY_LABELS = {
    "privacy_concern": "隐私与信任建立",
    "boundary_concern": "表达边界与安全感",
    "weak_input": "低表达输入承接",
    "dorm_conflict": "宿舍/同伴关系压力",
    "exam_anxiety": "考试与学业焦虑",
    "late_night_distress": "夜间睡眠困扰",
    "sleep_appetite_drift": "睡眠饮食节律受影响",
    "helplessness_escalation": "无助感升高",
    "rising_emotional_spiral": "情绪持续上升",
    "repetitive_help_seeking": "反复求助与持续消耗",
    "social_isolation": "社交退缩或孤立感",
    "exhaustion_withdrawal": "疲惫与退缩",
    "authority_pressure": "权威/家庭压力",
    "future_panic": "未来不确定焦虑",
    "self_blame": "自责与负性自我评价",
}

_RISK_RANK = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


def build_session_insight(
    *,
    session_id: str,
    records: list[dict[str, Any]],
    entropy_trace: list[dict[str, Any]],
    referral_events: list[dict[str, Any]],
) -> dict[str, Any]:
    if not records:
        return {
            "session_id": session_id,
            "state_summary": _empty_state_summary(referral_events),
            "current_focus": "转介记录待补充" if referral_events else None,
            "risk_route": "manual_followup" if referral_events else "observe",
            "entropy_trend": "baseline",
            "watch_items": ["已有转介或人工关注记录，需要补充完整对话记录。"] if referral_events else [],
            "recommended_next_steps": (
                ["补充完整对话记录，并确认用户是否已经获得现实支持。"]
                if referral_events
                else ["先收集至少一轮用户输入和系统回复。"]
            ),
            "evidence": {
                "response_count": 0,
                "latest_entropy_score": None,
                "latest_risk_level": None,
                "latest_policy_name": None,
                "referral_event_count": len(referral_events),
            },
        }

    latest = records[-1]
    latest_policy = latest.get("local_policy_name")
    latest_risk = latest.get("risk_level") or "low"
    latest_entropy = latest.get("entropy_score")
    latest_balance = latest.get("balance_state") or "stable"
    entropy_trend = _summarize_entropy_trend(entropy_trace)
    current_focus = _POLICY_LABELS.get(latest_policy, "一般支持对话")
    risk_route = _risk_route(latest_risk, latest_entropy, referral_events, entropy_trend)
    watch_items = _build_watch_items(records, entropy_trace, referral_events)
    next_steps = _build_next_steps(
        risk_route=risk_route,
        latest_policy=latest_policy,
        latest_risk=latest_risk,
        latest_entropy=latest_entropy,
        entropy_trend=entropy_trend,
    )

    return {
        "session_id": session_id,
        "state_summary": _state_summary(
            current_focus=current_focus,
            latest_risk=latest_risk,
            latest_entropy=latest_entropy,
            latest_balance=latest_balance,
            entropy_trend=entropy_trend,
            referral_events=referral_events,
        ),
        "current_focus": current_focus,
        "risk_route": risk_route,
        "entropy_trend": entropy_trend,
        "watch_items": watch_items,
        "recommended_next_steps": next_steps,
        "evidence": {
            "response_count": len(records),
            "latest_entropy_score": latest_entropy,
            "latest_risk_level": latest_risk,
            "latest_policy_name": latest_policy,
            "latest_balance_state": latest_balance,
            "referral_event_count": len(referral_events),
        },
    }


def _summarize_entropy_trend(entropy_trace: list[dict[str, Any]]) -> str:
    if len(entropy_trace) < 2:
        return "baseline"
    first = int(entropy_trace[0].get("score") or 0)
    latest = int(entropy_trace[-1].get("score") or 0)
    delta = latest - first
    if delta >= 10:
        return "rising"
    if delta <= -10:
        return "falling"
    return "stable"


def _risk_route(
    latest_risk: str,
    latest_entropy: int | None,
    referral_events: list[dict[str, Any]],
    entropy_trend: str,
) -> str:
    if latest_risk == "critical":
        return "urgent_referral"
    if latest_risk == "high":
        return "manual_followup"
    if any(event.get("manual_referral_recommended") for event in referral_events[-2:]):
        return "manual_followup"
    if latest_entropy is not None and latest_entropy >= 65:
        return "manual_followup"
    if entropy_trend == "rising":
        return "watch_closely"
    if referral_events:
        return "watch_closely"
    return "observe"


def _build_watch_items(
    records: list[dict[str, Any]],
    entropy_trace: list[dict[str, Any]],
    referral_events: list[dict[str, Any]],
) -> list[str]:
    items: list[str] = []
    policies = [record.get("local_policy_name") for record in records if record.get("local_policy_name")]
    risk_levels = [record.get("risk_level") for record in records if record.get("risk_level")]

    if policies:
        latest_policy = policies[-1]
        if latest_policy in {"sleep_appetite_drift", "late_night_distress"}:
            items.append("继续观察睡眠、饮食和夜间求助是否持续受影响。")
        if latest_policy in {"privacy_concern", "boundary_concern"}:
            items.append("优先建立安全感，不要过度追问细节。")
        if latest_policy in {"dorm_conflict", "social_isolation"}:
            items.append("关注人际压力是否从烦躁升级为回避、孤立或冲突。")
        if latest_policy in {"helplessness_escalation", "rising_emotional_spiral"}:
            items.append("关注无助感和情绪上升是否连续出现。")

    if any(_RISK_RANK.get(level, 0) >= 3 for level in risk_levels):
        items.append("已有高风险信号，后续回复必须优先安全和人工支持。")

    if len(entropy_trace) >= 2:
        latest = int(entropy_trace[-1].get("score") or 0)
        previous = int(entropy_trace[-2].get("score") or 0)
        if latest - previous >= 8:
            items.append("最近一轮心理熵上升明显，需要缩小问题范围并减少任务负担。")

    if referral_events:
        items.append("已有转介或人工跟进记录，后续要检查用户是否获得现实支持。")

    return _dedupe(items)


def _build_next_steps(
    *,
    risk_route: str,
    latest_policy: str | None,
    latest_risk: str,
    latest_entropy: int | None,
    entropy_trend: str,
) -> list[str]:
    if risk_route == "urgent_referral":
        return [
            "立即切换危机支持话术，避免继续普通聊天。",
            "提示联系校内心理危机渠道、辅导员或当地紧急服务。",
            "记录触发原因，方便人工接手。",
        ]
    if risk_route == "manual_followup":
        return [
            "建议进入人工关注队列，由老师或心理中心后续确认。",
            "下一轮对话先确认睡眠、饮食、安全感和现实支持情况。",
            "回复中减少分析词，优先给一个可执行的小动作。",
        ]
    if risk_route == "watch_closely":
        return [
            "继续保持温和追问，但每轮只追一个问题。",
            "观察心理熵是否继续上升，必要时升级为人工跟进。",
            "把建议压缩成一到两个小步骤，避免让用户负担更重。",
        ]

    if latest_policy in {"privacy_concern", "boundary_concern"}:
        return [
            "下一轮先确认用户愿意聊到什么程度。",
            "允许用户只说感受，不要求说完整事件。",
        ]
    if latest_policy in {"dorm_conflict", "exam_anxiety"}:
        return [
            "下一轮聚焦一个具体场景，不要同时处理所有问题。",
            "给出一个低成本行动，例如离开刺激源、拆分任务或短时休息。",
        ]
    if latest_entropy is not None and latest_entropy >= 45:
        return [
            "先稳定情绪和身体节律，再进入问题解决。",
            "下一轮只确认一个主要压力源。",
        ]
    if latest_risk == "medium" or entropy_trend == "rising":
        return [
            "继续观察风险和熵值变化。",
            "下一轮回复保持短而具体，避免说教。",
        ]
    return [
        "维持普通支持性对话。",
        "继续记录上下文，为后续状态识别积累样本。",
    ]


def _state_summary(
    *,
    current_focus: str,
    latest_risk: str,
    latest_entropy: int | None,
    latest_balance: str,
    entropy_trend: str,
    referral_events: list[dict[str, Any]],
) -> str:
    score_text = "暂无熵值" if latest_entropy is None else f"心理熵 {latest_entropy}"
    trend_text = {
        "baseline": "暂未形成趋势",
        "stable": "整体波动不大",
        "rising": "呈上升趋势",
        "falling": "呈下降趋势",
    }.get(entropy_trend, entropy_trend)
    referral_text = "，已有转介/人工关注记录" if referral_events else ""
    return (
        f"当前重点是{current_focus}，风险等级为 {latest_risk}，"
        f"{score_text}，状态为 {latest_balance}，{trend_text}{referral_text}。"
    )


def _empty_state_summary(referral_events: list[dict[str, Any]]) -> str:
    if referral_events:
        return "暂无完整对话记录，但已有转介或人工关注事件，需要优先补充上下文并确认跟进状态。"
    return "暂无足够对话记录，先完成一次文本支持交互。"


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
