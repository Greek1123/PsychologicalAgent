from __future__ import annotations

from collections import Counter
from typing import Any


BOUNDARY_STATES = {"privacy_boundary", "low_disclosure"}
ACTIVE_DISTRESS_STATES = {
    "academic_pressure",
    "academic_sleep_stress",
    "sleep_disruption",
    "dorm_interpersonal_distress",
    "sadness_distress",
    "self_blame_distress",
}


def build_session_continuity_summary(
    *,
    session_id: str,
    records: list[dict[str, Any]],
    conversation_history: list[dict[str, Any]] | None = None,
    entropy_trace: list[dict[str, Any]] | None = None,
    limit: int = 8,
) -> dict[str, Any]:
    recent_records = records[-limit:]
    recent_history = (conversation_history or [])[-limit:]
    states = [str(record.get("primary_state")) for record in recent_records if record.get("primary_state")]
    domains = _collect_domains(recent_records)
    boundaries = _collect_boundaries(recent_records, recent_history)
    latest_entropy = _latest_entropy(recent_records, entropy_trace or [])
    entropy_direction = _entropy_direction(recent_records, entropy_trace or [])
    repeated_assistant_moves = _repeated_assistant_moves(recent_records, recent_history)
    stage = _infer_dialogue_stage(
        records=recent_records,
        states=states,
        boundaries=boundaries,
        latest_entropy=latest_entropy,
        entropy_direction=entropy_direction,
    )

    return {
        "session_id": session_id,
        "dialogue_stage": stage,
        "continuity_brief": _brief(stage, states, domains, boundaries, entropy_direction),
        "dominant_recent_states": [item for item, _ in Counter(states).most_common(3)],
        "dominant_recent_domains": [item for item, _ in Counter(domains).most_common(4)],
        "user_boundaries": boundaries,
        "recent_user_needs": _recent_user_needs(recent_records, recent_history),
        "avoid_next_turn": _avoid_next_turn(stage, repeated_assistant_moves, boundaries),
        "recommended_next_moves": _recommended_next_moves(stage, domains, boundaries),
        "entropy_direction": entropy_direction,
        "latest_entropy_score": latest_entropy,
        "repeated_assistant_moves": repeated_assistant_moves,
        "evidence": {
            "recent_record_count": len(recent_records),
            "recent_history_count": len(recent_history),
            "states": states,
        },
    }


def enrich_student_context_with_continuity(
    student_context: dict[str, Any],
    continuity_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    if not continuity_summary:
        return dict(student_context or {})
    enriched = dict(student_context or {})
    enriched["session_continuity"] = {
        "dialogue_stage": continuity_summary.get("dialogue_stage"),
        "continuity_brief": continuity_summary.get("continuity_brief"),
        "recent_user_needs": continuity_summary.get("recent_user_needs") or [],
        "user_boundaries": continuity_summary.get("user_boundaries") or [],
        "avoid_next_turn": continuity_summary.get("avoid_next_turn") or [],
        "recommended_next_moves": continuity_summary.get("recommended_next_moves") or [],
        "instruction": (
            "Use this hidden continuity summary to continue the same support relationship. "
            "Do not repeat the previous assistant move, do not restart the conversation, "
            "and respect any user boundary listed here."
        ),
    }
    return enriched


def _collect_domains(records: list[dict[str, Any]]) -> list[str]:
    domains: list[str] = []
    for record in records:
        state_profile = record.get("state_profile") or {}
        for domain in state_profile.get("stress_domains") or []:
            clean = str(domain).strip()
            if clean:
                domains.append(clean)
    return domains


def _collect_boundaries(records: list[dict[str, Any]], history: list[dict[str, Any]]) -> list[str]:
    boundaries: list[str] = []
    for record in records:
        state_profile = record.get("state_profile") or {}
        boundaries.extend(str(item) for item in state_profile.get("boundary_flags") or [])
    user_text = " ".join(
        str(item.get("content") or "")
        for item in history
        if item.get("role") == "user"
    )
    if any(term in user_text for term in ("不想说", "不想细说", "别问", "怕别人知道", "会告诉别人")):
        boundaries.append("low_disclosure_or_privacy_concern")
    return sorted(set(item for item in boundaries if item))


def _latest_entropy(records: list[dict[str, Any]], entropy_trace: list[dict[str, Any]]) -> int | None:
    for item in reversed(entropy_trace):
        try:
            return int(item.get("score"))
        except (TypeError, ValueError):
            continue
    for record in reversed(records):
        try:
            return int(record.get("entropy_score"))
        except (TypeError, ValueError):
            continue
    return None


def _entropy_direction(records: list[dict[str, Any]], entropy_trace: list[dict[str, Any]]) -> str:
    scores: list[int] = []
    for item in entropy_trace:
        try:
            scores.append(int(item.get("score")))
        except (TypeError, ValueError):
            continue
    if not scores:
        for record in records:
            try:
                scores.append(int(record.get("entropy_score")))
            except (TypeError, ValueError):
                continue
    if len(scores) < 2:
        return "baseline"
    delta = scores[-1] - scores[0]
    if delta >= 8:
        return "rising"
    if delta <= -8:
        return "falling"
    return "stable"


def _repeated_assistant_moves(records: list[dict[str, Any]], history: list[dict[str, Any]]) -> list[str]:
    replies = [str(record.get("reply_text") or "").strip() for record in records if record.get("reply_text")]
    replies.extend(
        str(item.get("content") or "").strip()
        for item in history
        if item.get("role") == "assistant" and item.get("content")
    )
    repeated: list[str] = []
    if len(replies) >= 2 and replies[-1] == replies[-2]:
        repeated.append("exact_reply_repeated")
    last_two = " ".join(replies[-2:])
    if last_two.count("发生了什么") >= 2 or last_two.count("具体说") >= 2:
        repeated.append("too_many_detail_questions")
    if last_two.count("建议") >= 3:
        repeated.append("advice_pressure")
    return repeated


def _infer_dialogue_stage(
    *,
    records: list[dict[str, Any]],
    states: list[str],
    boundaries: list[str],
    latest_entropy: int | None,
    entropy_direction: str,
) -> str:
    if any(record.get("risk_level") == "critical" for record in records):
        return "safety_priority"
    if any(record.get("risk_level") == "high" for record in records):
        return "human_followup_watch"
    if boundaries or any(state in BOUNDARY_STATES for state in states):
        return "boundary_building"
    if latest_entropy is not None and latest_entropy >= 65:
        return "high_entropy_stabilization"
    if entropy_direction == "rising":
        return "deteriorating_watch"
    if len(records) <= 1:
        return "initial_contact"
    if any(state in ACTIVE_DISTRESS_STATES for state in states):
        return "active_support"
    if entropy_direction == "falling":
        return "stabilizing"
    return "maintenance"


def _brief(stage: str, states: list[str], domains: list[str], boundaries: list[str], entropy_direction: str) -> str:
    state_text = "、".join([item for item, _ in Counter(states).most_common(2)]) or "暂未形成稳定主题"
    domain_text = "、".join([item for item, _ in Counter(domains).most_common(2)]) or "未明确"
    boundary_text = "；用户有表达边界或隐私顾虑" if boundaries else ""
    return f"当前阶段为 {stage}；近期主要状态为 {state_text}；主要压力域为 {domain_text}；熵值趋势为 {entropy_direction}{boundary_text}。"


def _recent_user_needs(records: list[dict[str, Any]], history: list[dict[str, Any]]) -> list[str]:
    needs: list[str] = []
    if any((record.get("state_profile") or {}).get("boundary_flags") for record in records):
        needs.append("需要隐私保证和低压力陪伴")
    if any("sleep" in ((record.get("state_profile") or {}).get("stress_domains") or []) for record in records):
        needs.append("需要先稳定睡眠和身体节律")
    if any("academic" in ((record.get("state_profile") or {}).get("stress_domains") or []) for record in records):
        needs.append("需要把学业压力拆成更小步骤")
    if any("dorm" in str(record.get("primary_state") or "") for record in records):
        needs.append("需要处理宿舍触发和边界感")
    user_text = " ".join(str(item.get("content") or "") for item in history if item.get("role") == "user")
    if any(term in user_text for term in ("陪", "别走", "不知道怎么办", "难受")):
        needs.append("需要先被接住情绪，而不是立刻被分析")
    return list(dict.fromkeys(needs)) or ["需要自然承接上一轮，不要重新开场"]


def _avoid_next_turn(stage: str, repeated_moves: list[str], boundaries: list[str]) -> list[str]:
    avoid = ["不要重新自我介绍", "不要重复上一轮原句", "不要把心理熵等后台分析直接说给用户"]
    if boundaries or stage == "boundary_building":
        avoid.extend(["不要追问隐私细节", "不要要求用户完整解释原因"])
    if repeated_moves:
        avoid.extend(["不要继续使用同一种问法", "不要连续输出泛泛建议"])
    if stage in {"high_entropy_stabilization", "deteriorating_watch", "safety_priority"}:
        avoid.append("不要开玩笑或切换无关话题")
    return list(dict.fromkeys(avoid))


def _recommended_next_moves(stage: str, domains: list[str], boundaries: list[str]) -> list[str]:
    if stage == "safety_priority":
        return ["先确认现实安全", "鼓励联系身边可信任的人或校园紧急资源"]
    if stage == "human_followup_watch":
        return ["温和说明可以引入现实支持", "询问是否愿意联系辅导员或心理中心"]
    if stage == "boundary_building" or boundaries:
        return ["先明确不会逼用户细说", "给一个不暴露隐私也能做的小动作"]
    if stage in {"high_entropy_stabilization", "deteriorating_watch"}:
        return ["先把当下强度降下来", "只给一个可执行的小步骤"]
    if "academic" in domains:
        return ["承接考试压力", "把下一步缩小到十到十五分钟"]
    if "sleep" in domains:
        return ["承接睡眠受影响", "建议今晚先做低负荷收尾"]
    return ["承接上一轮具体内容", "只问一个和当前情绪有关的问题"]
