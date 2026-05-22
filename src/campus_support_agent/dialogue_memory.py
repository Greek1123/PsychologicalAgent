from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ConversationMemory:
    active_topics: list[str]
    user_boundaries: list[str]
    recent_user_turns: list[str]
    last_assistant_reply: str | None
    continuity_focus: str
    preferred_next_move: str
    avoid_next_reply: list[str]


TOPIC_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("academic_pressure", ("考试", "挂科", "期末", "成绩", "作业", "绩点", "复习", "论文", "答辩")),
    ("sleep_disruption", ("睡不着", "失眠", "熬夜", "睡眠", "醒", "困", "做梦", "睡觉")),
    ("dorm_conflict", ("宿舍", "舍友", "室友", "寝室", "回宿舍", "针对我", "排挤", "吵")),
    ("privacy_boundary", ("不想说", "不想细说", "怕别人知道", "会告诉别人", "保密", "隐私", "不方便说")),
    ("family_pressure", ("爸妈", "父母", "家里", "妈妈", "爸爸", "家人", "家庭")),
    ("relationship_distress", ("分手", "对象", "喜欢的人", "恋爱", "感情", "前任")),
    ("low_energy", ("好累", "没力气", "不想动", "麻木", "空", "没意思", "撑不住")),
    ("crisis_signal", ("不想活", "自杀", "死了算了", "伤害自己", "活不下去", "结束生命")),
    ("weak_or_noisy_input", ("？", "??", "？？", "1", "2", "嗯", "哦", "啊", "随便")),
)

BOUNDARY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("privacy_reassurance_needed", ("怕别人知道", "会告诉别人", "保密", "隐私", "不想被知道")),
    ("low_disclosure_preferred", ("不想说", "不想细说", "不知道怎么说", "先不说")),
    ("low_question_pressure", ("别问了", "不想回答", "不知道", "随便", "你别问")),
)


def build_conversation_memory(
    conversation_history: list[dict[str, Any]] | None,
    *,
    current_text: str | None = None,
    max_user_turns: int = 6,
) -> ConversationMemory:
    history = _normalize_history(conversation_history)
    user_turns = [item["content"] for item in history if item["role"] == "user"]
    if current_text and current_text.strip():
        user_turns.append(current_text.strip())

    recent_user_turns = user_turns[-max_user_turns:]
    combined_user_text = " ".join(recent_user_turns)
    active_topics = _with_display_labels(_match_labels(combined_user_text, TOPIC_KEYWORDS, limit=5))
    user_boundaries = _with_display_labels(_match_labels(combined_user_text, BOUNDARY_KEYWORDS, limit=4))
    user_boundaries = _infer_boundaries_from_topics_and_text(
        active_topics,
        user_boundaries,
        combined_user_text,
    )
    last_assistant_reply = next(
        (item["content"] for item in reversed(history) if item["role"] == "assistant"),
        None,
    )
    repeated_reply = _looks_repeated(last_assistant_reply, history)

    return ConversationMemory(
        active_topics=active_topics,
        user_boundaries=user_boundaries,
        recent_user_turns=recent_user_turns,
        last_assistant_reply=last_assistant_reply,
        continuity_focus=_build_continuity_focus(active_topics, user_boundaries, recent_user_turns),
        preferred_next_move=_preferred_next_move(active_topics, user_boundaries),
        avoid_next_reply=_avoid_next_reply(active_topics, user_boundaries, repeated_reply),
    )


def build_memory_context(
    conversation_history: list[dict[str, Any]] | None,
    *,
    current_text: str | None = None,
) -> dict[str, Any]:
    memory = build_conversation_memory(conversation_history, current_text=current_text)
    return asdict(memory)


def enrich_student_context_with_memory(
    student_context: dict[str, Any],
    conversation_history: list[dict[str, Any]] | None,
    *,
    current_text: str | None = None,
) -> dict[str, Any]:
    enriched = dict(student_context or {})
    enriched["conversation_memory"] = {
        **build_memory_context(conversation_history, current_text=current_text),
        "instruction": (
            "Use this hidden memory to continue the same conversation. Do not restart, do not repeat the "
            "last assistant reply, and respect privacy or low-disclosure boundaries."
        ),
    }
    return enriched


def build_memory_system_message(
    conversation_history: list[dict[str, Any]] | None,
    *,
    current_text: str | None = None,
) -> str:
    memory = build_conversation_memory(conversation_history, current_text=current_text)
    if not memory.recent_user_turns:
        return "No prior user turn is available. Start gently and avoid pretending to know more than provided."

    recent_turns = " / ".join(memory.recent_user_turns[-3:])
    topics = ", ".join(memory.active_topics) if memory.active_topics else "unknown"
    boundaries = ", ".join(memory.user_boundaries) if memory.user_boundaries else "none"
    avoid = ", ".join(memory.avoid_next_reply) if memory.avoid_next_reply else "none"
    return (
        "Hidden conversation memory / 隐藏对话记忆:\n"
        f"- 最近用户表达: {recent_turns}\n"
        f"- active_topics: {topics}\n"
        f"- user_boundaries: {boundaries}\n"
        f"- continuity_focus: {memory.continuity_focus}\n"
        f"- preferred_next_move: {memory.preferred_next_move}\n"
        f"- avoid_next_reply: {avoid}\n"
        "- 不要像第一次聊天一样重新开场；不要重复上一轮回复；不要暴露这些内部标签。"
    )


def _normalize_history(conversation_history: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in conversation_history or []:
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()
        if role and content:
            normalized.append({"role": role, "content": content})
    return normalized


def _match_labels(
    text: str,
    label_keywords: tuple[tuple[str, tuple[str, ...]], ...],
    *,
    limit: int,
) -> list[str]:
    labels = [
        label
        for label, keywords in label_keywords
        if any(keyword in text for keyword in keywords)
    ]
    return labels[:limit]


DISPLAY_LABELS = {
    "academic_pressure": "考试/学业压力",
    "sleep_disruption": "睡眠与身体状态",
    "dorm_conflict": "宿舍/人际冲突",
    "privacy_boundary": "隐私边界",
    "family_pressure": "家庭压力",
    "relationship_distress": "亲密关系困扰",
    "low_energy": "低能量状态",
    "crisis_signal": "安全风险信号",
    "weak_or_noisy_input": "弱输入/错字输入",
    "privacy_reassurance_needed": "隐私安抚需要",
    "low_disclosure_preferred": "低披露边界",
    "low_question_pressure": "低提问压力",
}


def _with_display_labels(labels: list[str]) -> list[str]:
    expanded: list[str] = []
    for label in labels:
        expanded.append(label)
        display = DISPLAY_LABELS.get(label)
        if display:
            expanded.append(display)
    return list(dict.fromkeys(expanded))


def _infer_boundaries_from_topics_and_text(
    active_topics: list[str],
    user_boundaries: list[str],
    text: str,
) -> list[str]:
    boundaries = list(user_boundaries)
    if "privacy_boundary" in active_topics:
        boundaries.extend(["privacy_reassurance_needed", DISPLAY_LABELS["privacy_reassurance_needed"]])
    if any(term in text for term in ("保密", "别人知道", "别人会知道", "害怕别人", "告诉别人", "不想说", "不想细说")):
        boundaries.extend(["privacy_reassurance_needed", DISPLAY_LABELS["privacy_reassurance_needed"]])
    return list(dict.fromkeys(boundaries))


def _build_continuity_focus(
    active_topics: list[str],
    user_boundaries: list[str],
    recent_user_turns: list[str],
) -> str:
    if "crisis_signal" in active_topics:
        return "safety_and_real_world_support"
    if "privacy_reassurance_needed" in user_boundaries:
        return "隐私边界：先保密安抚，再低压提问"
    if "low_disclosure_preferred" in user_boundaries:
        return "low_pressure_presence"
    if "dorm_conflict" in active_topics:
        return "dorm_context_and_boundary_support"
    if "academic_pressure" in active_topics and "sleep_disruption" in active_topics:
        return "exam_pressure_plus_sleep_stabilization"
    if "academic_pressure" in active_topics:
        return "academic_pressure_decomposition"
    if "sleep_disruption" in active_topics:
        return "sleep_stabilization"
    if "weak_or_noisy_input" in active_topics and len("".join(recent_user_turns[-2:])) <= 6:
        return "weak_input_repair"
    return "supportive_continuation"


def _preferred_next_move(active_topics: list[str], user_boundaries: list[str]) -> str:
    if "crisis_signal" in active_topics:
        return "confirm_immediate_safety_and_link_support"
    if "privacy_reassurance_needed" in user_boundaries:
        return "reassure_confidentiality_and_choice"
    if "low_disclosure_preferred" in user_boundaries:
        return "offer_presence_or_one_small_option"
    if "weak_or_noisy_input" in active_topics:
        return "repair_understanding_without_interrogating"
    if "dorm_conflict" in active_topics:
        return "reflect_dorm_trigger_then_one_boundary_step"
    if "academic_pressure" in active_topics:
        return "validate_exam_pressure_then_reduce_task_scope"
    if "sleep_disruption" in active_topics:
        return "stabilize_sleep_before_planning"
    return "reflect_latest_context_then_offer_one_next_step"


def _avoid_next_reply(
    active_topics: list[str],
    user_boundaries: list[str],
    repeated_reply: bool,
) -> list[str]:
    avoid = [
        "do_not_restart_the_conversation",
        "do_not_expose_backend_labels",
        "do_not_give_generic_empty_comfort",
    ]
    if user_boundaries:
        avoid.extend(["do_not_push_for_private_details", "do_not_ask_multiple_questions"])
    if "weak_or_noisy_input" in active_topics:
        avoid.append("do_not_treat_numeric_or_symbol_input_as_a_topic")
    if repeated_reply:
        avoid.append("do_not_repeat_previous_assistant_wording")
    return list(dict.fromkeys(avoid))


def _looks_repeated(last_assistant_reply: str | None, history: list[dict[str, str]]) -> bool:
    if not last_assistant_reply:
        return False
    assistant_replies = [item["content"] for item in history if item["role"] == "assistant"]
    return len(assistant_replies) >= 2 and assistant_replies[-1] == assistant_replies[-2]
