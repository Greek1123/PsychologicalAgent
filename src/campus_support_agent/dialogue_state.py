from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class DialogueStage(StrEnum):
    CASUAL = "casual"
    SUPPORT_OPENING = "support_opening"
    PRIVACY_BOUNDARY = "privacy_boundary"
    DISCLOSURE_BOUNDARY = "disclosure_boundary"
    WEAK_INPUT = "weak_input"
    DORM_DISTRESS = "dorm_distress"
    SLEEP_PRESSURE = "sleep_pressure"
    ACADEMIC_PRESSURE = "academic_pressure"
    INTERPERSONAL_STRESS = "interpersonal_stress"
    GENERAL_SUPPORT = "general_support"


@dataclass(frozen=True, slots=True)
class DialogueState:
    stage: DialogueStage
    confidence: float
    should_avoid_questions: bool = False
    should_avoid_advice: bool = False
    should_preserve_privacy: bool = False
    user_wants_light_contact: bool = False


def classify_dialogue_state(
    user_text: str,
    *,
    conversation_history: list[dict[str, Any]] | None = None,
) -> DialogueState:
    text = user_text.strip()
    history_text = _history_text(conversation_history)

    if _is_privacy(text):
        return DialogueState(
            DialogueStage.PRIVACY_BOUNDARY,
            0.95,
            should_avoid_questions=True,
            should_preserve_privacy=True,
            user_wants_light_contact=True,
        )
    if _is_disclosure_boundary(text):
        return DialogueState(
            DialogueStage.DISCLOSURE_BOUNDARY,
            0.9,
            should_avoid_questions=True,
            user_wants_light_contact=True,
        )
    if _is_weak(text):
        return DialogueState(
            DialogueStage.WEAK_INPUT,
            0.88,
            should_avoid_questions=_recent_boundary(history_text),
            should_avoid_advice=not _has_distress(history_text),
            user_wants_light_contact=True,
        )
    if _is_dorm(text):
        return DialogueState(DialogueStage.DORM_DISTRESS, 0.86)
    if _is_sleep_pressure(text):
        return DialogueState(DialogueStage.SLEEP_PRESSURE, 0.84)
    if _is_academic(text):
        return DialogueState(DialogueStage.ACADEMIC_PRESSURE, 0.8)
    if _is_interpersonal(text):
        return DialogueState(DialogueStage.INTERPERSONAL_STRESS, 0.78)
    if _is_casual(text):
        return DialogueState(DialogueStage.CASUAL, 0.7, user_wants_light_contact=True)
    if _has_distress(text):
        return DialogueState(DialogueStage.SUPPORT_OPENING, 0.72)
    return DialogueState(DialogueStage.GENERAL_SUPPORT, 0.55)


def _is_privacy(text: str) -> bool:
    return any(term in text for term in ("怕别人知道", "害怕别人会知道", "怕你会告诉", "隐私", "保密", "传出去"))


def _is_disclosure_boundary(text: str) -> bool:
    return any(term in text for term in ("不想说", "不太想细说", "不想细说", "不想被追问", "算了", "先不说"))


def _is_weak(text: str) -> bool:
    return text in {"", "?", "？", "...", "。", "嗯", "啊", "哦", "1", "2", "3", "ok", "OK"}


def _is_dorm(text: str) -> bool:
    return any(term in text for term in ("宿舍", "寝室", "舍友")) and any(
        term in text for term in ("烦", "压抑", "难受", "针对", "不想回", "吵", "冷落")
    )


def _is_sleep_pressure(text: str) -> bool:
    return any(term in text for term in ("睡不好", "失眠", "睡不着", "晚上睡", "睡眠")) and any(
        term in text for term in ("压力", "焦虑", "心慌", "烦", "考试", "论文")
    )


def _is_academic(text: str) -> bool:
    return any(term in text for term in ("考试", "挂科", "论文", "作业", "复习", "绩点", "导师", "截止"))


def _is_interpersonal(text: str) -> bool:
    return any(term in text for term in ("朋友", "舍友", "同学", "关系", "针对", "排挤", "冷落", "喜欢的人"))


def _is_casual(text: str) -> bool:
    return any(term in text for term in ("奶茶", "电影", "天气", "咖啡", "吃什么", "综艺"))


def _has_distress(text: str) -> bool:
    return any(term in text for term in ("压力", "睡不好", "宿舍", "烦", "害怕", "难受", "焦虑", "考试", "慌", "压抑"))


def _recent_boundary(history_text: str) -> bool:
    return _is_privacy(history_text) or _is_disclosure_boundary(history_text)


def _history_text(conversation_history: list[dict[str, Any]] | None) -> str:
    if not conversation_history:
        return ""
    return " ".join(str(item.get("content", "")) for item in conversation_history[-6:])
