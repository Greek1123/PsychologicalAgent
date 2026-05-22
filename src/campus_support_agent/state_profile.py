from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .schemas import PsychologicalEntropy, RiskAssessment, RiskLevel, StateProfile


DOMAIN_TERMS: dict[str, tuple[str, ...]] = {
    "academic": ("考试", "挂科", "期末", "绩点", "作业", "论文", "复习", "成绩", "补考", "考研"),
    "sleep": ("睡不着", "睡不找", "失眠", "熬夜", "睡不好", "醒", "困", "很累", "疲惫"),
    "dorm": ("宿舍", "舍友", "室友", "寝室", "回宿舍"),
    "interpersonal": ("朋友", "同学", "家人", "父母", "老师", "针对", "孤立", "关系", "吵架"),
    "future": ("未来", "毕业", "工作", "就业", "前途", "不知道怎么办"),
    "daily_functioning": ("吃不下", "没胃口", "不想动", "起不来", "什么都不想做"),
    "group_work": ("小组", "组员", "分工", "贡献", "PPT", "资料整理", "老师觉得我没贡献"),
    "public_speaking": ("上台", "汇报", "忘词", "手抖", "脑子空白", "所有人都看着"),
    "game_avoidance": ("打游戏", "游戏到凌晨", "控制不了自己", "不想去上课", "作业、论文、未来"),
    "family_middleman": ("父母离婚", "我妈", "我爸", "没人可以说话", "每天听", "骂爸爸"),
    "pet_grief": ("宠物", "没照顾好", "早点发现", "它会不会还在"),
    "safety_fear": ("跟踪", "尾随", "陌生人", "没有证据", "不敢独自"),
}

EMOTION_TERMS: dict[str, tuple[str, ...]] = {
    "anxiety": ("焦虑", "害怕", "怕", "担心", "慌", "紧张", "不安"),
    "sadness": ("难受", "想哭", "崩溃", "心情不好", "委屈", "痛苦", "低落"),
    "anger": ("烦", "烦躁", "生气", "讨厌", "受不了", "火大"),
    "helplessness": ("没办法", "不知道怎么办", "撑不住", "没用", "无助"),
}

BODY_TERMS: dict[str, tuple[str, ...]] = {
    "sleep_disruption": ("睡不着", "睡不找", "失眠", "睡不好", "熬夜", "醒"),
    "appetite_change": ("吃不下", "没胃口", "暴食", "不想吃"),
    "fatigue": ("累", "疲惫", "没力气", "头疼", "胸闷"),
}

COGNITIVE_TERMS: dict[str, tuple[str, ...]] = {
    "catastrophizing": ("完了", "肯定", "一定会", "毁了", "没救"),
    "rumination": ("一直想", "停不下来", "反复", "脑子里"),
    "overload": ("压力好大", "压梨好大", "太多", "喘不过气", "忙不过来", "乱", "不知道从哪里开始"),
    "self_blame": ("怪我", "我太差", "我不行", "没用", "都是我的错", "自己很差"),
}

BOUNDARY_TERMS: dict[str, tuple[str, ...]] = {
    "privacy_concern": ("怕别人知道", "会告诉别人", "保密", "隐私", "被知道", "告诉别人"),
    "low_disclosure": ("不想说", "不想细说", "不太想细说", "不方便说", "别问", "不知道怎么说"),
}

RISK_TERMS: dict[str, tuple[str, ...]] = {
    "self_harm": ("自杀", "不想活", "死了算了", "伤害自己", "结束生命"),
    "harm_others": ("伤害别人", "报复", "杀了", "弄死"),
    "loss_of_control": ("控制不住", "快崩溃", "撑不住", "受不了了"),
}

WEAK_INPUTS = {"", "?", "？", "??", "？？", "...", "。", "嗯", "哦", "啊", "1", "2", "3", "ok", "OK"}


def build_state_profile(
    text: str,
    *,
    risk: RiskAssessment,
    entropy: PsychologicalEntropy,
    conversation_history: list[dict[str, Any]] | None = None,
    noisy_input_detected: bool = False,
) -> StateProfile:
    """Build a structured state profile for strategy/risk layers.

    The profile is intentionally not written as user-facing advice. It gives the
    backend stable fields to route support, track trends, and avoid exposing
    clinical-sounding analysis in the chat reply.
    """

    clean_text = text.strip()
    history_text = _history_text(conversation_history)
    combined_text = f"{history_text} {clean_text}".strip()

    stress_domains = _match_labels(combined_text, DOMAIN_TERMS)
    emotion_signals = _match_labels(combined_text, EMOTION_TERMS)
    body_signals = _match_labels(combined_text, BODY_TERMS)
    cognitive_signals = _match_labels(combined_text, COGNITIVE_TERMS)
    boundary_flags = _match_labels(combined_text, BOUNDARY_TERMS)
    risk_signals = _match_labels(combined_text, RISK_TERMS)

    weak_input_detected = clean_text in WEAK_INPUTS
    primary_state = _primary_state(
        risk=risk,
        stress_domains=stress_domains,
        emotion_signals=emotion_signals,
        body_signals=body_signals,
        cognitive_signals=cognitive_signals,
        boundary_flags=boundary_flags,
        weak_input_detected=weak_input_detected,
    )
    recommended_focus = _recommended_focus(
        risk=risk,
        primary_state=primary_state,
        boundary_flags=boundary_flags,
        body_signals=body_signals,
        cognitive_signals=cognitive_signals,
        weak_input_detected=weak_input_detected,
    )
    evidence = _evidence(
        clean_text,
        [*DOMAIN_TERMS.values(), *EMOTION_TERMS.values(), *BODY_TERMS.values(), *COGNITIVE_TERMS.values()],
    )

    confidence = _confidence(
        stress_domains=stress_domains,
        emotion_signals=emotion_signals,
        body_signals=body_signals,
        cognitive_signals=cognitive_signals,
        boundary_flags=boundary_flags,
        risk_signals=risk_signals,
        weak_input_detected=weak_input_detected,
        noisy_input_detected=noisy_input_detected,
    )

    return StateProfile(
        primary_state=primary_state,
        intensity=_intensity(entropy=entropy, risk=risk),
        confidence=confidence,
        stress_domains=stress_domains,
        emotion_signals=emotion_signals,
        body_signals=body_signals,
        cognitive_signals=cognitive_signals,
        social_signals=_social_signals(stress_domains),
        boundary_flags=boundary_flags,
        risk_signals=[*risk_signals, *risk.trigger_terms],
        weak_input_detected=weak_input_detected,
        noisy_input_detected=noisy_input_detected,
        recommended_focus=recommended_focus,
        evidence=evidence[:8],
    )


def _match_labels(text: str, table: dict[str, tuple[str, ...]]) -> list[str]:
    labels = [label for label, terms in table.items() if any(term in text for term in terms)]
    return _dedupe(labels)


def _primary_state(
    *,
    risk: RiskAssessment,
    stress_domains: list[str],
    emotion_signals: list[str],
    body_signals: list[str],
    cognitive_signals: list[str],
    boundary_flags: list[str],
    weak_input_detected: bool,
) -> str:
    if risk.level in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
        return "safety_risk"
    if "privacy_concern" in boundary_flags:
        return "privacy_boundary"
    if "low_disclosure" in boundary_flags:
        return "low_disclosure"
    if weak_input_detected:
        return "low_disclosure"
    if "group_work" in stress_domains:
        return "group_work_marginalization"
    if "public_speaking" in stress_domains:
        return "public_speaking_panic"
    if "game_avoidance" in stress_domains:
        return "game_avoidance_loop"
    if "family_middleman" in stress_domains:
        return "family_middleman_stress"
    if "pet_grief" in stress_domains:
        return "pet_grief"
    if "safety_fear" in stress_domains:
        return "campus_safety_fear"
    if "dorm" in stress_domains:
        return "dorm_interpersonal_distress"
    if "academic" in stress_domains and "sleep" in stress_domains:
        return "academic_sleep_stress"
    if "academic" in stress_domains:
        return "academic_pressure"
    if "future" in stress_domains:
        return "future_uncertainty"
    if "sleep_disruption" in body_signals:
        return "sleep_disruption"
    if "self_blame" in cognitive_signals:
        return "self_blame_distress"
    if emotion_signals:
        return f"{emotion_signals[0]}_distress"
    return "general_support"


def _recommended_focus(
    *,
    risk: RiskAssessment,
    primary_state: str,
    boundary_flags: list[str],
    body_signals: list[str],
    cognitive_signals: list[str],
    weak_input_detected: bool,
) -> str:
    if risk.level in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
        return "safety_and_human_referral"
    if "privacy_concern" in boundary_flags:
        return "confidentiality_and_control"
    if weak_input_detected or "low_disclosure" in boundary_flags:
        return "low_pressure_presence"
    if primary_state == "group_work_marginalization":
        return "contribution_visibility"
    if primary_state == "public_speaking_panic":
        return "performance_grounding"
    if primary_state == "game_avoidance_loop":
        return "escape_loop_interruption"
    if primary_state == "family_middleman_stress":
        return "family_boundary_sustainability"
    if primary_state == "pet_grief":
        return "grief_without_self_blame"
    if primary_state == "campus_safety_fear":
        return "safety_reporting_without_blame"
    if "sleep_disruption" in body_signals:
        return "sleep_stabilization_first"
    if "catastrophizing" in cognitive_signals or "overload" in cognitive_signals:
        return "grounding_then_small_next_step"
    if primary_state == "dorm_interpersonal_distress":
        return "safe_space_and_boundary_options"
    return "supportive_listening"


def _intensity(*, entropy: PsychologicalEntropy, risk: RiskAssessment) -> int:
    risk_floor = {
        RiskLevel.LOW: 1,
        RiskLevel.MEDIUM: 3,
        RiskLevel.HIGH: 7,
        RiskLevel.CRITICAL: 9,
    }[risk.level]
    entropy_scaled = max(1, min(10, round(entropy.score / 10)))
    return max(risk_floor, entropy_scaled)


def _confidence(
    *,
    stress_domains: list[str],
    emotion_signals: list[str],
    body_signals: list[str],
    cognitive_signals: list[str],
    boundary_flags: list[str],
    risk_signals: list[str],
    weak_input_detected: bool,
    noisy_input_detected: bool,
) -> float:
    signal_count = sum(
        len(items)
        for items in (
            stress_domains,
            emotion_signals,
            body_signals,
            cognitive_signals,
            boundary_flags,
            risk_signals,
        )
    )
    base = 0.35 + min(0.45, signal_count * 0.07)
    if weak_input_detected:
        base -= 0.12
    if noisy_input_detected:
        base -= 0.08
    return round(max(0.25, min(0.92, base)), 2)


def _social_signals(stress_domains: list[str]) -> list[str]:
    signals: list[str] = []
    if "dorm" in stress_domains:
        signals.append("shared_living_tension")
    if "interpersonal" in stress_domains:
        signals.append("relationship_stress")
    return signals


def _evidence(text: str, term_groups: Iterable[tuple[str, ...]]) -> list[str]:
    hits: list[str] = []
    for terms in term_groups:
        for term in terms:
            if term in text:
                hits.append(term)
    return _dedupe(hits)


def _history_text(conversation_history: list[dict[str, Any]] | None) -> str:
    if not conversation_history:
        return ""
    return " ".join(str(item.get("content", "")) for item in conversation_history[-6:])


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
