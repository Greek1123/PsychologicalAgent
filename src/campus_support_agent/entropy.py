from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .logging_utils import get_logger
from .schemas import EntropyDimensions, EntropyTrend, PsychologicalEntropy, RiskAssessment, RiskLevel


logger = get_logger("entropy")


EMOTION_TERMS = {
    "焦虑",
    "难过",
    "害怕",
    "烦躁",
    "绝望",
    "崩溃",
    "痛苦",
    "压抑",
}

VOLATILITY_TERMS = {
    "失控",
    "撑不住",
    "控制不住",
    "反复",
    "突然",
    "崩溃",
    "惊恐",
}

COGNITIVE_LOAD_TERMS = {
    "内耗",
    "想太多",
    "脑子很乱",
    "担心",
    "挂科",
    "考试",
    "论文",
    "答辩",
    "未来",
    "做不完",
}

PHYSIOLOGICAL_TERMS = {
    "失眠",
    "睡不好",
    "睡不着",
    "头痛",
    "胸闷",
    "心慌",
    "疲惫",
    "吃不下",
}

SOCIAL_TERMS = {
    "室友",
    "人际",
    "孤独",
    "没人理解",
    "家里",
    "家庭",
    "关系紧张",
    "矛盾",
    "朋友",
}


def _find_hits(text: str, terms: Iterable[str]) -> list[str]:
    return sorted({term for term in terms if term in text})


def _score_from_hits(hit_count: int, *, base: int = 0, bonus: int = 0) -> int:
    return max(0, min(5, base + hit_count + bonus))


def _risk_pressure_from_level(level: RiskLevel) -> int:
    mapping = {
        RiskLevel.LOW: 1,
        RiskLevel.MEDIUM: 2,
        RiskLevel.HIGH: 4,
        RiskLevel.CRITICAL: 5,
    }
    return mapping[level]


def _balance_state_from_score(score: int, risk: RiskAssessment) -> str:
    if risk.level == RiskLevel.CRITICAL or score >= 85:
        return "crisis"
    if risk.level == RiskLevel.HIGH or score >= 65:
        return "fragile"
    if score >= 40:
        return "strained"
    return "stable"


def _level_from_score(score: int) -> int:
    if score >= 85:
        return 5
    if score >= 65:
        return 4
    if score >= 45:
        return 3
    if score >= 25:
        return 2
    return 1


def _top_drivers(driver_hits: dict[str, list[str]]) -> list[str]:
    ranked = sorted(driver_hits.items(), key=lambda item: len(item[1]), reverse=True)
    drivers: list[str] = []
    for label, hits in ranked:
        if hits:
            drivers.append(f"{label}({', '.join(hits[:2])})")
    return drivers[:3] or ["当前对话未出现明显高熵驱动，建议持续观察"]


def _rank_driver_tags(
    *,
    dimensions: EntropyDimensions,
    emotion_hits: list[str],
    volatility_hits: list[str],
    cognitive_hits: list[str],
    physiological_hits: list[str],
    social_hits: list[str],
    risk: RiskAssessment,
) -> list[str]:
    ranked = [
        ("emotion_intensity", dimensions.emotion_intensity, emotion_hits),
        ("emotional_volatility", dimensions.emotional_volatility, volatility_hits),
        ("cognitive_load", dimensions.cognitive_load, cognitive_hits),
        ("physiological_imbalance", dimensions.physiological_imbalance, physiological_hits),
        ("social_support_tension", dimensions.social_support_tension, social_hits),
        ("risk_pressure", dimensions.risk_pressure, risk.trigger_terms),
    ]
    ranked.sort(key=lambda item: (item[1], len(item[2])), reverse=True)
    return [tag for tag, score, hits in ranked if score >= 2 or hits][:3]


def evaluate_psychological_entropy(
    text: str,
    risk: RiskAssessment,
    student_context: dict[str, Any] | None = None,
    conversation_history: list[dict[str, Any]] | None = None,
) -> PsychologicalEntropy:
    clean_text = text.strip()
    history = conversation_history or []
    del student_context

    # 这一层先用可解释规则做熵评估，便于你后续把规则标签迁移成训练数据。
    emotion_hits = _find_hits(clean_text, EMOTION_TERMS)
    volatility_hits = _find_hits(clean_text, VOLATILITY_TERMS)
    cognitive_hits = _find_hits(clean_text, COGNITIVE_LOAD_TERMS)
    physiological_hits = _find_hits(clean_text, PHYSIOLOGICAL_TERMS)
    social_hits = _find_hits(clean_text, SOCIAL_TERMS)

    history_bonus = 1 if len(history) >= 4 else 0
    exam_bonus = 1 if any(token in clean_text for token in ["考试", "答辩", "论文"]) else 0
    isolation_bonus = 1 if any(token in clean_text for token in ["自己扛", "不敢说", "没人帮", "没人理解"]) else 0

    dimensions = EntropyDimensions(
        emotion_intensity=_score_from_hits(len(emotion_hits), base=1),
        emotional_volatility=_score_from_hits(len(volatility_hits), bonus=history_bonus),
        cognitive_load=_score_from_hits(len(cognitive_hits), base=1, bonus=exam_bonus),
        physiological_imbalance=_score_from_hits(len(physiological_hits), bonus=history_bonus),
        social_support_tension=_score_from_hits(len(social_hits), bonus=isolation_bonus),
        risk_pressure=_risk_pressure_from_level(risk.level),
    )

    total_points = (
        dimensions.emotion_intensity
        + dimensions.emotional_volatility
        + dimensions.cognitive_load
        + dimensions.physiological_imbalance
        + dimensions.social_support_tension
        + dimensions.risk_pressure
    )
    score = round(total_points / 30 * 100)
    level = _level_from_score(score)
    balance_state = _balance_state_from_score(score, risk)

    drivers = _top_drivers(
        {
            "情绪强度": emotion_hits,
            "认知负荷": cognitive_hits,
            "生理失衡": physiological_hits,
            "社会张力": social_hits,
            "情绪波动": volatility_hits,
        }
    )
    driver_tags = _rank_driver_tags(
        dimensions=dimensions,
        emotion_hits=emotion_hits,
        volatility_hits=volatility_hits,
        cognitive_hits=cognitive_hits,
        physiological_hits=physiological_hits,
        social_hits=social_hits,
        risk=risk,
    )

    entropy = PsychologicalEntropy(
        score=score,
        level=level,
        balance_state=balance_state,
        driver_tags=driver_tags,
        dominant_drivers=drivers,
        dimensions=dimensions,
        trend=EntropyTrend(
            previous_score=None,
            delta=None,
            direction="baseline",
        ),
    )
    logger.info(
        "Entropy evaluated score=%s level=%s state=%s drivers=%s",
        entropy.score,
        entropy.level,
        entropy.balance_state,
        entropy.dominant_drivers,
    )
    return entropy
