from __future__ import annotations

import re
from collections.abc import Iterable

from .schemas import RiskAssessment, RiskLevel


CRITICAL_TERMS = {
    "自杀",
    "想死",
    "不想活",
    "结束生命",
    "轻生",
    "跳楼",
    "割腕",
    "服药过量",
    "杀了自己",
    "杀人",
    "伤害他人",
}

HIGH_TERMS = {
    "自残",
    "伤害自己",
    "活不下去",
    "撑不住了",
    "崩溃",
    "失控",
    "绝望",
    "没人能帮我",
    "连续失眠",
    "惊恐",
}

MEDIUM_TERMS = {
    "焦虑",
    "压力",
    "睡不好",
    "失眠",
    "烦躁",
    "孤独",
    "无助",
    "害怕",
    "难过",
    "内耗",
    "挂科",
    "论文",
    "人际",
    "室友",
}


def _normalize(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"\s+", "", lowered)
    return lowered


def _find_terms(text: str, terms: Iterable[str]) -> list[str]:
    normalized = _normalize(text)
    return sorted({term for term in terms if term in normalized})


def evaluate_text_risk(text: str) -> RiskAssessment:
    # 先做高召回的规则筛查，保证危机词不会被模型生成过程稀释掉。
    critical_hits = _find_terms(text, CRITICAL_TERMS)
    if critical_hits:
        return RiskAssessment(
            level=RiskLevel.CRITICAL,
            score=95,
            reason="检测到明显的生命安全或严重伤害相关表达。",
            trigger_terms=critical_hits,
            needs_human_followup=True,
        )

    high_hits = _find_terms(text, HIGH_TERMS)
    if high_hits:
        return RiskAssessment(
            level=RiskLevel.HIGH,
            score=75,
            reason="检测到较强的失控、绝望或自伤风险信号。",
            trigger_terms=high_hits,
            needs_human_followup=True,
        )

    medium_hits = _find_terms(text, MEDIUM_TERMS)
    if medium_hits:
        return RiskAssessment(
            level=RiskLevel.MEDIUM,
            score=45,
            reason="检测到较明确的压力、睡眠或情绪困扰信号。",
            trigger_terms=medium_hits,
            needs_human_followup=False,
        )

    return RiskAssessment(
        level=RiskLevel.LOW,
        score=20,
        reason="未检测到显著危机词，但仍需结合上下文持续观察。",
        trigger_terms=[],
        needs_human_followup=False,
    )
