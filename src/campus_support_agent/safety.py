from __future__ import annotations

import re
from collections.abc import Iterable

from .noisy_input import analyze_noisy_distress_text
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

REAL_CRITICAL_TERMS = {
    "自杀",
    "不想活",
    "不活了",
    "死了算了",
    "结束生命",
    "伤害自己",
    "想死",
}

REAL_HIGH_TERMS = {
    "想伤害别人",
    "报复他们",
}

REAL_MEDIUM_TERMS = {
    "焦虑",
    "害怕",
    "担心",
    "压力",
    "难受",
    "想哭",
    "烦躁",
    "睡不着",
    "失眠",
    "睡不好",
    "挂科",
    "考试",
    "宿舍",
    "舍友",
    "室友",
    "怕别人知道",
    "压梨",
    "睡不找",
}


def _normalize(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"\s+", "", lowered)
    return lowered


def _find_terms(text: str, terms: Iterable[str]) -> list[str]:
    normalized = _normalize(text)
    return sorted({term for term in terms if term in normalized})


def evaluate_text_risk(text: str) -> RiskAssessment:
    real_critical_hits = _filter_negated_critical_hits(text, _find_terms(text, REAL_CRITICAL_TERMS))
    if real_critical_hits:
        return RiskAssessment(
            level=RiskLevel.CRITICAL,
            score=95,
            reason="User text includes direct self-harm or life-ending signals.",
            trigger_terms=real_critical_hits,
            needs_human_followup=True,
        )

    real_high_hits = _find_terms(text, REAL_HIGH_TERMS)
    if real_high_hits:
        return RiskAssessment(
            level=RiskLevel.HIGH,
            score=75,
            reason="User text includes loss-of-control or severe escalation signals.",
            trigger_terms=real_high_hits,
            needs_human_followup=True,
        )

    real_medium_hits = _find_terms(text, REAL_MEDIUM_TERMS)

    noisy_analysis = analyze_noisy_distress_text(text)
    if real_medium_hits:
        return RiskAssessment(
            level=RiskLevel.MEDIUM,
            score=45,
            reason="User text includes distress, pressure, sleep, academic, or interpersonal stress signals.",
            trigger_terms=real_medium_hits,
            needs_human_followup=False,
        )

    if "possible_crisis_typo" in noisy_analysis.typo_flags:
        return RiskAssessment(
            level=RiskLevel.CRITICAL,
            score=95,
            reason="疑似存在危机表达的错别字或近音字，按高敏感安全策略处理。",
            trigger_terms=noisy_analysis.inferred_terms,
            needs_human_followup=True,
        )

    if "possible_emotional_escalation_typo" in noisy_analysis.typo_flags:
        if _should_downgrade_contextual_high(text, noisy_analysis.inferred_terms):
            return RiskAssessment(
                level=RiskLevel.MEDIUM,
                score=45,
                reason="检测到强烈痛苦或失控感表达，但上下文更像具体压力场景，先按中等风险持续观察。",
                trigger_terms=noisy_analysis.inferred_terms,
                needs_human_followup=False,
            )
        return RiskAssessment(
            level=RiskLevel.HIGH,
            score=75,
            reason="疑似存在情绪失控或崩溃表达的错别字，建议提高关注等级。",
            trigger_terms=noisy_analysis.inferred_terms,
            needs_human_followup=True,
        )

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
        if _should_downgrade_contextual_high(text, high_hits):
            return RiskAssessment(
                level=RiskLevel.MEDIUM,
                score=45,
                reason="检测到强烈痛苦表达，但上下文更像学业、人际或任务压力，先按中等风险持续观察。",
                trigger_terms=high_hits,
                needs_human_followup=False,
            )
        return RiskAssessment(
            level=RiskLevel.HIGH,
            score=75,
            reason="检测到较强的失控、绝望或自伤风险信号。",
            trigger_terms=high_hits,
            needs_human_followup=True,
        )

    medium_hits = _find_terms(noisy_analysis.analysis_text, MEDIUM_TERMS)
    if "possible_distress_typo" in noisy_analysis.typo_flags:
        medium_hits = sorted({*medium_hits, *noisy_analysis.inferred_terms})
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


def _should_downgrade_contextual_high(text: str, high_hits: list[str]) -> bool:
    normalized = _normalize(text)
    contextual_terms = {"崩溃", "快崩溃", "失控", "控制不住", "受不了", "绝望", "撑不住", "撑不住了", "惊恐"}
    if not high_hits or any(hit not in contextual_terms for hit in high_hits):
        return False
    if any(term in normalized for term in ("自杀", "想死", "不想活", "伤害自己", "活不下去", "轻生", "天台", "遗书")):
        return False
    pressure_context = (
        "考试",
        "期末",
        "复习",
        "挂科",
        "作业",
        "报告",
        "代码",
        "展示",
        "专业",
        "室友",
        "舍友",
        "小组",
        "比赛",
        "项目",
        "分手",
        "朋友圈",
    )
    return any(term in normalized for term in pressure_context)


def _filter_negated_critical_hits(text: str, hits: list[str]) -> list[str]:
    normalized = _normalize(text)
    filtered: list[str] = []
    for hit in hits:
        if hit == "伤害自己" and any(phrase in normalized for phrase in ("不想伤害自己", "不会伤害自己", "没有想伤害自己")):
            continue
        filtered.append(hit)
    return filtered
