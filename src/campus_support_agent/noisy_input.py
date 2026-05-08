from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(slots=True)
class NoisyInputAnalysis:
    original_text: str
    analysis_text: str
    inferred_terms: list[str] = field(default_factory=list)
    typo_flags: list[str] = field(default_factory=list)

    @property
    def has_inference(self) -> bool:
        return bool(self.inferred_terms or self.typo_flags)


CRITICAL_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"想\s*[死似4sS]", "想死"),
    (r"不\s*想\s*[活或伙霍]", "不想活"),
    (r"[活或伙霍]\s*不\s*下\s*去", "活不下去"),
    (r"自\s*[杀殺鲨沙砂煞sS]", "自杀"),
    (r"结束\s*[生身]\s*命", "结束生命"),
    (r"[伤傷删]\s*害\s*[自字]\s*己", "伤害自己"),
)

HIGH_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"控\s*制\s*不\s*[住主]", "控制不住"),
    (r"撑\s*不\s*[住主]", "撑不住"),
    (r"快\s*[崩奔蹦]\s*溃", "快崩溃"),
    (r"[崩奔蹦]\s*溃", "崩溃"),
    (r"受\s*不\s*[了鸟]", "受不了"),
)

MEDIUM_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"[难南蓝男]\s*[受瘦]", "难受"),
    (r"好\s*[难南蓝男]\s*[受瘦]", "好难受"),
    (r"[压鸭]\s*力", "压力"),
    (r"[焦蕉]\s*[虑绿]", "焦虑"),
    (r"[害海]\s*怕", "害怕"),
    (r"睡\s*不\s*[着找召]", "睡不着"),
    (r"睡\s*不\s*好", "睡不好"),
    (r"[挂瓜]\s*科", "挂科"),
    (r"[宿数]\s*舍", "宿舍"),
    (r"[舍室]\s*友", "舍友"),
    (r"[父付]\s*母\s*吵\s*架", "父母吵架"),
)


def _compact(text: str) -> str:
    return re.sub(r"[\s，。！？!?,.;；：:、~～…]+", "", text.strip())


def _infer_terms(compact_text: str, patterns: tuple[tuple[str, str], ...]) -> list[str]:
    inferred: list[str] = []
    for pattern, canonical in patterns:
        if re.search(pattern, compact_text, flags=re.IGNORECASE) and canonical not in inferred:
            inferred.append(canonical)
    return inferred


def analyze_noisy_distress_text(text: str) -> NoisyInputAnalysis:
    """Infer likely distress terms from typo-heavy Chinese user input.

    This does not rewrite what we store as the user's message. It only adds a
    cautious analysis suffix for risk/entropy/policy layers, so misspellings like
    "我不想或了" are handled as potential crisis signals.
    """

    original = text.strip()
    compact = _compact(original)
    inferred: list[str] = []
    flags: list[str] = []

    critical = _infer_terms(compact, CRITICAL_PATTERNS)
    high = _infer_terms(compact, HIGH_PATTERNS)
    medium = _infer_terms(compact, MEDIUM_PATTERNS)

    for term in [*critical, *high, *medium]:
        if term not in inferred:
            inferred.append(term)

    if critical:
        flags.append("possible_crisis_typo")
    if high:
        flags.append("possible_emotional_escalation_typo")
    if medium:
        flags.append("possible_distress_typo")

    if inferred:
        suffix = "。可能存在情绪崩溃时的错别字，系统推断关键词：" + "，".join(inferred)
        analysis_text = f"{original}{suffix}"
    else:
        analysis_text = original

    return NoisyInputAnalysis(
        original_text=original,
        analysis_text=analysis_text,
        inferred_terms=inferred,
        typo_flags=flags,
    )
