from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .logging_utils import get_logger
from .schemas import CampusResource, RiskAssessment, RiskLevel


logger = get_logger("retrieval")


@dataclass(slots=True)
class KnowledgeEntry:
    resource_id: str
    title: str
    category: str
    keywords: list[str]
    summary: str
    recommended_actions: list[str]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text.lower().strip())


class CampusKnowledgeRetriever:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.entries = self._load_entries(Path(settings.campus_kb_path))
        logger.info("Loaded %s campus knowledge entries from %s", len(self.entries), settings.campus_kb_path)

    def retrieve(self, text: str, risk: RiskAssessment, top_k: int = 3) -> list[CampusResource]:
        normalized = _normalize(text)
        scored: list[tuple[int, KnowledgeEntry, list[str]]] = []

        # 用轻量关键词检索先把校园资源接进来，后续再升级成向量检索也不晚。
        for entry in self.entries:
            matched = [keyword for keyword in entry.keywords if keyword and keyword in normalized]
            score = len(matched) * 4

            if risk.level in {RiskLevel.HIGH, RiskLevel.CRITICAL} and entry.category in {"emergency", "counseling"}:
                score += 5
            elif risk.level == RiskLevel.MEDIUM and entry.category in {"counseling", "academic", "health"}:
                score += 2

            if score > 0:
                scored.append((score, entry, matched))

        scored.sort(key=lambda item: item[0], reverse=True)

        selected: list[CampusResource] = []
        for _, entry, matched in scored[:top_k]:
            reason = f"匹配到关键词：{', '.join(matched[:3])}" if matched else f"基于 {risk.level} 风险等级推荐"
            selected.append(
                CampusResource(
                    resource_id=entry.resource_id,
                    title=entry.title,
                    category=entry.category,
                    summary=entry.summary,
                    recommended_actions=entry.recommended_actions,
                    relevance_reason=reason,
                )
            )

        if selected:
            logger.info(
                "Retrieved campus resources for risk=%s: %s",
                risk.level,
                [item.title for item in selected],
            )
            return selected

        logger.warning("No campus knowledge entry matched text; using default counseling fallback.")
        return [
            CampusResource(
                resource_id="psych-center-default",
                title=f"{self.settings.campus_counseling_center}支持入口",
                category="counseling",
                summary="当系统无法明确匹配具体校园场景时，优先推荐心理中心作为通用支持入口。",
                recommended_actions=[
                    f"联系 {self.settings.campus_counseling_center}",
                    f"拨打 {self.settings.campus_counseling_hotline}",
                    f"发送邮件到 {self.settings.campus_counseling_email}",
                ],
                relevance_reason="默认推荐通用校园心理支持资源",
            )
        ]

    @staticmethod
    def _load_entries(path: Path) -> list[KnowledgeEntry]:
        if not path.exists():
            logger.warning("Campus knowledge file not found: %s", path)
            return []

        data = json.loads(path.read_text(encoding="utf-8"))
        entries: list[KnowledgeEntry] = []
        for item in data:
            entries.append(
                KnowledgeEntry(
                    resource_id=str(item["resource_id"]),
                    title=str(item["title"]),
                    category=str(item["category"]),
                    keywords=[str(keyword) for keyword in item.get("keywords", [])],
                    summary=str(item["summary"]),
                    recommended_actions=[str(action) for action in item.get("recommended_actions", [])],
                )
            )
        return entries
