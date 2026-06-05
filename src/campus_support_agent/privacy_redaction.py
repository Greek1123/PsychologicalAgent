from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PrivacyRedactionSummary:
    total_redactions: int = 0
    categories: dict[str, int] = field(default_factory=dict)

    def add(self, category: str, count: int) -> None:
        if count <= 0:
            return
        self.total_redactions += count
        self.categories[category] = self.categories.get(category, 0) + count

    def merge(self, other: "PrivacyRedactionSummary") -> None:
        self.total_redactions += other.total_redactions
        for category, count in other.categories.items():
            self.categories[category] = self.categories.get(category, 0) + count

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_redactions": self.total_redactions,
            "categories": dict(sorted(self.categories.items())),
        }


REDACTION_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("phone", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "[手机号]"),
    ("email", re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![A-Za-z0-9._%+-])"), "[邮箱]"),
    ("id_card", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)|(?<!\d)\d{15}(?!\d)"), "[证件号]"),
    ("student_id", re.compile(r"(学号|学生号|工号)[:：]?\s*[A-Za-z0-9_-]{6,20}"), r"\1:[编号]"),
    ("wechat", re.compile(r"(微信|vx|VX|V信)[:：]?\s*[A-Za-z][A-Za-z0-9_-]{5,19}"), r"\1:[账号]"),
    ("qq", re.compile(r"(QQ|qq)[:：]?\s*[1-9]\d{4,11}"), r"\1:[账号]"),
)


TEXT_KEYS_TO_REDACT = {
    "input_text",
    "transcript",
    "reply_text",
    "latest_reply_text",
    "note",
    "user_note",
    "next_action",
    "summary",
    "reason",
    "text",
    "content",
}


def redact_private_identifiers(text: str) -> tuple[str, PrivacyRedactionSummary]:
    summary = PrivacyRedactionSummary()
    redacted = text
    for category, pattern, replacement in REDACTION_PATTERNS:
        redacted, count = pattern.subn(replacement, redacted)
        summary.add(category, count)
    return redacted, summary


def redact_private_identifiers_in_value(
    value: Any,
    *,
    redact_all_strings: bool = False,
) -> tuple[Any, PrivacyRedactionSummary]:
    summary = PrivacyRedactionSummary()
    if isinstance(value, str):
        redacted, text_summary = redact_private_identifiers(value)
        summary.merge(text_summary)
        return redacted, summary
    if isinstance(value, list):
        redacted_items = []
        for item in value:
            redacted_item, item_summary = redact_private_identifiers_in_value(
                item,
                redact_all_strings=redact_all_strings,
            )
            redacted_items.append(redacted_item)
            summary.merge(item_summary)
        return redacted_items, summary
    if isinstance(value, dict):
        redacted_dict: dict[str, Any] = {}
        for key, item in value.items():
            should_redact = redact_all_strings or key in TEXT_KEYS_TO_REDACT
            if should_redact:
                redacted_item, item_summary = redact_private_identifiers_in_value(
                    item,
                    redact_all_strings=True,
                )
            elif isinstance(item, (dict, list)):
                redacted_item, item_summary = redact_private_identifiers_in_value(
                    item,
                    redact_all_strings=False,
                )
            else:
                redacted_item = item
                item_summary = PrivacyRedactionSummary()
            redacted_dict[key] = redacted_item
            summary.merge(item_summary)
        return redacted_dict, summary
    return value, summary
