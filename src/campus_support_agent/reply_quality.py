from __future__ import annotations

from collections import Counter
from typing import Any


def build_reply_quality_timeline(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    previous_reply = ""
    for record in records:
        reply_text = str(record.get("reply_text") or "").strip()
        input_text = str(record.get("input_text") or "").strip()
        issues = _detect_reply_issues(
            user_text=input_text,
            reply_text=reply_text,
            previous_reply=previous_reply,
        )
        timeline.append(
            {
                "response_id": record.get("response_id"),
                "created_at": record.get("created_at"),
                "input_text": input_text,
                "reply_preview": reply_text[:180],
                "quality_score": _quality_score(issues),
                "needs_review": bool(issues),
                "issues": issues,
                "risk_level": record.get("risk_level"),
                "entropy_score": record.get("entropy_score"),
                "care_route": record.get("orchestration_route"),
                "adjustment_loop_action": record.get("adjustment_loop_action"),
            }
        )
        if reply_text:
            previous_reply = reply_text
    return timeline


def summarize_reply_quality(timeline: list[dict[str, Any]]) -> dict[str, Any]:
    issue_counter: Counter[str] = Counter()
    needs_review = 0
    scores: list[int] = []
    for item in timeline:
        scores.append(int(item.get("quality_score") or 0))
        issues = [str(issue) for issue in item.get("issues") or []]
        if issues:
            needs_review += 1
        issue_counter.update(issues)
    return {
        "total_replies": len(timeline),
        "needs_review": needs_review,
        "clean_replies": max(len(timeline) - needs_review, 0),
        "average_quality_score": round(sum(scores) / len(scores), 2) if scores else None,
        "issue_counts": dict(issue_counter),
        "top_issues": [
            {"issue": issue, "count": count}
            for issue, count in issue_counter.most_common(8)
        ],
    }


def build_reply_quality_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    timeline = build_reply_quality_timeline(records)
    return {
        "summary": summarize_reply_quality(timeline),
        "timeline": timeline,
    }


def _detect_reply_issues(*, user_text: str, reply_text: str, previous_reply: str) -> list[str]:
    issues: list[str] = []
    compact_reply = _compact(reply_text)
    compact_user = _compact(user_text)
    if not compact_reply:
        return ["empty_reply"]
    if _looks_mojibake(compact_reply):
        issues.append("mojibake_or_encoding_artifact")
    if _exposes_backend_terms(compact_reply):
        issues.append("backend_terms_exposed")
    if _assistant_claims_personal_experience(compact_reply):
        issues.append("assistant_role_drift")
    if _too_short(compact_reply) and _has_distress_context(compact_user):
        issues.append("too_short_for_distress")
    if _is_numeric_or_symbol_input(compact_user) and _continues_numeric_pattern(compact_reply):
        issues.append("numeric_pattern_continuation")
    if previous_reply and _compact(previous_reply) == compact_reply:
        issues.append("repeated_previous_reply")
    if _pushes_for_private_details(compact_reply) and _privacy_boundary(compact_user):
        issues.append("pushes_after_privacy_boundary")
    if _generic_advice(compact_reply) and _has_distress_context(compact_user):
        issues.append("generic_advice_under_distress")
    return _dedupe(issues)


def _quality_score(issues: list[str]) -> int:
    score = 100
    penalties = {
        "empty_reply": 100,
        "mojibake_or_encoding_artifact": 70,
        "backend_terms_exposed": 45,
        "assistant_role_drift": 45,
        "numeric_pattern_continuation": 35,
        "repeated_previous_reply": 35,
        "too_short_for_distress": 25,
        "pushes_after_privacy_boundary": 25,
        "generic_advice_under_distress": 20,
    }
    for issue in issues:
        score -= penalties.get(issue, 10)
    return max(score, 0)


def _compact(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _looks_mojibake(text: str) -> bool:
    artifact_codepoints = {0x00E5, 0x00E6, 0x00E7, 0x00E8, 0x00E9, 0x00EE, 0x00F0, 0x20AC, 0xFFFD}
    artifact_chars = sum(1 for char in text if ord(char) in artifact_codepoints)
    return artifact_chars >= 3


def _exposes_backend_terms(text: str) -> bool:
    compact = text.lower()
    terms = (
        "entropy",
        "loop_action",
        "care_phase",
        "backend",
        "phase=",
        "\u5fc3\u7406\u71b5",
        "\u8ba4\u77e5\u71b5",
        "\u98ce\u9669\u5206\u6570",
        "\u540e\u7aef",
        "\u5185\u90e8\u8bc4\u4f30",
    )
    return any(term in compact for term in terms)


def _assistant_claims_personal_experience(text: str) -> bool:
    compact = text.replace(" ", "")
    terms = (
        "\u6211\u4e5f\u6015\u6302\u79d1",
        "\u6211\u4e5f\u5f88\u6015\u6302\u79d1",
        "\u6211\u4e5f\u5f88\u96be\u53d7",
        "\u6211\u4e5f\u7761\u4e0d\u7740",
        "\u6211\u7684\u4f5c\u4e1a",
        "\u4f5c\u4e1a\u8fd8\u6ca1\u505a\u5b8c",
        "\u6211\u7684\u8003\u8bd5",
        "\u6211\u7684\u820d\u53cb",
    )
    return any(term in compact for term in terms)


def _too_short(text: str) -> bool:
    return len(text.replace(" ", "")) < 24


def _is_numeric_or_symbol_input(text: str) -> bool:
    return text in {"", "?", "??", "...", "1", "2", "3", "4", "5", "\uff1f"}


def _continues_numeric_pattern(text: str) -> bool:
    stripped = text.strip()
    return stripped in {"1", "2", "3", "4", "5", "6"} or stripped[:1] in {"1", "2", "3", "4", "5", "6"}


def _has_distress_context(text: str) -> bool:
    return any(
        term in text
        for term in (
            "\u96be\u53d7",
            "\u538b\u529b",
            "\u70e6",
            "\u5bb3\u6015",
            "\u7761\u4e0d\u7740",
            "\u6302\u79d1",
            "\u5bbf\u820d",
            "\u820d\u53cb",
            "\u5fc3\u60c5\u4e0d\u597d",
            "\u60f3\u54ed",
        )
    )


def _privacy_boundary(text: str) -> bool:
    return any(
        term in text
        for term in (
            "\u4e0d\u60f3\u8bf4",
            "\u4e0d\u60f3\u7ec6\u8bf4",
            "\u6015\u522b\u4eba\u77e5\u9053",
            "\u6015\u4f60\u544a\u8bc9\u522b\u4eba",
            "\u4fdd\u5bc6",
            "\u88ab\u77e5\u9053",
        )
    )


def _pushes_for_private_details(text: str) -> bool:
    return any(
        term in text
        for term in (
            "\u4e3a\u4ec0\u4e48",
            "\u5177\u4f53\u53d1\u751f\u4e86\u4ec0\u4e48",
            "\u8bf7\u4f60\u8be6\u7ec6\u8bf4",
            "\u5fc5\u987b\u8bf4",
        )
    )


def _generic_advice(text: str) -> bool:
    return any(
        term in text
        for term in (
            "\u4e0d\u8981\u60f3\u592a\u591a",
            "\u653e\u677e\u4e00\u4e0b",
            "\u522b\u62c5\u5fc3",
            "\u4f60\u8981\u52a0\u6cb9",
            "\u4e00\u5207\u90fd\u4f1a\u597d\u7684",
        )
    )


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item not in result:
            result.append(item)
    return result
