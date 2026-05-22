from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .reply_quality import build_reply_quality_timeline


def build_reply_quality_bad_cases(
    records: list[dict[str, Any]],
    *,
    min_quality_score: int = 80,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    timeline = build_reply_quality_timeline(records)
    timeline_by_response_id = {
        str(item.get("response_id") or ""): item
        for item in timeline
        if item.get("response_id")
    }

    bad_cases: list[dict[str, Any]] = []
    for record in records:
        response_id = str(record.get("response_id") or "")
        quality_item = timeline_by_response_id.get(response_id)
        if not quality_item:
            continue
        quality_score = int(quality_item.get("quality_score") or 0)
        issues = [str(issue) for issue in quality_item.get("issues") or []]
        if quality_score > min_quality_score and not issues:
            continue

        bad_cases.append(_build_bad_case(record, quality_item))
        if limit is not None and len(bad_cases) >= limit:
            break
    return bad_cases


def export_reply_quality_bad_cases(
    records: list[dict[str, Any]],
    output_path: str | Path,
    *,
    min_quality_score: int = 80,
    limit: int | None = None,
) -> dict[str, Any]:
    bad_cases = build_reply_quality_bad_cases(
        records,
        min_quality_score=min_quality_score,
        limit=limit,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for case in bad_cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")

    issue_counts: dict[str, int] = {}
    for case in bad_cases:
        for issue in case.get("quality_review", {}).get("issues", []):
            issue_counts[str(issue)] = issue_counts.get(str(issue), 0) + 1

    return {
        "output": str(output),
        "records_seen": len(records),
        "exported": len(bad_cases),
        "min_quality_score": min_quality_score,
        "issue_counts": dict(sorted(issue_counts.items(), key=lambda item: (-item[1], item[0]))),
    }


def _build_bad_case(record: dict[str, Any], quality_item: dict[str, Any]) -> dict[str, Any]:
    response = record.get("response") or {}
    reply_text = response.get("reply_text") or record.get("reply_text") or ""
    input_text = record.get("input_text") or ""
    response_id = record.get("response_id") or ""
    issues = [str(issue) for issue in quality_item.get("issues") or []]
    review_note = _build_review_note(issues)
    return {
        "id": f"reply_quality_bad_case_{response_id}",
        "response_id": response_id,
        "session_id": record.get("session_id"),
        "source": record.get("source"),
        "language": _detect_language(str(input_text)),
        "created_at": record.get("created_at"),
        "input_text": input_text,
        "conversation_history": record.get("conversation_history", []),
        "assistant_reply": reply_text,
        "risk": response.get("risk", {}),
        "entropy": response.get("entropy", {}),
        "local_policy": response.get("local_policy", {}),
        "referral_decision": response.get("referral_decision", {}),
        "quality_review": {
            "quality_score": quality_item.get("quality_score"),
            "issues": issues,
            "needs_review": quality_item.get("needs_review"),
            "review_note": review_note,
        },
        "feedback": {
            "helpful_score": -1,
            "tags": issues,
            "user_note": review_note,
            "source": "reply_quality_monitor",
        },
        "failure_review": {
            "suspected_problem": issues,
            "human_review_note": review_note,
            "rewrite_needed": True,
            "preferred_reply": "",
            "review_status": "needs_rewrite",
        },
        "sft_draft": {
            "messages": [
                *record.get("conversation_history", []),
                {"role": "user", "content": input_text},
            ],
            "rejected": reply_text,
            "chosen": "",
        },
    }


def _build_review_note(issues: list[str]) -> str:
    if not issues:
        return "Reply quality monitor flagged this sample for manual review."
    readable = ", ".join(issues)
    return f"Reply quality monitor flagged: {readable}."


def _detect_language(text: str) -> str:
    if any("\u4e00" <= char <= "\u9fff" for char in text):
        return "zh"
    if any(("a" <= char.lower() <= "z") for char in text):
        return "en"
    return "unknown"
