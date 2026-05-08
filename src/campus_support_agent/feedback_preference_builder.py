from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .logging_utils import get_logger


logger = get_logger("feedback_preference_builder")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _extract_chosen(record: dict[str, Any]) -> str:
    draft = record.get("sft_draft") or {}
    review = record.get("failure_review") or {}
    return str(draft.get("chosen") or review.get("preferred_reply") or "").strip()


def _extract_rejected(record: dict[str, Any]) -> str:
    draft = record.get("sft_draft") or {}
    return str(draft.get("rejected") or record.get("assistant_reply") or "").strip()


def _extract_prompt(record: dict[str, Any]) -> list[dict[str, str]]:
    draft = record.get("sft_draft") or {}
    messages = draft.get("messages") or [
        *record.get("conversation_history", []),
        {"role": "user", "content": record.get("input_text", "")},
    ]
    prompt: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if role and content:
            prompt.append({"role": role, "content": content})
    return prompt


def _to_preference_record(record: dict[str, Any]) -> dict[str, Any] | None:
    prompt = _extract_prompt(record)
    chosen = _extract_chosen(record)
    rejected = _extract_rejected(record)
    if not prompt or not chosen or not rejected:
        return None

    return {
        "id": record.get("id") or record.get("response_id"),
        "task_type": "feedback_preference",
        "language": record.get("language", "zh"),
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "meta": {
            "session_id": record.get("session_id"),
            "response_id": record.get("response_id"),
            "feedback": record.get("feedback", {}),
            "risk": record.get("risk", {}),
            "entropy": record.get("entropy", {}),
            "failure_review": record.get("failure_review", {}),
        },
    }


def _to_ms_swift_dpo_record(preference: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            *preference["prompt"],
            {"role": "assistant", "content": preference["chosen"]},
        ],
        "rejected_response": preference["rejected"],
    }


def build_feedback_preference_dataset(
    input_path: str,
    preference_output_path: str,
    ms_swift_output_path: str | None = None,
) -> dict[str, Any]:
    source = Path(input_path)
    preference_output = Path(preference_output_path)
    ms_swift_output = Path(ms_swift_output_path) if ms_swift_output_path else None

    preference_records: list[dict[str, Any]] = []
    skipped_pending = 0
    skipped_invalid = 0
    for record in _read_jsonl(source):
        converted = _to_preference_record(record)
        if converted is None:
            if _extract_rejected(record) and not _extract_chosen(record):
                skipped_pending += 1
            else:
                skipped_invalid += 1
            continue
        preference_records.append(converted)

    _write_jsonl(preference_output, preference_records)
    ms_swift_records = [_to_ms_swift_dpo_record(record) for record in preference_records]
    if ms_swift_output:
        _write_jsonl(ms_swift_output, ms_swift_records)

    stats = {
        "input": str(source),
        "preference_output": str(preference_output),
        "ms_swift_output": str(ms_swift_output) if ms_swift_output else None,
        "ready_for_dpo": len(preference_records),
        "pending_chosen": skipped_pending,
        "skipped_invalid": skipped_invalid,
    }
    logger.info(
        "Built feedback preference dataset ready=%s pending=%s invalid=%s",
        stats["ready_for_dpo"],
        stats["pending_chosen"],
        stats["skipped_invalid"],
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert reviewed bad cases into DPO preference JSONL.")
    parser.add_argument("--input", required=True, help="Input bad_cases JSONL.")
    parser.add_argument("--out", required=True, help="Output preference JSONL.")
    parser.add_argument("--ms-swift-out", default=None, help="Optional ms-swift DPO JSONL output.")
    args = parser.parse_args()

    stats = build_feedback_preference_dataset(args.input, args.out, args.ms_swift_out)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
