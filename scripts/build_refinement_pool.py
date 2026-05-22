from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


FORBIDDEN_ASSISTANT_TERMS = (
    "心理熵",
    "认知熵",
    "减熵重点",
    "风险等级",
    "状态识别",
    "预警信号",
    "高风险信号",
    "风险信号",
    "内部评估",
    "内部判断",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_sft_record(record: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(record.get("messages"), list):
        messages = record["messages"]
        if len(messages) >= 2:
            assistant_text = "\n".join(
                str(message.get("content") or "")
                for message in messages
                if isinstance(message, dict) and message.get("role") == "assistant"
            )
            if _has_forbidden_assistant_terms(assistant_text):
                return None
            return {"messages": messages}

    draft = record.get("sft_draft") or {}
    chosen = str(draft.get("chosen") or (record.get("failure_review") or {}).get("preferred_reply") or "").strip()
    messages = draft.get("messages") or record.get("prompt") or []
    if not chosen or not isinstance(messages, list):
        return None
    clean_messages: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "").strip()
        if role and content:
            clean_messages.append({"role": role, "content": content})
    if not clean_messages:
        return None
    if _has_forbidden_assistant_terms(chosen):
        return None
    return {"messages": [*clean_messages, {"role": "assistant", "content": chosen}]}


def _has_forbidden_assistant_terms(text: str) -> bool:
    return any(term in text for term in FORBIDDEN_ASSISTANT_TERMS)


def category_of(record: dict[str, Any]) -> str:
    if record.get("category"):
        return str(record["category"])
    meta = record.get("meta") or {}
    if meta.get("category"):
        return str(meta["category"])
    if record.get("scenario_category"):
        return str(record["scenario_category"])
    feedback = record.get("feedback") or {}
    tags = feedback.get("tags") or []
    if tags:
        return str(tags[0])
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a combined refinement SFT pool.")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    categories: Counter[str] = Counter()
    skipped = 0

    for input_path in [Path(item) for item in args.input]:
        for source_record in read_jsonl(input_path):
            converted = extract_sft_record(source_record)
            if converted is None:
                skipped += 1
                continue
            key = json.dumps(converted["messages"], ensure_ascii=False, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            converted["meta"] = {
                "source": str(input_path),
                "category": category_of(source_record),
            }
            categories[converted["meta"]["category"]] += 1
            records.append(converted)

    write_jsonl(Path(args.out), records)
    summary = {
        "inputs": args.input,
        "output": args.out,
        "records": len(records),
        "skipped": skipped,
        "category_counts": dict(categories),
    }
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
