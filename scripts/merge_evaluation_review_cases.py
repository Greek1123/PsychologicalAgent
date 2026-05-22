from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REVIEW_COLUMNS = [
    "id",
    "response_id",
    "session_id",
    "input_text",
    "assistant_reply",
    "risk_level",
    "entropy_score",
    "local_policy",
    "mark_bad",
    "problem_tags",
    "chosen",
    "review_note",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def is_marked_bad(record: dict[str, Any]) -> bool:
    feedback = record.get("feedback") or {}
    failure = record.get("failure_review") or {}
    return bool(failure.get("rewrite_needed")) or int(feedback.get("helpful_score") or 0) < 0


def stable_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("scenario_id") or ""),
        str(record.get("turn_index") or ""),
        str(record.get("input_text") or "").strip(),
    )


def merge_cases(inputs: list[Path], output_jsonl: Path, output_csv: Path) -> dict[str, Any]:
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    source_counts: dict[str, int] = {}

    for path in inputs:
        count = 0
        for record in read_jsonl(path):
            if not is_marked_bad(record):
                continue
            key = stable_key(record)
            existing = merged.get(key)
            if existing is None:
                merged[key] = record
            else:
                existing_tags = set((existing.get("feedback") or {}).get("tags") or [])
                new_tags = set((record.get("feedback") or {}).get("tags") or [])
                if new_tags - existing_tags:
                    existing.setdefault("feedback", {})["tags"] = sorted(existing_tags | new_tags)
                    existing.setdefault("failure_review", {})["suspected_problem"] = sorted(existing_tags | new_tags)
                existing.setdefault("meta", {})["merged_duplicate_from"] = str(path)
            count += 1
        source_counts[str(path)] = count

    records = list(merged.values())
    write_jsonl(output_jsonl, records)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for record in records:
            feedback = record.get("feedback") or {}
            failure = record.get("failure_review") or {}
            risk = record.get("risk") or {}
            entropy = record.get("entropy") or {}
            local_policy = record.get("local_policy") or {}
            draft = record.get("sft_draft") or {}
            writer.writerow(
                {
                    "id": record.get("id", ""),
                    "response_id": record.get("response_id", ""),
                    "session_id": record.get("session_id", ""),
                    "input_text": record.get("input_text", ""),
                    "assistant_reply": record.get("assistant_reply", ""),
                    "risk_level": risk.get("level", ""),
                    "entropy_score": entropy.get("score", ""),
                    "local_policy": local_policy.get("policy_name", ""),
                    "mark_bad": "1",
                    "problem_tags": ",".join(feedback.get("tags") or failure.get("suspected_problem") or []),
                    "chosen": draft.get("chosen") or failure.get("preferred_reply") or "",
                    "review_note": failure.get("human_review_note") or feedback.get("user_note") or "",
                }
            )

    return {
        "inputs": [str(path) for path in inputs],
        "source_bad_counts": source_counts,
        "output_jsonl": str(output_jsonl),
        "output_csv": str(output_csv),
        "merged_bad_cases": len(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge bad cases from multiple model evaluation review outputs.")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sheet-out", required=True)
    args = parser.parse_args()
    stats = merge_cases(
        [Path(item) for item in args.input],
        Path(args.out),
        Path(args.sheet_out),
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
