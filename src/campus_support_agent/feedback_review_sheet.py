from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .logging_utils import get_logger


logger = get_logger("feedback_review_sheet")

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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _is_marked_bad(value: str | None) -> bool:
    return str(value or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "bad",
        "是",
        "坏",
        "差",
        "需要修改",
    }


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "bad", "是", "差", "需要改"}


def build_feedback_review_sheet(input_path: str, output_csv_path: str, *, limit: int | None = None) -> int:
    source = Path(input_path)
    output = Path(output_csv_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    records = _read_jsonl(source)
    if limit is not None:
        records = records[:limit]

    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for record in records:
            risk = record.get("risk") or {}
            entropy = record.get("entropy") or {}
            local_policy = record.get("local_policy") or {}
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
                    "mark_bad": "",
                    "problem_tags": "",
                    "chosen": "",
                    "review_note": "",
                }
            )

    logger.info("Built feedback review sheet at %s with %s rows", output, len(records))
    return len(records)


def apply_feedback_review_sheet(input_path: str, review_csv_path: str, output_path: str) -> dict[str, Any]:
    source = Path(input_path)
    review_csv = Path(review_csv_path)
    output = Path(output_path)

    records = _read_jsonl(source)
    with review_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    annotations: dict[str, dict[str, str]] = {}
    for row in rows:
        record_id = str(row.get("id") or "").strip()
        response_id = str(row.get("response_id") or "").strip()
        if record_id:
            annotations[record_id] = row
        if response_id:
            annotations[response_id] = row

    updated = 0
    marked_bad = 0
    for record in records:
        annotation = annotations.get(str(record.get("id") or "")) or annotations.get(str(record.get("response_id") or ""))
        if not annotation:
            continue

        chosen = str(annotation.get("chosen") or "").strip()
        problem_tags = [
            item.strip()
            for item in str(annotation.get("problem_tags") or "").replace("，", ",").split(",")
            if item.strip()
        ]
        review_note = str(annotation.get("review_note") or "").strip()
        is_bad = _is_marked_bad(annotation.get("mark_bad")) or bool(chosen)
        if is_bad:
            marked_bad += 1

        failure_review = record.setdefault("failure_review", {})
        feedback = record.setdefault("feedback", {})
        draft = record.setdefault("sft_draft", {})

        if problem_tags:
            failure_review["suspected_problem"] = problem_tags
            feedback["tags"] = problem_tags
        if review_note:
            failure_review["human_review_note"] = review_note
            feedback["user_note"] = review_note
        if is_bad:
            failure_review["rewrite_needed"] = True
            failure_review["review_status"] = "reviewed" if chosen else "needs_rewrite"
        if chosen:
            failure_review["preferred_reply"] = chosen
            draft["chosen"] = chosen
            updated += 1

    _write_jsonl(output, records)
    stats = {
        "input": str(source),
        "review_csv": str(review_csv),
        "output": str(output),
        "total_records": len(records),
        "marked_bad": marked_bad,
        "updated_with_chosen": updated,
    }
    logger.info(
        "Applied feedback review sheet output=%s total=%s marked_bad=%s chosen=%s",
        output,
        stats["total_records"],
        stats["marked_bad"],
        stats["updated_with_chosen"],
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/apply CSV review sheets for feedback bad cases.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build CSV sheet from review_cases JSONL.")
    build_parser.add_argument("--input", required=True, help="Input review_cases JSONL.")
    build_parser.add_argument("--out", required=True, help="Output CSV path.")
    build_parser.add_argument("--limit", type=int, default=None, help="Optional row limit.")

    apply_parser = subparsers.add_parser("apply", help="Apply edited CSV sheet back to JSONL.")
    apply_parser.add_argument("--input", required=True, help="Original review_cases JSONL.")
    apply_parser.add_argument("--sheet", required=True, help="Edited CSV sheet.")
    apply_parser.add_argument("--out", required=True, help="Output reviewed JSONL.")

    args = parser.parse_args()
    if args.command == "build":
        count = build_feedback_review_sheet(args.input, args.out, limit=args.limit)
        print(json.dumps({"written": count, "out": args.out}, ensure_ascii=False))
    else:
        stats = apply_feedback_review_sheet(args.input, args.sheet, args.out)
        print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
