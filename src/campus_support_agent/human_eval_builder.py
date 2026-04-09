from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .logging_utils import get_logger


logger = get_logger("human_eval_builder")


EVAL_COLUMNS = [
    "id",
    "language",
    "task_type",
    "stage_goal",
    "user_turns",
    "assistant_turns",
    "naturalness",
    "empathy",
    "followup_quality",
    "summary_accuracy",
    "safety",
    "notes",
]


def _unwrap_sample(record: dict[str, Any]) -> dict[str, Any]:
    """Support direct style samples and triaged wrapper records."""
    if isinstance(record.get("sample"), dict):
        return record["sample"]
    return record


def _serialize_turns(messages: list[dict[str, str]], role: str) -> str:
    lines = []
    for item in messages:
        if item.get("role") == role:
            lines.append(item.get("content", ""))
    return "\n\n".join(lines)


def build_human_eval_sheet(input_path: str, output_csv: str, *, limit: int | None = None) -> int:
    source = Path(input_path)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)

    with source.open("r", encoding="utf-8") as handle:
        samples = [_unwrap_sample(json.loads(line)) for line in handle if line.strip()]
    if limit is not None:
        samples = samples[:limit]

    with output.open("w", encoding="utf-8-sig", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=EVAL_COLUMNS)
        writer.writeheader()

        for sample in samples:
            messages = sample.get("messages", [])
            writer.writerow(
                {
                    "id": sample.get("id", ""),
                    "language": sample.get("language", ""),
                    "task_type": sample.get("task_type", ""),
                    "stage_goal": sample.get("stage_goal", ""),
                    "user_turns": _serialize_turns(messages, "user"),
                    "assistant_turns": _serialize_turns(messages, "assistant"),
                    "naturalness": "",
                    "empathy": "",
                    "followup_quality": "",
                    "summary_accuracy": "",
                    "safety": "",
                    "notes": "",
                }
            )

    logger.info("Built human evaluation sheet with %s rows at %s", len(samples), output)
    return len(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a human counselor evaluation CSV sheet.")
    parser.add_argument("--input", required=True, help="Input JSONL dataset.")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit.")
    args = parser.parse_args()

    count = build_human_eval_sheet(args.input, args.out, limit=args.limit)
    print(f"wrote {count} rows")


if __name__ == "__main__":
    main()
