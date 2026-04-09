from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .logging_utils import get_logger


logger = get_logger("preference_template_builder")


def _unwrap_sample(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Support both raw style samples and triaged review records."""
    if isinstance(record.get("sample"), dict):
        sample = record["sample"]
        review_meta = {
            "bucket": record.get("bucket", ""),
            "quality_score": record.get("quality_score", ""),
            "quality_reasons": record.get("quality_reasons", []),
        }
        return sample, review_meta
    return record, {}


def _extract_prompt_and_chosen(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str] | None:
    """Use the latest assistant turn as the preferred response target."""
    last_assistant_index = -1
    for index, message in enumerate(messages):
        if message.get("role") == "assistant":
            last_assistant_index = index

    if last_assistant_index <= 0:
        return None

    prompt = messages[:last_assistant_index]
    chosen = str(messages[last_assistant_index].get("content", "")).strip()
    if not prompt or not chosen:
        return None

    return prompt, chosen


def build_preference_templates(input_path: str, output_path: str, *, limit: int | None = None) -> int:
    source = Path(input_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with source.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if limit is not None:
        records = records[:limit]

    written = 0
    skipped = 0
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            sample, review_meta = _unwrap_sample(record)
            messages = sample.get("messages", [])
            extracted = _extract_prompt_and_chosen(messages)
            if extracted is None:
                skipped += 1
                continue

            prompt, chosen = extracted
            review_notes = ""
            quality_reasons = review_meta.get("quality_reasons", [])
            if isinstance(quality_reasons, list) and quality_reasons:
                review_notes = "; ".join(str(item) for item in quality_reasons)

            template = {
                "id": sample.get("id"),
                "language": sample.get("language", ""),
                "task_type": "style_preference",
                "stage_goal": sample.get("stage_goal", ""),
                "prompt": prompt,
                "chosen": chosen,
                "rejected": "",
                "review_notes": review_notes,
                "meta": {
                    "source_bucket": review_meta.get("bucket", ""),
                    "quality_score": review_meta.get("quality_score", ""),
                    "source_task_type": sample.get("task_type", ""),
                },
            }
            handle.write(json.dumps(template, ensure_ascii=False) + "\n")
            written += 1

    logger.info(
        "Built %s style preference templates at %s (skipped=%s)",
        written,
        output,
        skipped,
    )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Build style preference templates for DPO/ORPO annotation.")
    parser.add_argument("--input", required=True, help="Input JSONL file.")
    parser.add_argument("--out", required=True, help="Output preference JSONL path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max sample count.")
    args = parser.parse_args()

    count = build_preference_templates(args.input, args.out, limit=args.limit)
    print(f"wrote {count} samples")


if __name__ == "__main__":
    main()
