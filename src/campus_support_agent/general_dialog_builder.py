from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from .logging_utils import get_logger


logger = get_logger("general_dialog_builder")
_SPEAKER_PATTERN = re.compile(r"(Human|Assistant):")


def _alternate_messages(turns: list[str], *, max_turns: int) -> list[dict[str, str]]:
    """Convert alternating raw utterances into user/assistant messages."""
    cleaned = [str(turn).strip() for turn in turns if str(turn).strip()]
    if len(cleaned) < 2:
        return []

    clipped = cleaned[:max_turns]
    # Keep an even number of turns so each user utterance has a paired reply.
    if len(clipped) % 2 == 1:
        clipped = clipped[:-1]

    messages: list[dict[str, str]] = []
    for index, content in enumerate(clipped):
        role = "user" if index % 2 == 0 else "assistant"
        messages.append({"role": role, "content": content})
    return messages


def _parse_instruction_transcript(text: str) -> list[dict[str, str]]:
    """Parse `Human:` / `Assistant:` transcripts into message records."""
    transcript = str(text or "").strip()
    if not transcript:
        return []

    matches = list(_SPEAKER_PATTERN.finditer(transcript))
    if not matches:
        return []

    messages: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        role = "user" if match.group(1) == "Human" else "assistant"
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(transcript)
        content = transcript[start:end].strip()
        if not content:
            continue
        messages.append({"role": role, "content": content})
    return messages


def _clip_messages(messages: list[dict[str, str]], *, max_turns: int) -> list[dict[str, str]]:
    clipped = messages[:max_turns]
    if len(clipped) % 2 == 1:
        clipped = clipped[:-1]
    return clipped


def _messages_from_instruction_record(record: dict[str, Any], *, max_turns: int) -> list[dict[str, str]]:
    messages = _parse_instruction_transcript(record.get("instruction", ""))

    raw_input = str(record.get("input", "") or "").strip()
    if raw_input:
        messages.append({"role": "user", "content": raw_input})

    raw_output = str(record.get("output", "") or "").strip()
    if raw_output:
        messages.append({"role": "assistant", "content": raw_output})

    if not messages or messages[0]["role"] != "user":
        return []

    return _clip_messages(messages, max_turns=max_turns)


def _load_records(source: Path) -> list[dict[str, Any]]:
    text = source.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError("Expected a JSON list when the source file starts with '['.")
        return [record for record in payload if isinstance(record, dict)]

    decoder = json.JSONDecoder()
    records: list[dict[str, Any]] = []
    index = 0
    text_length = len(text)
    while index < text_length:
        while index < text_length and text[index].isspace():
            index += 1
        if index >= text_length:
            break
        record, next_index = decoder.raw_decode(text, index)
        if isinstance(record, dict):
            records.append(record)
        index = next_index
    return records


def build_general_multiturn_dataset(
    input_path: str,
    output_path: str,
    *,
    min_turns: int = 10,
    max_turns: int = 20,
    limit: int | None = 6000,
    seed: int = 42,
) -> dict[str, int]:
    source = Path(input_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    records = _load_records(source)
    random.Random(seed).shuffle(records)

    written = 0
    skipped_short = 0
    skipped_empty = 0

    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            messages: list[dict[str, str]] = []
            source_name = "unknown"

            if {"instruction", "output"} <= set(record.keys()):
                messages = _messages_from_instruction_record(record, max_turns=max_turns)
                source_name = "instruction_multiturn"
            else:
                turns = record.get("content", []) if isinstance(record, dict) else []
                if isinstance(turns, list):
                    messages = _alternate_messages(turns, max_turns=max_turns)
                    source_name = "dialog_release"

            if not messages:
                skipped_empty += 1
                continue
            if len(messages) < min_turns:
                skipped_short += 1
                continue

            sample = {
                "id": f"dialog-release-{record.get('dialog_id', written)}",
                "language": "zh",
                "task_type": "general_multiturn_dialogue",
                "stage_goal": "contextual_continuation",
                "messages": messages,
                "meta": {
                    "source": source_name,
                    "dialog_id": str(record.get("dialog_id", "") or ""),
                    "document_id": str(record.get("document_id", "") or ""),
                },
            }
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1

            if limit is not None and written >= limit:
                break

    stats = {
        "written": written,
        "skipped_short": skipped_short,
        "skipped_empty": skipped_empty,
    }
    logger.info(
        "Built general multi-turn warmup dataset at %s with written=%s skipped_short=%s skipped_empty=%s",
        output,
        written,
        skipped_short,
        skipped_empty,
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a general multi-turn warmup dataset from dialog_release.json.")
    parser.add_argument("--input", required=True, help="Path to dialog_release.json.")
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    parser.add_argument("--min-turns", type=int, default=10, help="Minimum kept message count.")
    parser.add_argument("--max-turns", type=int, default=20, help="Maximum kept message count.")
    parser.add_argument("--limit", type=int, default=6000, help="Optional sample cap.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed before sampling.")
    args = parser.parse_args()

    stats = build_general_multiturn_dataset(
        args.input,
        args.out,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        limit=args.limit,
        seed=args.seed,
    )
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
