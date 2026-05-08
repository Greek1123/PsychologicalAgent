from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LOW_VALUE_CHOSEN_PATTERNS = (
    "我在这里陪着你，不用急着组织语言",
    "我在这儿陪着你，不用着急",
    "不用急着把话说完整",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _messages_text(row: dict[str, Any]) -> str:
    return " ".join(str(message.get("content") or "") for message in row.get("messages") or [])


def _chosen_text(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    if not messages:
        return ""
    return str(messages[-1].get("content") or "")


def _looks_corrupted(text: str) -> bool:
    if "????????" in text:
        return True
    question_count = text.count("?")
    return question_count >= 8 and question_count > len(text) * 0.15


def _is_low_value_chosen(text: str) -> bool:
    return any(pattern in text for pattern in LOW_VALUE_CHOSEN_PATTERNS)


def filter_dpo_dataset(input_path: str, output_path: str) -> dict[str, Any]:
    source = Path(input_path)
    output = Path(output_path)
    kept: list[dict[str, Any]] = []
    removed_corrupted = 0
    removed_low_value = 0
    for row in _read_jsonl(source):
        messages_text = _messages_text(row)
        chosen = _chosen_text(row)
        if _looks_corrupted(messages_text):
            removed_corrupted += 1
            continue
        if _is_low_value_chosen(chosen):
            removed_low_value += 1
            continue
        kept.append(row)

    _write_jsonl(output, kept)
    return {
        "input": str(source),
        "output": str(output),
        "kept": len(kept),
        "removed_corrupted": removed_corrupted,
        "removed_low_value": removed_low_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter DPO rows with corrupted prompts or low-value chosen replies.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(filter_dpo_dataset(args.input, args.out), ensure_ascii=False))


if __name__ == "__main__":
    main()
