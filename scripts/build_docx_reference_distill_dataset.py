"""Build SFT samples from low-scoring DOCX reference evaluation turns."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def turn_score(turn: dict[str, Any]) -> int | float | None:
    comparison = turn.get("comparison") or {}
    score = comparison.get("score")
    return score if isinstance(score, (int, float)) else None


def build_messages(turns: list[dict[str, Any]], turn_index: int, max_history_turns: int) -> list[dict[str, str]]:
    start = max(0, turn_index - max_history_turns)
    messages: list[dict[str, str]] = []
    for previous in turns[start:turn_index]:
        user = str(previous.get("user") or "").strip()
        assistant = str(previous.get("reference_reply") or "").strip()
        if user:
            messages.append({"role": "user", "content": user})
        if assistant:
            messages.append({"role": "assistant", "content": assistant})

    current = turns[turn_index]
    user = str(current.get("user") or "").strip()
    assistant = str(current.get("reference_reply") or "").strip()
    if user:
        messages.append({"role": "user", "content": user})
    if assistant:
        messages.append({"role": "assistant", "content": assistant})
    return messages


def build_records(
    cases: list[dict[str, Any]],
    *,
    max_score: float,
    max_history_turns: int,
    repeat: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    preference_records: list[dict[str, Any]] = []
    sft_records: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()

    for case in cases:
        turns = list(case.get("turns") or [])
        for index, turn in enumerate(turns):
            score = turn_score(turn)
            if score is None or score > max_score:
                continue
            messages = build_messages(turns, index, max_history_turns)
            if len(messages) < 2 or messages[-1]["role"] != "assistant":
                continue
            key = tuple(f"{item['role']}:{item['content']}" for item in messages)
            if key in seen:
                continue
            seen.add(key)

            record_id = f"docx_reference_distill_{case.get('case_id', 'unknown')}_{turn.get('turn_index', index + 1)}"
            meta = {
                "source": "docx_reference_eval_low_score_distill",
                "case_id": case.get("case_id"),
                "title": case.get("title"),
                "turn_index": turn.get("turn_index"),
                "score": score,
                "max_score": max_score,
            }
            preference_records.append(
                {
                    "id": record_id,
                    "task_type": "docx_reference_distill",
                    "language": "zh",
                    "category": "docx_low_score_reference",
                    "prompt": messages[:-1],
                    "chosen": messages[-1]["content"],
                    "rejected": str(turn.get("model_reply") or ""),
                    "meta": meta,
                }
            )
            for repeat_index in range(max(1, repeat)):
                repeated_meta = dict(meta)
                repeated_meta["repeat_index"] = repeat_index
                sft_records.append({"messages": messages, "meta": repeated_meta})

    return preference_records, sft_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DOCX reference distillation data from low-scoring eval turns.")
    parser.add_argument("--input", required=True, help="Evaluation JSONL produced by evaluate_backend_docx_reference_cases.py.")
    parser.add_argument("--out", required=True, help="Preference-style output JSONL.")
    parser.add_argument("--ms-swift-out", required=True, help="ms-swift messages-only SFT output JSONL.")
    parser.add_argument("--max-score", type=float, default=64.0)
    parser.add_argument("--max-history-turns", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--base-ms-swift", help="Optional existing ms-swift JSONL to prepend into a combined dataset.")
    parser.add_argument("--combined-ms-swift-out", help="Optional combined ms-swift JSONL output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = read_jsonl(Path(args.input))
    preference_records, sft_records = build_records(
        cases,
        max_score=args.max_score,
        max_history_turns=args.max_history_turns,
        repeat=args.repeat,
    )

    write_jsonl(Path(args.out), preference_records)
    write_jsonl(Path(args.ms_swift_out), sft_records)

    combined_records: list[dict[str, Any]] = []
    if args.base_ms_swift and args.combined_ms_swift_out:
        base_records = read_jsonl(Path(args.base_ms_swift))
        combined_records = [*base_records, *sft_records]
        write_jsonl(Path(args.combined_ms_swift_out), combined_records)

    title_counts = Counter(str(record["meta"].get("title")) for record in preference_records)
    payload = {
        "input": args.input,
        "out": args.out,
        "ms_swift_out": args.ms_swift_out,
        "records": len(preference_records),
        "sft_records": len(sft_records),
        "combined_ms_swift_out": args.combined_ms_swift_out or "",
        "combined_records": len(combined_records),
        "top_titles": dict(title_counts.most_common(12)),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
