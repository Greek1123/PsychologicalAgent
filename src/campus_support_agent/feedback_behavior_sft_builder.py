from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .logging_utils import get_logger


logger = get_logger("feedback_behavior_sft_builder")


FORBIDDEN_CHOSEN_PHRASES = (
    "当前减熵重点",
    "认知熵",
    "心理熵",
    "你好，感谢你前来咨询",
    "感谢你前来咨询",
    "抱抱你",
    "宝",
    "绝对保密",
    "不会告诉任何人",
)

NOISY_USER_PHRASES = (
    "傻逼",
    "弱智",
    "垃圾",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _text(value: Any) -> str:
    return str(value or "").strip()


def _extract_chosen(record: dict[str, Any]) -> str:
    draft = record.get("sft_draft") or {}
    review = record.get("failure_review") or {}
    return _text(record.get("chosen") or draft.get("chosen") or review.get("preferred_reply"))


def _extract_prompt_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    if isinstance(record.get("prompt"), list):
        source = record["prompt"]
    else:
        draft = record.get("sft_draft") or {}
        source = draft.get("messages") or [
            *record.get("conversation_history", []),
            {"role": "user", "content": record.get("input_text", "")},
        ]

    messages: list[dict[str, str]] = []
    for message in source:
        role = _text(message.get("role"))
        content = _text(message.get("content"))
        if role and content:
            messages.append({"role": role, "content": content})
    return messages


def _recent_user_context(messages: list[dict[str, str]], *, max_user_turns: int = 3) -> str:
    user_turns = [message["content"] for message in messages if message.get("role") == "user"]
    user_turns = [turn for turn in user_turns if turn]
    cleaned_turns: list[str] = []
    final_turn = user_turns[-1] if user_turns else ""
    for turn in user_turns:
        if any(phrase in turn for phrase in NOISY_USER_PHRASES):
            continue
        if turn != final_turn and turn.strip() in {"?", "？", "??", "？？", "嗯", "哦", "啊"}:
            continue
        if cleaned_turns and cleaned_turns[-1] == turn:
            continue
        cleaned_turns.append(turn)
    user_turns = cleaned_turns
    if not user_turns:
        return ""

    recent = user_turns[-max_user_turns:]
    if len(recent) == 1:
        return recent[0]

    lines = ["请根据用户最近几句话自然回应，不要重复专业分析，也不要追问太多。"]
    for index, turn in enumerate(recent[:-1], start=1):
        lines.append(f"前文{index}：{turn}")
    lines.append(f"用户刚刚说：{recent[-1]}")
    return "\n".join(lines)


def _looks_low_quality_reply(text: str) -> bool:
    if len(text) < 12:
        return True
    return any(phrase in text for phrase in FORBIDDEN_CHOSEN_PHRASES)


def _to_sft_record(record: dict[str, Any]) -> dict[str, Any] | None:
    chosen = _extract_chosen(record)
    if not chosen or _looks_low_quality_reply(chosen):
        return None

    prompt_messages = _extract_prompt_messages(record)
    user_context = _recent_user_context(prompt_messages)
    if not user_context:
        return None

    return {
        "messages": [
            {"role": "user", "content": user_context},
            {"role": "assistant", "content": chosen},
        ]
    }


def build_feedback_behavior_sft_dataset(input_path: str, output_path: str) -> dict[str, Any]:
    source = Path(input_path)
    output = Path(output_path)

    records: list[dict[str, Any]] = []
    skipped_invalid = 0
    skipped_low_quality = 0
    seen: set[str] = set()

    for source_record in _read_jsonl(source):
        converted = _to_sft_record(source_record)
        if converted is None:
            chosen = _extract_chosen(source_record)
            if chosen and _looks_low_quality_reply(chosen):
                skipped_low_quality += 1
            else:
                skipped_invalid += 1
            continue

        dedupe_key = json.dumps(converted["messages"], ensure_ascii=False, sort_keys=True)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        records.append(converted)

    _write_jsonl(output, records)

    stats = {
        "input": str(source),
        "output": str(output),
        "written": len(records),
        "skipped_invalid": skipped_invalid,
        "skipped_low_quality": skipped_low_quality,
    }
    logger.info("Built feedback behavior SFT dataset: %s", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert reviewed feedback preferences into behavior SFT rows.")
    parser.add_argument("--input", required=True, help="Reviewed preference JSONL or reviewed bad-case JSONL.")
    parser.add_argument("--out", required=True, help="Output ms-swift SFT JSONL.")
    args = parser.parse_args()

    print(json.dumps(build_feedback_behavior_sft_dataset(args.input, args.out), ensure_ascii=False))


if __name__ == "__main__":
    main()
