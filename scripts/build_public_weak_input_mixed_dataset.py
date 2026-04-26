from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

BAD_PUBLIC_PATTERNS = (
    "后续对话内容略",
    "以上对话",
    "以下对话",
    "虚构示例",
    "实际咨询过程",
    "此对话仅为示例",
    "仅供参考",
    "祝你好运",
    "best of luck",
    "helpful conversation",
)

BAD_ASSISTANT_PATTERNS = (
    "我是个女生",
    "我是女生",
    "我是男生",
    "我是张伟",
    "我是小",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def _clean_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    cleaned = []
    for message in messages:
        role = message.get("role")
        content = (message.get("content") or "").strip()
        if role == "system":
            continue
        if role in {"user", "assistant"} and content:
            cleaned.append({"role": role, "content": content})
    return cleaned


def _is_usable_dialog(record: dict[str, Any]) -> bool:
    messages = _clean_messages(record.get("messages", []))
    if len(messages) < 2:
        return False
    roles = {message["role"] for message in messages}
    if not {"user", "assistant"}.issubset(roles):
        return False
    joined = "\n".join(message["content"] for message in messages).lower()
    if any(pattern.lower() in joined for pattern in BAD_PUBLIC_PATTERNS):
        return False
    if any(
        message["role"] == "assistant"
        and any(pattern in message["content"] for pattern in BAD_ASSISTANT_PATTERNS)
        for message in messages
    ):
        return False
    if any(
        message["role"] == "assistant" and len(message["content"]) > 900
        for message in messages
    ):
        return False
    total_chars = sum(len(message["content"]) for message in messages)
    return 20 <= total_chars <= 2600


def build_mixed_dataset(
    weak_path: Path,
    public_path: Path,
    output_path: Path,
    public_limit: int,
    weak_repeat: int,
    seed: int,
) -> dict[str, int]:
    rng = random.Random(seed)
    weak_records = _load_jsonl(weak_path)
    public_records = [record for record in _load_jsonl(public_path) if _is_usable_dialog(record)]
    rng.shuffle(public_records)

    mixed: list[dict[str, Any]] = []
    for repeat_index in range(weak_repeat):
        for record in weak_records:
            mixed.append(
                {
                    "messages": _clean_messages(record["messages"]),
                    "meta": {
                        "source": "weak_input_repair",
                        "repeat": repeat_index,
                        "id": record.get("id"),
                    },
                }
            )

    for index, record in enumerate(public_records[:public_limit]):
        mixed.append(
            {
                "messages": _clean_messages(record["messages"]),
                "meta": {
                    "source": record.get("meta", {}).get("source", "public_phase0"),
                    "mix_type": "public_anchor",
                    "index": index,
                },
            }
        )

    rng.shuffle(mixed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in mixed:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "written": len(mixed),
        "weak_base": len(weak_records),
        "weak_written": len(weak_records) * weak_repeat,
        "public_candidates": len(public_records),
        "public_written": min(public_limit, len(public_records)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build mixed public + weak-input SFT data.")
    parser.add_argument(
        "--weak",
        default=str(ROOT / "data" / "training" / "weak_input" / "weak_input_phase0_5_train_ms_swift.jsonl"),
    )
    parser.add_argument(
        "--public",
        default=str(ROOT / "data" / "training" / "public_sft" / "public_phase0_chat_train_ms_swift.jsonl"),
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "data" / "training" / "weak_input" / "public_weak_input_mixed_train_ms_swift.jsonl"),
    )
    parser.add_argument("--public-limit", type=int, default=500)
    parser.add_argument("--weak-repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stats = build_mixed_dataset(
        weak_path=Path(args.weak),
        public_path=Path(args.public),
        output_path=Path(args.out),
        public_limit=args.public_limit,
        weak_repeat=args.weak_repeat,
        seed=args.seed,
    )
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
