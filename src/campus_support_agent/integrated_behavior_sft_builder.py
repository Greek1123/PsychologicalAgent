from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .logging_utils import get_logger


logger = get_logger("integrated_behavior_sft_builder")


DEFAULT_SOURCE_SPECS = (
    ("public_chat", "data/training/public_sft/public_phase0_chat_train_ms_swift.jsonl", 1200),
    ("public_safety", "data/training/public_sft/public_safety_train_ms_swift.jsonl", 200),
    ("weak_input", "data/training/weak_input/public_weak_input_mixed_train_ms_swift.jsonl", 300),
    ("feedback_behavior", "data/training/feedback_bad_cases/feedback_behavior_sft_ms_swift.jsonl", 0),
    ("noisy_distress", "data/training/noisy_distress/noisy_distress_sft_ms_swift.jsonl", 0),
    ("role_boundary", "data/training/role_boundary/role_boundary_sft_ms_swift.jsonl", 0),
)

LOW_VALUE_REPLY_MARKERS = (
    "你好，感谢你前来咨询",
    "感谢你前来咨询",
    "我是张伟",
    "我是小智，很高兴认识你",
    "当前减熵重点",
    "认知熵",
    "心理熵",
    "我也很怕",
    "我也很焦虑",
    "我也很难受",
    "我的作业",
    "我现在感觉好累",
    "我好想考",
    "我也有点这样的困扰",
)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: Path
    limit: int
    min_cjk_ratio: float | None = None


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def _text(value: Any) -> str:
    return str(value or "").strip()


def _message_text(record: dict[str, Any]) -> str:
    return "\n".join(_text(message.get("content")) for message in record.get("messages", []))


def _cjk_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha() or "\u4e00" <= char <= "\u9fff"]
    if not letters:
        return 0.0
    cjk = [char for char in letters if "\u4e00" <= char <= "\u9fff"]
    return len(cjk) / len(letters)


def _assistant_texts(record: dict[str, Any]) -> list[str]:
    return [
        _text(message.get("content"))
        for message in record.get("messages", [])
        if message.get("role") == "assistant"
    ]


def _looks_corrupted(text: str) -> bool:
    if "\ufffd" in text:
        return True
    compact = "".join(text.split())
    if not compact:
        return True
    question_marks = compact.count("?") + compact.count("？")
    if question_marks >= 5 and question_marks / max(len(compact), 1) > 0.35:
        return True
    return False


def _is_valid_sft_record(record: dict[str, Any], *, min_cjk_ratio: float, max_chars: int) -> bool:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return False

    roles = [_text(message.get("role")) for message in messages if isinstance(message, dict)]
    if "user" not in roles or "assistant" not in roles:
        return False
    first_dialog_role = next((role for role in roles if role != "system"), "")
    if first_dialog_role != "user":
        return False

    text = _message_text(record)
    if not text or len(text) > max_chars:
        return False
    if _looks_corrupted(text):
        return False
    for message in messages:
        if isinstance(message, dict) and _looks_corrupted(_text(message.get("content"))):
            return False
    if _cjk_ratio(text) < min_cjk_ratio:
        return False

    assistant_text = "\n".join(_assistant_texts(record))
    if len(assistant_text) < 12:
        return False
    return not any(marker in assistant_text for marker in LOW_VALUE_REPLY_MARKERS)


def _dedupe_key(record: dict[str, Any]) -> str:
    messages = [
        {"role": _text(message.get("role")), "content": _text(message.get("content"))}
        for message in record.get("messages", [])
        if isinstance(message, dict)
    ]
    return json.dumps(messages, ensure_ascii=False, sort_keys=True)


def _sample_records(
    path: Path,
    *,
    limit: int,
    rng: random.Random,
    min_cjk_ratio: float,
    max_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if limit < 0:
        return [], {
            "available": 0,
            "skipped_invalid": 0,
            "skipped_quality": 0,
        }

    valid: list[dict[str, Any]] = []
    skipped_invalid = 0
    skipped_quality = 0

    for record in _read_jsonl(path):
        if not _is_valid_sft_record(record, min_cjk_ratio=min_cjk_ratio, max_chars=max_chars):
            skipped_quality += 1
            continue
        valid.append(record)

    if limit > 0 and len(valid) > limit:
        valid = rng.sample(valid, limit)

    return valid, {
        "available": len(valid),
        "skipped_invalid": skipped_invalid,
        "skipped_quality": skipped_quality,
    }


def _resolve_specs(root: Path, args: argparse.Namespace) -> list[SourceSpec]:
    public_external_limit = -1 if args.public_external_limit == 0 else args.public_external_limit
    raw_specs = [
        ("public_chat", args.public_chat, args.public_chat_limit, None),
        ("public_safety", args.public_safety, args.public_safety_limit, None),
        ("weak_input", args.weak_input, args.weak_input_limit, None),
        ("feedback_behavior", args.feedback_behavior, args.feedback_behavior_limit, None),
        ("noisy_distress", args.noisy_distress, args.noisy_distress_limit, None),
        ("role_boundary", args.role_boundary, args.role_boundary_limit, None),
        ("public_external", args.public_external, public_external_limit, 0.0),
    ]
    specs: list[SourceSpec] = []
    for name, raw_path, limit, source_min_cjk_ratio in raw_specs:
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        specs.append(SourceSpec(name=name, path=path, limit=limit, min_cjk_ratio=source_min_cjk_ratio))
    return specs


def build_integrated_behavior_sft_dataset(
    *,
    out: str,
    seed: int = 42,
    min_cjk_ratio: float = 0.45,
    max_chars: int = 2600,
    specs: list[SourceSpec],
) -> dict[str, Any]:
    rng = random.Random(seed)
    output_path = Path(out)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_stats: dict[str, dict[str, int | str]] = {}

    for spec in specs:
        if not spec.path.exists():
            source_stats[spec.name] = {"path": str(spec.path), "written": 0, "missing": 1}
            continue

        sampled, stats = _sample_records(
            spec.path,
            limit=spec.limit,
            rng=rng,
            min_cjk_ratio=min_cjk_ratio if spec.min_cjk_ratio is None else spec.min_cjk_ratio,
            max_chars=max_chars,
        )

        written = 0
        for record in sampled:
            key = _dedupe_key(record)
            if key in seen:
                continue
            seen.add(key)
            records.append(record)
            written += 1

        source_stats[spec.name] = {
            "path": str(spec.path),
            "written": written,
            **stats,
        }

    rng.shuffle(records)
    written_total = _write_jsonl(output_path, records)

    result = {
        "output": str(output_path),
        "written": written_total,
        "seed": seed,
        "min_cjk_ratio": min_cjk_ratio,
        "max_chars": max_chars,
        "sources": source_stats,
    }
    logger.info("Built integrated behavior SFT dataset: %s", result)
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build one mixed SFT dataset from public, feedback, weak-input, and noisy-distress data.")
    parser.add_argument("--out", default="data/training/integrated_behavior/integrated_behavior_train_ms_swift.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-cjk-ratio", type=float, default=0.45)
    parser.add_argument("--max-chars", type=int, default=2600)
    parser.add_argument("--public-chat", default=DEFAULT_SOURCE_SPECS[0][1])
    parser.add_argument("--public-safety", default=DEFAULT_SOURCE_SPECS[1][1])
    parser.add_argument("--weak-input", default=DEFAULT_SOURCE_SPECS[2][1])
    parser.add_argument("--feedback-behavior", default=DEFAULT_SOURCE_SPECS[3][1])
    parser.add_argument("--noisy-distress", default=DEFAULT_SOURCE_SPECS[4][1])
    parser.add_argument("--role-boundary", default=DEFAULT_SOURCE_SPECS[5][1])
    parser.add_argument("--public-external", default="data/training/public_external/public_external_candidate_sft_ms_swift.jsonl")
    parser.add_argument("--public-chat-limit", type=int, default=1200)
    parser.add_argument("--public-safety-limit", type=int, default=200)
    parser.add_argument("--weak-input-limit", type=int, default=300)
    parser.add_argument("--feedback-behavior-limit", type=int, default=0, help="0 means keep all valid rows.")
    parser.add_argument("--noisy-distress-limit", type=int, default=0, help="0 means keep all valid rows.")
    parser.add_argument("--role-boundary-limit", type=int, default=0, help="0 means keep all valid rows.")
    parser.add_argument("--public-external-limit", type=int, default=0, help="0 disables external English data; set a small number only after review.")
    args = parser.parse_args()

    result = build_integrated_behavior_sft_dataset(
        out=args.out,
        seed=args.seed,
        min_cjk_ratio=args.min_cjk_ratio,
        max_chars=args.max_chars,
        specs=_resolve_specs(root, args),
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
