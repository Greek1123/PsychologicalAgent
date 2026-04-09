from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .logging_utils import get_logger


logger = get_logger("style_training_pack_builder")

_WEAK_USER_OPENINGS = {
    "嗯",
    "恩",
    "哦",
    "喔",
    "好",
    "好的",
    "是的",
    "对",
    "对的",
    "行",
    "可以",
    "知道了",
    "明白了",
    "收到",
    "yes",
    "yeah",
    "yep",
    "ok",
    "okay",
    "sure",
    "fine",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _unwrap_sample(record: dict[str, Any]) -> dict[str, Any]:
    """Support both direct style samples and triaged wrapper records."""
    if isinstance(record.get("sample"), dict):
        return record["sample"]
    return record


def _normalize_opening_text(text: str) -> str:
    lowered = text.strip().lower()
    return re.sub(r"[\s\W_]+", "", lowered, flags=re.UNICODE)


def _is_weak_user_opening(text: str) -> bool:
    """Filter out mid-session acknowledgements that teach poor continuation habits."""
    normalized = _normalize_opening_text(text)
    if not normalized:
        return True
    if normalized in _WEAK_USER_OPENINGS:
        return True
    if normalized.isdigit():
        return True
    # Very short openers are usually backchannel turns rather than usable session starts.
    if len(normalized) <= 2:
        return True
    return False


def _normalize_style_sample(record: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Keep only usable user-led dialogues for style SFT.

    Many imported records start in the middle of a therapy session. We strip leading
    assistant turns, require a meaningful first user message, and trim dangling
    non-assistant endings so the model learns to answer rather than blindly continue.
    """
    cloned = dict(_unwrap_sample(record))
    raw_messages = cloned.get("messages") or []
    messages = [
        {"role": str(message.get("role", "")).strip(), "content": str(message.get("content", "")).strip()}
        for message in raw_messages
        if str(message.get("content", "")).strip()
    ]
    if not messages:
        return None, "empty_messages"

    system_messages: list[dict[str, str]] = []
    body_start = 0
    for index, message in enumerate(messages):
        if message["role"] == "system":
            system_messages.append(message)
            body_start = index + 1
            continue
        break

    body = messages[body_start:]
    while body and body[0]["role"] != "user":
        body.pop(0)
    if not body:
        return None, "missing_user_start"
    if _is_weak_user_opening(body[0]["content"]):
        return None, "weak_user_start"

    # Drop trailing user-only turns so every sample still teaches an assistant reply.
    while body and body[-1]["role"] != "assistant":
        body.pop()
    if len(body) < 2:
        return None, "missing_assistant_reply"

    assistant_turns = sum(1 for message in body if message["role"] == "assistant")
    if assistant_turns == 0:
        return None, "missing_assistant_reply"

    cloned["messages"] = system_messages + body
    return cloned, None


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _tag_origin(records: list[dict[str, Any]], origin: str) -> list[dict[str, Any]]:
    tagged: list[dict[str, Any]] = []
    for record in records:
        cloned = dict(_unwrap_sample(record))
        meta = dict(cloned.get("meta") or {})
        meta["training_origin"] = origin
        cloned["meta"] = meta
        tagged.append(cloned)
    return tagged


def _language_breakdown(records: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(str(record.get("language", "unknown")) for record in records)
    return dict(counter)


def _clean_style_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    cleaned: list[dict[str, Any]] = []
    dropped = Counter()
    for record in records:
        normalized, reason = _normalize_style_sample(record)
        if normalized is None:
            dropped[reason or "unknown"] += 1
            continue
        cleaned.append(normalized)
    return cleaned, dict(dropped)


def build_style_training_pack(
    style_train_path: str,
    style_dev_path: str,
    style_test_path: str,
    expanded_single_turn_path: str,
    preference_path: str,
    output_dir: str,
    *,
    synthetic_ratio: float = 0.25,
    seed: int = 42,
) -> dict[str, Any]:
    if synthetic_ratio < 0:
        raise ValueError("synthetic_ratio must be non-negative.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    style_train = _read_jsonl(Path(style_train_path))
    style_dev = _read_jsonl(Path(style_dev_path))
    style_test = _read_jsonl(Path(style_test_path))
    expanded_single_turn = _read_jsonl(Path(expanded_single_turn_path))
    preference_templates = _read_jsonl(Path(preference_path))

    cleaned_style_train, dropped_train = _clean_style_records(style_train)
    cleaned_style_dev, dropped_dev = _clean_style_records(style_dev)
    cleaned_style_test, dropped_test = _clean_style_records(style_test)

    rng = random.Random(seed)
    synthetic_limit = min(len(expanded_single_turn), int(len(cleaned_style_train) * synthetic_ratio))
    synthetic_subset = list(expanded_single_turn)
    if synthetic_limit and synthetic_limit < len(expanded_single_turn):
        synthetic_subset = rng.sample(expanded_single_turn, synthetic_limit)
    elif synthetic_limit == 0:
        synthetic_subset = []

    phase1_train = _tag_origin(cleaned_style_train, "real_multiturn") + _tag_origin(
        synthetic_subset,
        "single_turn_expansion",
    )
    rng.shuffle(phase1_train)

    phase1_dev = _tag_origin(cleaned_style_dev, "real_multiturn")
    phase1_test = _tag_origin(cleaned_style_test, "real_multiturn")

    phase1_train_path = output / "style_phase1_train.jsonl"
    phase1_dev_path = output / "style_phase1_dev.jsonl"
    phase1_test_path = output / "style_phase1_test.jsonl"
    phase2_preference_path = output / "style_phase2_preference.jsonl"
    manifest_path = output / "style_training_manifest.json"

    _write_jsonl(phase1_train_path, phase1_train)
    _write_jsonl(phase1_dev_path, phase1_dev)
    _write_jsonl(phase1_test_path, phase1_test)
    _write_jsonl(phase2_preference_path, preference_templates)

    manifest = {
        "phase1": {
            "raw_real_multiturn_train": len(style_train),
            "clean_real_multiturn_train": len(cleaned_style_train),
            "synthetic_train": len(synthetic_subset),
            "final_train": len(phase1_train),
            "raw_dev": len(style_dev),
            "dev": len(phase1_dev),
            "raw_test": len(style_test),
            "test": len(phase1_test),
            "dropped_dialogues": {
                "train": dropped_train,
                "dev": dropped_dev,
                "test": dropped_test,
            },
            "language_breakdown": {
                "train": _language_breakdown(phase1_train),
                "dev": _language_breakdown(phase1_dev),
                "test": _language_breakdown(phase1_test),
            },
        },
        "phase2": {
            "preference_candidates": len(preference_templates),
        },
        "recommended_order": [
            "Phase 1: SFT on style_phase1_train/dev/test.jsonl",
            "Phase 2: annotate style_phase2_preference.jsonl for chosen/rejected pairs",
            "Phase 3: run human evaluation on the phase-1/phase-2 model checkpoints",
            "Phase 4: train the analysis layer separately with structured labels",
        ],
        "files": {
            "style_phase1_train": str(phase1_train_path),
            "style_phase1_dev": str(phase1_dev_path),
            "style_phase1_test": str(phase1_test_path),
            "style_phase2_preference": str(phase2_preference_path),
            "manifest": str(manifest_path),
        },
    }

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        "Built cleaned style training pack at %s with raw_train=%s clean_train=%s phase1_train=%s phase1_dev=%s phase1_test=%s phase2_preference=%s",
        output,
        len(style_train),
        len(cleaned_style_train),
        len(phase1_train),
        len(phase1_dev),
        len(phase1_test),
        len(preference_templates),
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a style-first training pack for staged fine-tuning.")
    parser.add_argument("--style-train", required=True, help="Clean real multi-turn train JSONL.")
    parser.add_argument("--style-dev", required=True, help="Clean real multi-turn dev JSONL.")
    parser.add_argument("--style-test", required=True, help="Clean real multi-turn test JSONL.")
    parser.add_argument("--expanded-single-turn", required=True, help="Expanded single-turn auxiliary JSONL.")
    parser.add_argument("--preference", required=True, help="Preference template JSONL.")
    parser.add_argument("--outdir", required=True, help="Output directory for the staged pack.")
    parser.add_argument("--synthetic-ratio", type=float, default=0.25, help="Synthetic-to-real ratio for phase 1.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    args = parser.parse_args()

    manifest = build_style_training_pack(
        args.style_train,
        args.style_dev,
        args.style_test,
        args.expanded_single_turn,
        args.preference,
        args.outdir,
        synthetic_ratio=args.synthetic_ratio,
        seed=args.seed,
    )
    print(json.dumps(manifest["phase1"], ensure_ascii=False))


if __name__ == "__main__":
    main()
