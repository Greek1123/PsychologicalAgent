from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .logging_utils import get_logger


logger = get_logger("general_phase0_mixer")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _record_signature(record: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    messages = record.get("messages", [])
    if not isinstance(messages, list):
        return tuple()

    signature: list[tuple[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        signature.append((str(message.get("role", "")), str(message.get("content", "")).strip()))
    return tuple(signature)


def _normalize_meta(meta: Any) -> dict[str, str]:
    if not isinstance(meta, dict):
        return {}
    normalized: dict[str, str] = {}
    for key, value in meta.items():
        normalized[str(key)] = str(value or "")
    return normalized


def build_mixed_general_phase0_dataset(
    base_input_path: str,
    augment_input_path: str,
    output_path: str,
    *,
    target_total: int = 8000,
    augment_ratio: float = 0.35,
    seed: int = 42,
) -> dict[str, int | float]:
    base_path = Path(base_input_path)
    augment_path = Path(augment_input_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    base_records = _load_jsonl(base_path)
    augment_records = _load_jsonl(augment_path)

    rng = random.Random(seed)
    rng.shuffle(base_records)
    rng.shuffle(augment_records)

    target_total = max(1, target_total)
    augment_target = min(len(augment_records), round(target_total * augment_ratio))
    base_target = min(len(base_records), target_total - augment_target)

    selected: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()

    def append_unique(records: list[dict[str, Any]], limit: int, source_name: str) -> int:
        added = 0
        for record in records:
            if added >= limit:
                break
            signature = _record_signature(record)
            if not signature or signature in seen:
                continue
            seen.add(signature)
            copied = dict(record)
            meta = _normalize_meta(copied.get("meta", {}))
            meta["mixed_source"] = source_name
            copied["meta"] = meta
            selected.append(copied)
            added += 1
        return added

    base_added = append_unique(base_records, base_target, "base_general_phase0")
    augment_added = append_unique(augment_records, augment_target, "augment_multiturn_0_8m")

    remaining = target_total - len(selected)
    if remaining > 0:
        remaining_added = append_unique(base_records, remaining, "base_general_phase0")
        remaining -= remaining_added
    if remaining > 0:
        append_unique(augment_records, remaining, "augment_multiturn_0_8m")

    rng.shuffle(selected)

    with output.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    stats: dict[str, int | float] = {
        "written": len(selected),
        "base_candidates": len(base_records),
        "augment_candidates": len(augment_records),
        "base_selected": sum(
            1 for record in selected if record.get("meta", {}).get("mixed_source") == "base_general_phase0"
        ),
        "augment_selected": sum(
            1 for record in selected if record.get("meta", {}).get("mixed_source") == "augment_multiturn_0_8m"
        ),
        "target_total": target_total,
        "augment_ratio": augment_ratio,
    }
    logger.info(
        "Built mixed phase-0 dataset at %s with written=%s base_selected=%s augment_selected=%s",
        output,
        stats["written"],
        stats["base_selected"],
        stats["augment_selected"],
    )
    return stats
