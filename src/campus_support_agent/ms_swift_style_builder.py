from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .logging_utils import get_logger


logger = get_logger("ms_swift_style_builder")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _unwrap_sample(record: dict[str, Any]) -> dict[str, Any]:
    """Support direct style records and triaged wrappers."""
    if isinstance(record.get("sample"), dict):
        return record["sample"]
    return record


def _to_sft_record(sample: dict[str, Any]) -> dict[str, Any]:
    """Export the minimal ms-swift SFT standard format."""
    sample = _unwrap_sample(sample)
    return {
        "messages": sample.get("messages", []),
    }


def _to_dpo_record(sample: dict[str, Any]) -> dict[str, Any] | None:
    """Export ms-swift DPO standard format when a rejected answer is present."""
    sample = _unwrap_sample(sample)
    prompt = sample.get("prompt", [])
    chosen = str(sample.get("chosen", "")).strip()
    rejected = str(sample.get("rejected", "")).strip()
    if not prompt or not chosen or not rejected:
        return None

    chosen_messages = [*prompt, {"role": "assistant", "content": chosen}]
    return {
        "messages": chosen_messages,
        "rejected_response": rejected,
    }


def build_ms_swift_style_datasets(style_pack_dir: str, output_dir: str) -> dict[str, Any]:
    pack_dir = Path(style_pack_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    phase1_train = _read_jsonl(pack_dir / "style_phase1_train.jsonl")
    phase1_dev = _read_jsonl(pack_dir / "style_phase1_dev.jsonl")
    phase1_test = _read_jsonl(pack_dir / "style_phase1_test.jsonl")
    phase2_preference = _read_jsonl(pack_dir / "style_phase2_preference.jsonl")

    sft_train = [_to_sft_record(sample) for sample in phase1_train]
    sft_dev = [_to_sft_record(sample) for sample in phase1_dev]
    sft_test = [_to_sft_record(sample) for sample in phase1_test]

    dpo_ready: list[dict[str, Any]] = []
    dpo_pending = 0
    for sample in phase2_preference:
        converted = _to_dpo_record(sample)
        if converted is None:
            dpo_pending += 1
            continue
        dpo_ready.append(converted)

    train_path = output / "style_phase1_train_ms_swift.jsonl"
    dev_path = output / "style_phase1_dev_ms_swift.jsonl"
    test_path = output / "style_phase1_test_ms_swift.jsonl"
    dpo_path = output / "style_phase2_preference_ms_swift.jsonl"
    manifest_path = output / "ms_swift_style_manifest.json"

    _write_jsonl(train_path, sft_train)
    _write_jsonl(dev_path, sft_dev)
    _write_jsonl(test_path, sft_test)
    _write_jsonl(dpo_path, dpo_ready)

    manifest = {
        "phase1": {
            "train": len(sft_train),
            "dev": len(sft_dev),
            "test": len(sft_test),
        },
        "phase2": {
            "ready_for_dpo": len(dpo_ready),
            "pending_rejected_annotation": dpo_pending,
        },
        "files": {
            "train": str(train_path),
            "dev": str(dev_path),
            "test": str(test_path),
            "preference": str(dpo_path),
            "manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        "Built ms-swift datasets at %s with train=%s dev=%s test=%s dpo_ready=%s pending=%s",
        output,
        len(sft_train),
        len(sft_dev),
        len(sft_test),
        len(dpo_ready),
        dpo_pending,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ms-swift-ready datasets from the style-first pack.")
    parser.add_argument("--style-pack", required=True, help="Input style-first pack directory.")
    parser.add_argument("--outdir", required=True, help="Output directory for ms-swift datasets.")
    args = parser.parse_args()

    manifest = build_ms_swift_style_datasets(args.style_pack, args.outdir)
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
