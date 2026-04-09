from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .logging_utils import get_logger


logger = get_logger("style_dataset_splitter")


def _unwrap_sample(record: dict[str, Any]) -> dict[str, Any]:
    """Allow the splitter to read both raw samples and triaged wrapper records."""
    if isinstance(record.get("sample"), dict):
        return record["sample"]
    return record


def _bucket_key(sample: dict[str, Any]) -> str:
    language = str(sample.get("language", "unknown"))
    stage_goal = str(sample.get("stage_goal", "unknown"))
    return f"{language}::{stage_goal}"


def split_style_dataset(
    input_path: str,
    output_dir: str,
    *,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, int]:
    if round(train_ratio + dev_ratio + test_ratio, 5) != 1.0:
        raise ValueError("train/dev/test ratio must sum to 1.0")

    source = Path(input_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    with source.open("r", encoding="utf-8") as handle:
        samples = [_unwrap_sample(json.loads(line)) for line in handle if line.strip()]

    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        buckets[_bucket_key(sample)].append(sample)

    splits = {"train": [], "dev": [], "test": []}
    for group in buckets.values():
        rng.shuffle(group)
        total = len(group)
        train_cut = int(total * train_ratio)
        dev_cut = train_cut + int(total * dev_ratio)

        # Keep language/stage distributions roughly aligned across the three splits.
        splits["train"].extend(group[:train_cut])
        splits["dev"].extend(group[train_cut:dev_cut])
        splits["test"].extend(group[dev_cut:])

    for split_name, split_samples in splits.items():
        rng.shuffle(split_samples)
        target = output / f"style_{split_name}.jsonl"
        with target.open("w", encoding="utf-8") as handle:
            for sample in split_samples:
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    counts = {name: len(items) for name, items in splits.items()}
    logger.info(
        "Split style dataset %s into train=%s dev=%s test=%s",
        source,
        counts["train"],
        counts["dev"],
        counts["test"],
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Split style SFT dataset into train/dev/test.")
    parser.add_argument("--input", required=True, help="Input style JSONL path.")
    parser.add_argument("--outdir", required=True, help="Output directory for split JSONL files.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    counts = split_style_dataset(args.input, args.outdir, seed=args.seed)
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
