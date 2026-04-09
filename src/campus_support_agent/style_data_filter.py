from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .logging_utils import get_logger


logger = get_logger("style_data_filter")


THERAPY_HEAVY_TERMS = {
    "理情行为疗法",
    "认知行为疗法",
    "REBT",
    "CBT",
    "非理性信念",
    "治疗师",
    "诊断术语",
    "DSM",
}

OVERLY_DIRECTIVE_TERMS = {
    "你应该",
    "必须",
    "一定要",
    "马上去",
    "不要再",
}


def _contains_any(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms)


def _quality_score(messages: list[dict[str, str]]) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []

    if len(messages) >= 8:
        score += 2
    elif len(messages) >= 4:
        score += 1
    else:
        reasons.append("轮次过少")

    assistant_turns = [item["content"] for item in messages if item["role"] == "assistant"]
    user_turns = [item["content"] for item in messages if item["role"] == "user"]

    if assistant_turns and user_turns:
        score += 1
    else:
        reasons.append("角色结构不完整")

    if any("？" in text or "?" in text for text in assistant_turns):
        score += 1
    else:
        reasons.append("缺少追问")

    if any(any(token in text for token in ["理解", "听起来", "辛苦", "可以理解", "sounds", "understand"]) for text in assistant_turns):
        score += 1
    else:
        reasons.append("共情表达偏弱")

    avg_assistant_len = sum(len(text) for text in assistant_turns) / max(len(assistant_turns), 1)
    if avg_assistant_len < 8:
        reasons.append("回复过短")
    elif avg_assistant_len > 260:
        reasons.append("回复偏长")
    else:
        score += 1

    joined_text = "\n".join(item["content"] for item in messages)
    if _contains_any(joined_text, THERAPY_HEAVY_TERMS):
        reasons.append("治疗学派痕迹较强")
    else:
        score += 1

    if _contains_any(joined_text, OVERLY_DIRECTIVE_TERMS):
        reasons.append("存在较强指令式表达")
    else:
        score += 1

    return score, reasons


def classify_style_sample(sample: dict[str, Any]) -> dict[str, Any]:
    messages = sample.get("messages") or []
    filtered_messages = [item for item in messages if item.get("role") in {"user", "assistant"}]
    score, reasons = _quality_score(filtered_messages)

    if len(filtered_messages) < 4:
        bucket = "drop"
    elif score >= 6 and "治疗学派痕迹较强" not in reasons:
        bucket = "keep"
    elif score >= 4:
        bucket = "review"
    else:
        bucket = "drop"

    result = {
        "bucket": bucket,
        "quality_score": score,
        "quality_reasons": reasons,
        "sample": sample,
    }
    return result


def triage_style_dataset(input_path: str, output_dir: str) -> dict[str, int]:
    source = Path(input_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    keep_path = output / "style_keep.jsonl"
    review_path = output / "style_review.jsonl"
    drop_path = output / "style_drop.jsonl"

    counts = {"keep": 0, "review": 0, "drop": 0}
    with source.open("r", encoding="utf-8") as handle, \
        keep_path.open("w", encoding="utf-8") as keep_file, \
        review_path.open("w", encoding="utf-8") as review_file, \
        drop_path.open("w", encoding="utf-8") as drop_file:

        for line in handle:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            classified = classify_style_sample(sample)
            bucket = classified["bucket"]
            counts[bucket] += 1

            target_file = {
                "keep": keep_file,
                "review": review_file,
                "drop": drop_file,
            }[bucket]
            target_file.write(json.dumps(classified, ensure_ascii=False) + "\n")

    logger.info(
        "Triaged style dataset from %s into keep=%s review=%s drop=%s",
        source,
        counts["keep"],
        counts["review"],
        counts["drop"],
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Triage style-SFT dataset into keep/review/drop buckets.")
    parser.add_argument("--input", required=True, help="Input style JSONL file.")
    parser.add_argument("--outdir", required=True, help="Output directory for triage files.")
    args = parser.parse_args()

    counts = triage_style_dataset(args.input, args.outdir)
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
