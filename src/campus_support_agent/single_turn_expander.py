from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .logging_utils import get_logger


logger = get_logger("single_turn_expander")


def _detect_language(text: str) -> str:
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    if re.search(r"[A-Za-z]", text):
        return "en"
    return "unknown"


def _system_prompt(language: str) -> str:
    if language == "en":
        return (
            "You are a campus mental health support assistant. "
            "Be natural, empathic, warm, and avoid exposing internal analysis."
        )
    return "你是一个校园心理支持助手。你的表达要自然、共情、温和，不要暴露内部分析过程。"


def _probe_turn(language: str, user_text: str) -> str:
    if language == "en":
        return (
            "It sounds like this has been sitting heavily with you. "
            "If we slow it down for a moment, what feels hardest about this right now?"
        )
    return "听起来这件事已经在你心里压了一阵子了。如果我们先慢一点看，你现在觉得最难受的部分是什么？"


def _elaboration_turn(language: str, user_text: str) -> str:
    if language == "en":
        return (
            "I think what makes it harder is that it keeps staying in my mind, "
            "and I do not really know how to settle myself down."
        )
    return "我觉得最难受的是这件事一直在脑子里转，我不知道该怎么让自己慢慢稳定下来。"


def _build_multiturn_sample(record: dict[str, Any], index: int) -> dict[str, Any]:
    user_text = str(record.get("input", "")).strip()
    assistant_text = str(record.get("output", "")).strip()
    instruction = str(record.get("instruction", "")).strip()
    language = _detect_language(f"{user_text}\n{assistant_text}")

    # 这里把单轮问答扩成 4 轮，对应“接住情绪 -> 轻追问 -> 用户展开 -> 支持回应”。
    return {
        "id": f"single-turn-expanded-{index:06d}",
        "language": language,
        "task_type": "style_support",
        "stage_goal": "gentle_probe_and_reframe",
        "messages": [
            {"role": "system", "content": _system_prompt(language)},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": _probe_turn(language, user_text)},
            {"role": "user", "content": _elaboration_turn(language, user_text)},
            {"role": "assistant", "content": assistant_text},
        ],
        "meta": {
            "instruction": instruction,
            "source": "single_turn_expansion",
        },
    }


def expand_single_turn_dataset(input_path: str, output_path: str) -> int:
    source = Path(input_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(source.read_text(encoding="utf-8"))
    samples: list[dict[str, Any]] = []
    for idx, record in enumerate(data, start=1):
        user_text = str(record.get("input", "")).strip()
        assistant_text = str(record.get("output", "")).strip()
        if not user_text or not assistant_text:
            continue
        samples.append(_build_multiturn_sample(record, idx))

    with output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info("Expanded %s single-turn samples from %s into %s", len(samples), source, output)
    return len(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand single-turn support data into multi-turn style SFT JSONL.")
    parser.add_argument("--input", required=True, help="Input single-turn JSON path.")
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    args = parser.parse_args()

    count = expand_single_turn_dataset(args.input, args.out)
    print(f"wrote {count} samples")


if __name__ == "__main__":
    main()
