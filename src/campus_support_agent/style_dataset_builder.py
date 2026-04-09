from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .logging_utils import get_logger


logger = get_logger("style_dataset_builder")


def _language_from_filename(path: Path) -> str:
    name = path.name.lower()
    if "cn_" in name or name.startswith("cn"):
        return "zh"
    if "en_" in name or name.startswith("en"):
        return "en"
    return "unknown"


def _map_stage(stage: str, language: str) -> str:
    normalized = (stage or "").strip().lower()
    if not normalized:
        return "gentle_probe_and_reframe"

    if "early" in normalized or "初期" in normalized or "早期" in normalized:
        return "emotional_containment"
    if "middle" in normalized or "中期" in normalized:
        return "gentle_probe_and_reframe"
    if "late" in normalized or "后期" in normalized or "结束" in normalized:
        return "summary_and_next_step"

    # 如果原字段不标准，默认走中间阶段。
    if language in {"zh", "en"}:
        return "gentle_probe_and_reframe"
    return "unknown"


def _system_prompt(language: str) -> str:
    if language == "en":
        return (
            "You are a campus mental health support assistant. "
            "Be warm, natural, empathic, non-judgmental, and help the student keep talking. "
            "Do not reveal hidden analysis or internal reasoning."
        )
    return (
        "你是一个校园心理支持助手。你的目标是自然、温和、共情、不过度说教地支持来访者继续表达。"
        "不要把内部分析过程直接讲出来。"
    )


def _convert_dialog(dialog: list[dict[str, Any]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for turn in dialog:
        speaker = str(turn.get("speaker", "")).strip().lower()
        content = str(turn.get("content", "")).strip()
        if not content:
            continue

        if speaker == "supporter":
            role = "assistant"
        elif speaker == "seeker":
            role = "user"
        else:
            continue

        messages.append({"role": role, "content": content})
    return messages


def _is_valid_style_dialog(messages: list[dict[str, str]]) -> bool:
    # 至少包含一轮 user/assistant 往返，才对风格训练有意义。
    if len(messages) < 4:
        return False

    roles = {item["role"] for item in messages}
    if "user" not in roles or "assistant" not in roles:
        return False

    return True


def build_style_sft_samples_from_file(path: str) -> list[dict[str, Any]]:
    source_path = Path(path)
    language = _language_from_filename(source_path)
    data = json.loads(source_path.read_text(encoding="utf-8"))
    samples: list[dict[str, Any]] = []

    for record in data:
        raw_dialog = record.get("dialog") or []
        messages = _convert_dialog(raw_dialog)
        if not _is_valid_style_dialog(messages):
            continue

        # 这里只保留“会聊”所需的字段，不把 reasoning/guide 等内部说明直接喂给模型。
        sample = {
            "id": f"style-{language}-{record.get('dialog_id', len(samples))}",
            "language": language,
            "task_type": "style_support",
            "stage_goal": _map_stage(str(record.get("stage", "")), language),
            "messages": [{"role": "system", "content": _system_prompt(language)}, *messages],
            "meta": {
                "topic": record.get("topic", ""),
                "theme": record.get("theme", ""),
                "psychotherapy": record.get("psychotherapy", ""),
                "summary": record.get("summary", ""),
                "source_file": source_path.name,
            },
        }
        samples.append(sample)

    logger.info("Converted %s valid style samples from %s", len(samples), source_path.name)
    return samples


def write_style_sft_dataset(input_paths: list[str], output_path: str) -> int:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    all_samples: list[dict[str, Any]] = []
    for path in input_paths:
        all_samples.extend(build_style_sft_samples_from_file(path))

    with output.open("w", encoding="utf-8") as handle:
        for sample in all_samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info("Wrote %s bilingual style SFT samples to %s", len(all_samples), output)
    return len(all_samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bilingual style-SFT JSONL from raw counseling dialogues.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input JSON files such as cn_data_version7.json and en_data_version7.json",
    )
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    args = parser.parse_args()

    count = write_style_sft_dataset(args.inputs, args.out)
    print(f"wrote {count} samples")


if __name__ == "__main__":
    main()
