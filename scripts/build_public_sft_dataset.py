from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = ROOT / "data" / "public_training_datasets"
OUT_DIR = ROOT / "data" / "training" / "public_sft"

CHAT_SYSTEM = (
    "你是一个面向学生的心理支持助手。你要自然、温和、具体地回应用户，"
    "先接住情绪，再帮助用户把问题缩小到下一步。不要乱诊断，不要说教。"
)

SAFETY_SYSTEM = (
    "你是一个面向学生的心理安全支持助手。遇到自伤、自杀、伤人或极端绝望表达时，"
    "必须优先保护用户安全，建议联系现实中的可信任人员、学校心理中心或紧急服务。"
)


def clean_text(value: Any, *, max_chars: int = 1600) -> str:
    text = str(value or "").replace("\r", "\n")
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    text = " ".join(text.split())
    return text[:max_chars].strip()


def has_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def record(messages: list[dict[str, str]], source: str, **meta: Any) -> dict[str, Any]:
    return {
        "messages": messages,
        "meta": {
            "source": source,
            **{key: value for key, value in meta.items() if value is not None},
        },
    }


def system_user_assistant(system: str, user: str, assistant: str, source: str, **meta: Any) -> dict[str, Any] | None:
    user = clean_text(user, max_chars=1400)
    assistant = clean_text(assistant, max_chars=1800)
    if len(user) < 4 or len(assistant) < 8:
        return None
    return record(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        source,
        **meta,
    )


def from_cpsy_coun(limit: int) -> Iterable[dict[str, Any]]:
    path = PUBLIC_ROOT / "CPsyCoun" / "CPsyCounD.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    count = 0
    for item in data:
        messages = [{"role": "system", "content": CHAT_SYSTEM}]
        for user, assistant in item.get("history", []):
            user = clean_text(user)
            assistant = clean_text(assistant)
            if user and assistant:
                messages.append({"role": "user", "content": user})
                messages.append({"role": "assistant", "content": assistant})
        instruction = clean_text(item.get("instruction"))
        input_text = clean_text(item.get("input"))
        output = clean_text(item.get("output"))
        user_text = " ".join(part for part in [instruction, input_text] if part)
        if user_text and output:
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": output})
        if len(messages) >= 4:
            yield record(messages[-13:] if len(messages) > 13 else messages, "CPsyCoun")
            count += 1
        if count >= limit:
            break


def from_esconv(limit: int) -> Iterable[dict[str, Any]]:
    path = PUBLIC_ROOT / "ESConv" / "ESConv.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    count = 0
    for item in data:
        messages = [{"role": "system", "content": CHAT_SYSTEM}]
        for turn in item.get("dialog", []):
            speaker = turn.get("speaker")
            content = clean_text(turn.get("content"))
            if not content:
                continue
            if speaker == "seeker":
                messages.append({"role": "user", "content": content})
            elif speaker == "supporter":
                messages.append({"role": "assistant", "content": content})
        if len(messages) >= 5 and messages[-1]["role"] == "assistant":
            yield record(
                messages[-13:] if len(messages) > 13 else messages,
                "ESConv",
                language="en",
                emotion_type=item.get("emotion_type"),
                problem_type=item.get("problem_type"),
            )
            count += 1
        if count >= limit:
            break


def from_augesc(limit: int) -> Iterable[dict[str, Any]]:
    path = PUBLIC_ROOT / "AugESC" / "augesc.txt"
    if not path.exists():
        return
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                turns = json.loads(line)
            except json.JSONDecodeError:
                continue
            messages = [{"role": "system", "content": CHAT_SYSTEM}]
            for role, content in turns:
                content = clean_text(content)
                if not content:
                    continue
                if role == "usr":
                    messages.append({"role": "user", "content": content})
                elif role == "sys":
                    messages.append({"role": "assistant", "content": content})
            if len(messages) >= 5 and messages[-1]["role"] == "assistant":
                yield record(messages[-13:] if len(messages) > 13 else messages, "AugESC", language="en")
                count += 1
            if count >= limit:
                break


def from_mentalchat_csv(path: Path, source: str, limit: int) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    count = 0
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample = system_user_assistant(
                CHAT_SYSTEM,
                row.get("input", ""),
                row.get("output", ""),
                source,
                language="en",
            )
            if sample:
                yield sample
                count += 1
            if count >= limit:
                break


def from_chinese_psych_qa(limit: int) -> Iterable[dict[str, Any]]:
    path = ROOT / "data" / "Chinese-Psychological-QA-DataSet" / "ques_ans1.json"
    if not path.exists():
        path = ROOT / "data" / "Chinese-Psychological-QA-DataSet" / "ques_ans_sample.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    count = 0
    for item in data:
        ques = item.get("ques_info", {})
        title = clean_text(ques.get("title"), max_chars=260)
        content = clean_text(ques.get("content"), max_chars=1000)
        user = "。".join(part for part in [title, content] if part)
        answers = item.get("answers_info") or []
        answers = sorted(
            answers,
            key=lambda answer: int(str(answer.get("zan", "0")).strip() or 0),
            reverse=True,
        )
        for answer in answers[:2]:
            assistant = clean_text(answer.get("content"), max_chars=1600)
            if not has_chinese(user + assistant):
                continue
            sample = system_user_assistant(
                CHAT_SYSTEM,
                user,
                assistant,
                "Chinese-Psychological-QA-DataSet",
                language="zh",
                labels=ques.get("ques_label", []),
            )
            if sample:
                yield sample
                count += 1
                break
        if count >= limit:
            break


def from_psysuicide(limit: int) -> Iterable[dict[str, Any]]:
    paths = [
        PUBLIC_ROOT / "PsySUICIDE" / "train.json",
        PUBLIC_ROOT / "PsySUICIDE" / "valid.json",
        PUBLIC_ROOT / "PsySUICIDE" / "test.json",
    ]
    count = 0
    for path in paths:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data:
            text = clean_text(item.get("text"), max_chars=800)
            label = ",".join(str(label) for label in item.get("labels", []))
            if not text:
                continue
            assistant = (
                "我很担心你现在的安全。请先不要一个人扛着，尽快联系身边可信任的人、学校心理中心、辅导员，"
                "或者当地紧急服务。如果你已经有具体伤害自己的计划或工具，请立刻远离危险物品，并马上向现实中的人求助。"
            )
            sample = system_user_assistant(
                SAFETY_SYSTEM,
                text,
                assistant,
                "PsySUICIDE",
                language="zh",
                risk_label=label,
                task_type="safety",
            )
            if sample:
                yield sample
                count += 1
            if count >= limit:
                return


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def dedupe(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in records:
        signature = json.dumps(item.get("messages", []), ensure_ascii=False)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(item)
    return unique


def build_public_sft_dataset(
    *,
    chat_out: Path,
    safety_out: Path,
    seed: int = 42,
    cpsy_limit: int = 3000,
    chinese_qa_limit: int = 1200,
    esconv_limit: int = 1000,
    augesc_limit: int = 1000,
    mentalchat_interview_limit: int = 800,
    mentalchat_synth_limit: int = 600,
    safety_limit: int = 800,
) -> dict[str, Any]:
    rng = random.Random(seed)
    chat_records: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}

    def extend(source_name: str, iterator: Iterable[dict[str, Any]]) -> None:
        added = 0
        for sample in iterator:
            chat_records.append(sample)
            added += 1
        source_counts[source_name] = added

    extend("CPsyCoun", from_cpsy_coun(cpsy_limit))
    extend("Chinese-Psychological-QA-DataSet", from_chinese_psych_qa(chinese_qa_limit))
    extend("ESConv", from_esconv(esconv_limit))
    extend("AugESC", from_augesc(augesc_limit))
    extend(
        "MentalChat16K-Interview",
        from_mentalchat_csv(PUBLIC_ROOT / "MentalChat16K" / "Interview_Data_6K.csv", "MentalChat16K-Interview", mentalchat_interview_limit),
    )
    extend(
        "MentalChat16K-Synthetic",
        from_mentalchat_csv(PUBLIC_ROOT / "MentalChat16K" / "Synthetic_Data_10K.csv", "MentalChat16K-Synthetic", mentalchat_synth_limit),
    )

    chat_records = dedupe(chat_records)
    rng.shuffle(chat_records)
    write_jsonl(chat_out, chat_records)

    safety_records = dedupe(list(from_psysuicide(safety_limit)))
    rng.shuffle(safety_records)
    write_jsonl(safety_out, safety_records)

    return {
        "chat_out": str(chat_out),
        "safety_out": str(safety_out),
        "chat_written": len(chat_records),
        "safety_written": len(safety_records),
        "source_counts": source_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ms-swift SFT JSONL from public mental-health dialogue datasets.")
    parser.add_argument("--chat-out", default=str(OUT_DIR / "public_phase0_chat_train_ms_swift.jsonl"))
    parser.add_argument("--safety-out", default=str(OUT_DIR / "public_safety_train_ms_swift.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpsy-limit", type=int, default=3000)
    parser.add_argument("--chinese-qa-limit", type=int, default=1200)
    parser.add_argument("--esconv-limit", type=int, default=1000)
    parser.add_argument("--augesc-limit", type=int, default=1000)
    parser.add_argument("--mentalchat-interview-limit", type=int, default=800)
    parser.add_argument("--mentalchat-synth-limit", type=int, default=600)
    parser.add_argument("--safety-limit", type=int, default=800)
    args = parser.parse_args()

    stats = build_public_sft_dataset(
        chat_out=Path(args.chat_out),
        safety_out=Path(args.safety_out),
        seed=args.seed,
        cpsy_limit=args.cpsy_limit,
        chinese_qa_limit=args.chinese_qa_limit,
        esconv_limit=args.esconv_limit,
        augesc_limit=args.augesc_limit,
        mentalchat_interview_limit=args.mentalchat_interview_limit,
        mentalchat_synth_limit=args.mentalchat_synth_limit,
        safety_limit=args.safety_limit,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
