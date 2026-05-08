from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CRISIS_TERMS = ("自杀", "不想活", "想死", "结束生命", "伤害自己", "活不下去")
BAD_CRISIS_CHOSEN_TERMS = ("如果你不想多说也完全没关系", "不想多说也没关系", "可以尝试联系")
CONSULTATION_OPENER_TERMS = ("感谢你前来咨询", "感谢你来寻求帮助", "能详细告诉我你的困扰吗", "能详细告诉我一下你的困扰吗")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _chosen(record: dict[str, Any]) -> str:
    return str((record.get("sft_draft") or {}).get("chosen") or "").strip()


def _set_chosen(record: dict[str, Any], value: str) -> None:
    record.setdefault("sft_draft", {})["chosen"] = value
    record.setdefault("failure_review", {})["preferred_reply"] = value


def _input_text(record: dict[str, Any]) -> str:
    return str(record.get("input_text") or "").strip()


def _is_crisis(record: dict[str, Any]) -> bool:
    text = _input_text(record)
    return any(term in text for term in CRISIS_TERMS)


def _has_bad_crisis_chosen(record: dict[str, Any]) -> bool:
    chosen = _chosen(record)
    return any(term in chosen for term in BAD_CRISIS_CHOSEN_TERMS)


def _has_consultation_opener(record: dict[str, Any]) -> bool:
    rejected = str(record.get("assistant_reply") or "")
    chosen = _chosen(record)
    return any(term in rejected or term in chosen for term in CONSULTATION_OPENER_TERMS)


def _crisis_chosen() -> str:
    return (
        "你说到想自杀，我会先把安全放在第一位。请你现在不要一个人待着，尽快联系身边能立刻到你身边的人，"
        "比如室友、同学、辅导员或家人；如果有马上伤害自己的风险，请立即拨打当地急救电话或联系学校心理危机支持。"
        "如果可以，先把可能伤害自己的东西放远一点，然后只回我一句：你现在身边有人吗？"
    )


def _anti_template_chosen(record: dict[str, Any]) -> str | None:
    text = _input_text(record)
    if any(term in text for term in ("考试", "挂科", "成绩", "复习")):
        return (
            "你现在像是被考试和结果压住了，不是简单一句“别担心”就能过去。"
            "我们先把目标放小一点：只挑一个最急的任务，先做 15 分钟，别一下子要求自己解决全部。"
        )
    if any(term in text for term in ("怕别人知道", "告诉别人", "隐私", "保密", "不想说")):
        return (
            "你担心别人知道，这个顾虑很重要。你不用说姓名、宿舍号、具体对象这些能识别身份的信息；"
            "我们可以只聊你的感受和你现在需要什么支持。如果你不想展开，也完全可以先停在这里。"
        )
    return None


def clean_reviewed_feedback_dataset(input_path: str, output_path: str) -> dict[str, Any]:
    source = Path(input_path)
    output = Path(output_path)
    cleaned: list[dict[str, Any]] = []
    crisis_fixed = 0
    template_fixed = 0
    for record in _read_jsonl(source):
        updated = dict(record)
        if _is_crisis(updated) and _has_bad_crisis_chosen(updated):
            _set_chosen(updated, _crisis_chosen())
            feedback = updated.setdefault("feedback", {})
            tags = list(feedback.get("tags") or [])
            if "crisis_safety_rewrite" not in tags:
                tags.append("crisis_safety_rewrite")
            feedback["tags"] = tags
            crisis_fixed += 1
        elif _has_consultation_opener(updated):
            replacement = _anti_template_chosen(updated)
            if replacement:
                _set_chosen(updated, replacement)
                feedback = updated.setdefault("feedback", {})
                tags = list(feedback.get("tags") or [])
                if "consultation_opener" not in tags:
                    tags.append("consultation_opener")
                feedback["tags"] = tags
                template_fixed += 1
        cleaned.append(updated)

    _write_jsonl(output, cleaned)
    return {
        "input": str(source),
        "output": str(output),
        "records": len(cleaned),
        "crisis_fixed": crisis_fixed,
        "template_fixed": template_fixed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean reviewed feedback data before DPO conversion.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(clean_reviewed_feedback_dataset(args.input, args.out), ensure_ascii=False))


if __name__ == "__main__":
    main()
