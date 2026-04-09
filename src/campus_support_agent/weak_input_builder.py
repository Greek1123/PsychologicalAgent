from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .logging_utils import get_logger


logger = get_logger("weak_input_builder")


def _zh_records() -> list[dict[str, Any]]:
    """Curated Chinese weak-input repair samples.

    These records teach the model not to blindly continue counting or loop on
    backchannel tokens such as "嗯" and "?". Instead, the model should clarify,
    re-anchor the topic, and offer a small next step.
    """
    return [
        {
            "id": "weak-zh-001",
            "language": "zh",
            "task_type": "general_multiturn_dialogue",
            "stage_goal": "weak_input_repair",
            "messages": [
                {"role": "user", "content": "我最近心里很乱。"},
                {"role": "assistant", "content": "我在，你可以慢一点说。现在最卡住你的是什么？"},
                {"role": "user", "content": "嗯"},
                {
                    "role": "assistant",
                    "content": "没关系，卡住也很正常。你不用一次讲完整，只要告诉我你现在更接近烦、累，还是慌就可以。",
                },
            ],
        },
        {
            "id": "weak-zh-002",
            "language": "zh",
            "task_type": "general_multiturn_dialogue",
            "stage_goal": "weak_input_repair",
            "messages": [
                {"role": "user", "content": "我和室友最近关系很差。"},
                {"role": "assistant", "content": "听起来你已经被这件事压了几天。是发生了冲突，还是一直在冷着？"},
                {"role": "user", "content": "?"},
                {
                    "role": "assistant",
                    "content": "我的意思是，我们先分清楚是吵起来了，还是一直不说话。你只要选一个更接近的情况就行。",
                },
            ],
        },
        {
            "id": "weak-zh-003",
            "language": "zh",
            "task_type": "general_multiturn_dialogue",
            "stage_goal": "weak_input_repair",
            "messages": [
                {"role": "user", "content": "最近考试很多，我睡不好。"},
                {"role": "assistant", "content": "听起来你现在既累又紧绷。昨晚大概睡了多久？"},
                {"role": "user", "content": "1"},
                {
                    "role": "assistant",
                    "content": "如果你是在说只睡了一小时，那确实会很难受；如果你是在试探我也没关系。你可以直接告诉我，你昨晚大概睡了几小时。",
                },
            ],
        },
        {
            "id": "weak-zh-004",
            "language": "zh",
            "task_type": "general_multiturn_dialogue",
            "stage_goal": "weak_input_repair",
            "messages": [
                {"role": "user", "content": "我感觉大家都在针对我。"},
                {"role": "assistant", "content": "这听起来很压抑。是最近发生了什么具体的事，还是这种感觉已经持续一阵子了？"},
                {"role": "user", "content": "不知道"},
                {
                    "role": "assistant",
                    "content": "不知道也没关系，那我们就不急着下判断。你先回想一下，今天有没有哪一件小事让你最不舒服？",
                },
            ],
        },
        {
            "id": "weak-zh-005",
            "language": "zh",
            "task_type": "general_multiturn_dialogue",
            "stage_goal": "weak_input_repair",
            "messages": [
                {"role": "user", "content": "我其实不太想说。"},
                {"role": "assistant", "content": "可以，我们不用勉强自己。你更想先说事情经过，还是先说你现在的感觉？"},
                {"role": "user", "content": "算了"},
                {
                    "role": "assistant",
                    "content": "如果现在不想展开，我们也可以先停在这里。临走前你只要告诉我一句，你现在最希望先缓解的是烦躁、委屈，还是慌乱？",
                },
            ],
        },
        {
            "id": "weak-zh-006",
            "language": "zh",
            "task_type": "general_multiturn_dialogue",
            "stage_goal": "weak_input_repair",
            "messages": [
                {"role": "user", "content": "我有点怕回宿舍。"},
                {"role": "assistant", "content": "这句话挺重要的。我先不追问太多，你愿意告诉我是怕见到人，还是怕再起冲突吗？"},
                {"role": "user", "content": "哦"},
                {
                    "role": "assistant",
                    "content": "我们可以慢一点。你如果还没想好怎么讲，就先告诉我：回宿舍前你身体会不会已经开始紧绷，比如心慌或者头疼？",
                },
            ],
        },
        {
            "id": "weak-zh-007",
            "language": "zh",
            "task_type": "general_multiturn_dialogue",
            "stage_goal": "weak_input_repair",
            "messages": [
                {"role": "user", "content": "你在吗"},
                {"role": "assistant", "content": "我在。你可以慢慢来，不用马上组织得很完整。"},
                {"role": "user", "content": "..."},
                {
                    "role": "assistant",
                    "content": "没关系，先空一下也可以。你只用回答我一个最小的问题：现在最困扰你的是学习、人际，还是睡眠？",
                },
            ],
        },
        {
            "id": "weak-zh-008",
            "language": "zh",
            "task_type": "general_multiturn_dialogue",
            "stage_goal": "weak_input_repair",
            "messages": [
                {"role": "user", "content": "我脑子一团乱。"},
                {"role": "assistant", "content": "那我们先别一下子理全部。你刚刚最先想到的，是考试、关系，还是家里的事？"},
                {"role": "user", "content": "2"},
                {
                    "role": "assistant",
                    "content": "如果你是在选第二个，那我理解成你更偏向关系上的压力；如果不是，也可以直接用一句话告诉我是哪一块最乱。",
                },
            ],
        },
        {
            "id": "weak-zh-009",
            "language": "zh",
            "task_type": "general_multiturn_dialogue",
            "stage_goal": "weak_input_repair",
            "messages": [
                {"role": "user", "content": "今天真的很烦。"},
                {"role": "assistant", "content": "我听到了。你要是现在没力气细讲，也可以先告诉我，是哪一个瞬间把你情绪顶上去的？"},
                {"role": "user", "content": "啊"},
                {
                    "role": "assistant",
                    "content": "像是那一下又想起来了。你不用详细复述，只要告诉我，是和某个人有关，还是和一件事有关。",
                },
            ],
        },
        {
            "id": "weak-zh-010",
            "language": "zh",
            "task_type": "general_multiturn_dialogue",
            "stage_goal": "weak_input_repair",
            "messages": [
                {"role": "user", "content": "我现在就想随便聊两句。"},
                {"role": "assistant", "content": "可以，我们不一定非得一下子聊重的。最近有没有什么事让你一直放不下？"},
                {"role": "user", "content": "随便"},
                {
                    "role": "assistant",
                    "content": "那我帮你把范围缩小一点。你最近更容易被学习、人际，还是睡眠打断心情？",
                },
            ],
        },
    ]


def _en_records() -> list[dict[str, Any]]:
    """Curated English weak-input repair samples for bilingual chat stability."""
    return [
        {
            "id": "weak-en-001",
            "language": "en",
            "task_type": "general_multiturn_dialogue",
            "stage_goal": "weak_input_repair",
            "messages": [
                {"role": "user", "content": "I feel stuck and I do not know how to explain it."},
                {"role": "assistant", "content": "That is okay. We can go slowly. What feels heaviest right now?"},
                {"role": "user", "content": "hmm"},
                {
                    "role": "assistant",
                    "content": "No rush. You do not need a full story yet. Just tell me whether it feels more like stress, sadness, or fear.",
                },
            ],
        },
        {
            "id": "weak-en-002",
            "language": "en",
            "task_type": "general_multiturn_dialogue",
            "stage_goal": "weak_input_repair",
            "messages": [
                {"role": "user", "content": "I think my roommates are avoiding me."},
                {"role": "assistant", "content": "That sounds painful. Did something specific happen, or has the distance been building up for a while?"},
                {"role": "user", "content": "?"},
                {
                    "role": "assistant",
                    "content": "I mean: was there one clear conflict, or is it more of a slow change in the atmosphere? Either answer is fine.",
                },
            ],
        },
        {
            "id": "weak-en-003",
            "language": "en",
            "task_type": "general_multiturn_dialogue",
            "stage_goal": "weak_input_repair",
            "messages": [
                {"role": "user", "content": "I barely slept before my exam."},
                {"role": "assistant", "content": "That sounds exhausting. Roughly how much sleep did you get?"},
                {"role": "user", "content": "1"},
                {
                    "role": "assistant",
                    "content": "If you mean one hour, that is really rough. If you are just testing me, that is okay too. You can simply tell me how many hours you slept.",
                },
            ],
        },
        {
            "id": "weak-en-004",
            "language": "en",
            "task_type": "general_multiturn_dialogue",
            "stage_goal": "weak_input_repair",
            "messages": [
                {"role": "user", "content": "I do not really want to talk about it."},
                {"role": "assistant", "content": "You do not have to force it. We can keep it small. Do you want to name the event, or just the feeling?"},
                {"role": "user", "content": "whatever"},
                {
                    "role": "assistant",
                    "content": "That is okay. Let me narrow it down for us: is the pressure coming more from school, people, or sleep lately?",
                },
            ],
        },
        {
            "id": "weak-en-005",
            "language": "en",
            "task_type": "general_multiturn_dialogue",
            "stage_goal": "weak_input_repair",
            "messages": [
                {"role": "user", "content": "Are you there"},
                {"role": "assistant", "content": "I am here. Take your time."},
                {"role": "user", "content": "..."},
                {
                    "role": "assistant",
                    "content": "We can start with one tiny step. Which word is closer right now: overwhelmed, lonely, or angry?",
                },
            ],
        },
    ]


def build_weak_input_dataset(output_path: str) -> dict[str, int]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    records = _zh_records() + _en_records()
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    stats = {
        "written": len(records),
        "zh": len([record for record in records if record["language"] == "zh"]),
        "en": len([record for record in records if record["language"] == "en"]),
    }
    logger.info("Built weak-input repair dataset at %s with stats=%s", output, stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a weak-input repair dataset for chat-first SFT.")
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    args = parser.parse_args()

    stats = build_weak_input_dataset(args.out)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
