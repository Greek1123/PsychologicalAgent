from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .logging_utils import get_logger


logger = get_logger("weak_input_builder")


def _record(record_id: str, language: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "id": record_id,
        "language": language,
        "task_type": "general_multiturn_dialogue",
        "stage_goal": "weak_input_repair",
        "messages": messages,
    }


def _zh_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    single_turns = [
        (
            "嗯",
            "我在。你不用急着解释很多，如果现在只想停一下也可以。要不要先用一个词说说现在更像是累、烦，还是有点慌？",
        ),
        (
            "？",
            "我可能刚才没有接住你的意思。我们可以先不展开，你只要告诉我：现在你更希望我陪你安静待一会儿，还是帮你把事情理一理？",
        ),
        (
            "1",
            "我不会接着数数字。你如果是在试我，也没关系；如果是不知道怎么开口，我们可以从最简单的开始：现在心里最明显的是烦、怕，还是累？",
        ),
        (
            "2",
            "收到。这个数字本身信息不多，我不乱猜。你可以只回一个字，比如“烦”“累”“怕”，我会按你的节奏来。",
        ),
        (
            "...",
            "没关系，沉默也可以算是一种表达。你不用马上把事情讲清楚，我先陪你把这一小会儿撑过去。",
        ),
        (
            "随便",
            "那我们就先不聊很重的内容。你可以选一个很小的话题开始：今天最累的一刻、最烦的一件事，或者现在身体哪里最紧。",
        ),
        (
            "不知道",
            "不知道也很正常，很多时候压力太多会让人说不清楚。我们先不用找原因，只判断一下：这份难受更像压力、人际，还是睡眠被影响？",
        ),
        (
            "我不想说",
            "可以的，你有权利不说细节。我不会逼你讲。那我们先只照顾当下：你现在需要被安慰、需要安静，还是需要一个很小的办法让自己缓一缓？",
        ),
        (
            "我不想被追问",
            "明白，我不会连续追问你。我们可以先停在这里，我只给你一个很轻的选择：要不要先做一次慢呼吸，或者只是让我陪你待一会儿？",
        ),
        (
            "我怕你会告诉别人",
            "你担心隐私被别人知道，这个顾虑很重要。你不用告诉我姓名、宿舍号、具体对象这些信息；我们只谈你的感受和你现在需要什么支持就可以。",
        ),
        (
            "我害怕别人会知道",
            "我理解，这种害怕会让人更不敢开口。你可以不说任何能识别身份的细节，我们先只聊“这件事让你有多紧张、现在怎么让自己安全一点”。",
        ),
        (
            "但我不太想细说",
            "没关系，不细说也可以。我不会逼你把事情讲完整。我们可以只从当下开始：这件事现在对你的影响，是睡不好、一直想，还是不想见人？",
        ),
        (
            "我今天本来还好，一回宿舍就烦",
            "听起来宿舍像是一个会触发你情绪的地方。我们不急着追问发生了什么，先看看你回宿舍后的烦更像紧绷、委屈，还是想逃开？",
        ),
        (
            "我最近压力好大，晚上睡不好",
            "这已经影响到睡眠了，确实会很消耗。今晚先别要求自己一下解决所有压力，我们可以先把目标降到很小：让身体稍微放松一点，把脑子里的事先放一放。",
        ),
        (
            "我怕考试挂科",
            "考试压力很容易把人推到最坏结果里。我们先不急着证明你一定不会挂科，先把压力拆小一点：你最担心的是复习来不及、基础不稳，还是一想到考试就慌？",
        ),
        (
            "你会不会觉得我很矫情",
            "不会。你愿意把这种担心说出来，说明你其实很怕自己的感受不被接住。这里不用证明自己够严重，难受本身就值得被认真对待。",
        ),
        (
            "算了",
            "可以先算了，我们不硬聊。但我还是想把一句话放在这里：你不用一个人把所有情绪都憋完，等你愿意的时候，我们可以从很小的一点重新开始。",
        ),
        (
            "没事",
            "我先不急着把“没事”当成真的没事。也许你只是还没准备好说。那我们先轻一点：现在你更想分散注意力，还是想安静一会儿？",
        ),
        (
            "好烦",
            "这种烦可能已经堆了一阵子了。我们先不分析太多，你可以只告诉我：它更像被人影响的烦，还是事情太多压住的烦？",
        ),
        (
            "啊",
            "我在。你这个“啊”听起来像是不知道怎么接了。没关系，我们可以慢一点，我不会催你。",
        ),
    ]

    for index, (user, assistant) in enumerate(single_turns, start=1):
        records.append(_record(f"weak-zh-single-{index:03d}", "zh", [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]))

    multiturns = [
        [
            {"role": "user", "content": "我最近压力好大，晚上睡不好。"},
            {"role": "assistant", "content": "这听起来已经不只是普通的忙了，连睡眠都被影响会很难受。我们可以先不解决全部问题，只先看看今晚怎么让你舒服一点。"},
            {"role": "user", "content": "但我不太想细说。"},
            {"role": "assistant", "content": "可以，不细说也没关系。我不会追着你问细节。那我们先只做一个很小的判断：你现在更需要安静陪伴，还是需要一个能马上缓一缓的办法？"},
        ],
        [
            {"role": "user", "content": "我不是很想说，我害怕别人会知道。"},
            {"role": "assistant", "content": "你这么谨慎是可以理解的。你不用说任何能暴露身份的信息，我们可以只谈你的感受、压力程度，以及现在怎么保护好你自己。"},
            {"role": "user", "content": "我怕你会告诉别人。"},
            {"role": "assistant", "content": "我听到你最担心的是边界和隐私。那我们先把范围收小：不说名字、不说具体地点、不说具体对象，只说“这件事让你现在有多难受”，这样会不会安全一点？"},
        ],
        [
            {"role": "user", "content": "我一回宿舍就烦。"},
            {"role": "assistant", "content": "宿舍好像对你来说不是一个能放松的地方。我们不急着讲原因，先看当下：你回去后是更想躲开人，还是更容易被一点小事点燃？"},
            {"role": "user", "content": "我不想被追问。"},
            {"role": "assistant", "content": "明白，我不追问。那我们先把目标放低一点：今晚先找一个能让你少被打扰的位置，或者先戴耳机、洗个澡、离开宿舍十分钟，选一个最容易做到的就好。"},
        ],
        [
            {"role": "user", "content": "你喜欢喝奶茶吗？"},
            {"role": "assistant", "content": "偶尔会喜欢，尤其是天气热的时候。你是想随便聊聊放松一下，还是想用轻松的话题先缓一缓？"},
            {"role": "user", "content": "嗯"},
            {"role": "assistant", "content": "那我们就轻一点。你不用马上进入严肃话题，我可以先陪你聊几句日常，等你想说别的再转过去。"},
        ],
        [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "我不会把它当成数字游戏继续接。你如果只是不知道说什么，可以回我一个词：烦、累、怕、乱，都可以。"},
            {"role": "user", "content": "2"},
            {"role": "assistant", "content": "我还是不会接着数。我们先停一下，看看你是不是在试探我会不会乱回复。你可以放心，我会尽量按你的真实意思来。"},
        ],
        [
            {"role": "user", "content": "我压力很大。"},
            {"role": "assistant", "content": "压力大到说出来，通常已经不轻了。我们先不把它讲成大道理，你现在最需要的是有人听你说、帮你拆任务，还是先让情绪降下来？"},
            {"role": "user", "content": "？"},
            {"role": "assistant", "content": "我换个更简单的说法：你现在想要我陪你、安慰你，还是帮你想一个下一步？你可以只选一个。"},
        ],
    ]
    for index, messages in enumerate(multiturns, start=1):
        records.append(_record(f"weak-zh-multi-{index:03d}", "zh", messages))

    return records


def _en_records() -> list[dict[str, Any]]:
    turns = [
        ("hmm", "I am here. You do not need to explain everything right away. We can go slowly."),
        ("?", "I may not have understood you clearly. Would you rather pause for a moment, or have me help you sort out one small part?"),
        ("1", "I will not continue the number sequence. If you are testing me, that is okay; if you do not know how to start, one feeling word is enough."),
        ("2", "I will still not treat this as a counting game. You can answer with just one word: stressed, tired, scared, or confused."),
        ("I do not really want to talk about it.", "That is okay. I will not push for details. We can stay with what you need right now: comfort, quiet, or one small next step."),
        ("I am scared other people will find out.", "Privacy matters. You do not need to share names, places, or identifying details. We can only talk about how this is affecting you."),
        ("I am afraid you will tell someone.", "I hear that you are worried about being exposed. We can keep this very limited and avoid any identifying details."),
        ("I do not want to be questioned.", "Understood. I will not keep questioning you. We can pause here, or I can offer one gentle way to calm down."),
        ("I was fine earlier, but when I got back to my dorm I felt irritated.", "It sounds like the dorm environment may be triggering something. We do not need the full story yet; we can first notice whether it feels like tension, anger, or wanting to get away."),
        ("I have been under a lot of pressure and I cannot sleep.", "That sounds exhausting. Since sleep is already affected, we can lower the goal for tonight: not solving everything, just helping your body settle a little."),
    ]
    return [
        _record(f"weak-en-single-{index:03d}", "en", [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
        for index, (user, assistant) in enumerate(turns, start=1)
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
        "zh": sum(1 for record in records if record["language"] == "zh"),
        "en": sum(1 for record in records if record["language"] == "en"),
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
