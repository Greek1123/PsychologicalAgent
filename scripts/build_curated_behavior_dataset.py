from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

EXAM_USERS = [
    "我最近很烦躁，快到期末考试了，我好害怕挂科，我压力好大。",
    "我一想到考试就睡不着，感觉脑子里全是挂科两个字。",
    "我复习不进去，越看越慌，感觉自己肯定完了。",
    "我怕这次成绩很差，已经连续几天晚上睡不好了。",
    "我感觉自己跟不上了，看到别人复习得很好我就更崩溃。",
]

DORM_USERS = [
    "我最近心情不好，一回到宿舍就感觉很烦。",
    "我见到舍友就烦，但我也不知道怎么说。",
    "宿舍里一点小声音都会让我很焦躁，我快受不了了。",
    "我不想回宿舍，回去就觉得压抑。",
    "我感觉舍友好像都在针对我，心里特别堵。",
]

PRIVACY_USERS = [
    "我不是很想说，我害怕别人会知道。",
    "我怕你会告诉别人，所以我不太敢说。",
    "我想说，但我又担心这些话被别人看到。",
    "这件事有点隐私，我不知道能不能讲。",
    "我不想细说，感觉说出来会很丢脸。",
]

WEAK_USERS = ["？", "嗯", "1", "2", "算了", "不知道", "不想说", "额"]

TYPO_USERS = [
    "我鸭力好大，晚上睡不召。",
    "我最近很奔溃，感觉脑子停不下来。",
    "我好海怕挂科，越想越睡不找。",
    "我一回数舍就烦，看到室友就难瘦。",
    "我真的不想或了，感觉撑不住。",
]

ROLE_CHALLENGE_USERS = [
    "你为啥感觉累？",
    "你怎么还有作业？",
    "大哥，你是人还是我是人？",
    "你也会挂科吗？",
    "你别说你自己，我是在说我。",
]

CASUAL_USERS = [
    "你喜欢喝奶茶吗？",
    "今天天气不错，晚上想去聚餐。",
    "你听过疯狂动物城吗？",
    "我想随便聊两句。",
]

EXAM_REPLIES = [
    "听起来你现在不是单纯不想复习，而是被挂科这件事压得很紧，连睡眠都被影响了。我们先不要一下子处理所有科目，先挑今晚最急的一科，把任务缩到二十分钟能开始的程度。你不用马上变得很有状态，先让自己从最小的一步恢复一点掌控感。",
    "你现在的害怕是有原因的：考试临近、复习压力和睡不好叠在一起，人很容易把结果想得特别糟。我们先把目标放低一点，不追求今晚全部补完，只先列出明天必须看的三个小点。这样不是逃避，而是在帮大脑从慌乱里先落地。",
    "我能感觉到你已经绷了好几天，越担心越睡不好，越睡不好又越觉得自己要完了。先暂停一下灾难化的循环，我们可以把问题拆成两个部分：今晚怎么先睡得稍微稳一点，明天怎么开始复习第一块内容。",
]

DORM_REPLIES = [
    "宿舍本来应该是休息的地方，但现在它像是一直在触发你的烦躁，这确实会很消耗。我们先不急着判断是不是谁对谁错，先照顾你今晚的状态。你可以先找一个相对安静的位置，或者短暂离开宿舍十分钟，让身体先从那个环境里退出来一点。",
    "听起来你不是无缘无故烦，而是回到宿舍这个场景时，情绪一下子被拉起来了。我们可以先把目标放得很小：今晚先减少冲突和刺激，不急着马上沟通清楚。等你缓一点，再一起想要不要说、怎么说、跟谁说。",
    "见到舍友就烦，说明这件事已经在你心里积了一阵子。你现在可以不用马上解释完整原因，我更想先确认你此刻能不能稍微安全、安静一点。先让自己离开刺激源一小会儿，或者戴上耳机、洗把脸，都可以算是有效处理。",
]

PRIVACY_REPLIES = [
    "你担心别人知道，这个顾虑很正常，我会尊重你的边界。你不需要把细节都说出来，也可以只说现在最明显的感受，比如害怕、委屈、烦躁，或者不安全。我们可以先从不涉及隐私的部分开始。",
    "可以不细说，保护自己的隐私很重要。我不会逼你把事情完整讲出来，你也可以只告诉我这件事带给你的影响，比如睡不好、心慌，还是一想到就难受。先讲到你觉得安全的位置就好。",
    "你能说出“我怕别人知道”，其实已经是在告诉我这件事对你很敏感。我们先不碰具体细节，只照顾你现在的感受。你可以用很模糊的方式说，比如“人际”“家里”“学习”，不用说名字和具体经过。",
]

WEAK_REPLIES = [
    "我看到你现在可能不太想展开说，也没关系。我们先不追问细节，你只要告诉我现在更需要我陪你待一会儿，还是给你一个很小的缓解办法就行。",
    "你只回了很短的一句，我会理解成你现在可能有点卡住了。没关系，不用组织完整语言。你可以只回一个词，比如“烦”“怕”“累”“宿舍”或者“考试”。",
    "先不用急着解释。很多人在很难受的时候确实只说得出一个字或者一个符号。我会先陪你把节奏放慢一点，你现在最明显的是身体累，还是心里乱？",
]

TYPO_REPLIES = [
    "我大概能明白，你可能是在说压力很大、晚上睡不着。情绪很崩的时候打错字很正常，不用因为这个觉得尴尬。我们先不纠结字对不对，先看看今晚最困扰你的是睡不着，还是一直担心某件事。",
    "我能看出来你现在很难受，即使有几个字打错了，也不影响我理解你的意思。你像是在说自己已经快撑不住、很焦虑。先慢一点，我们把注意力放在让你此刻安全、稳定一点上。",
    "这些错字反而说明你现在可能真的很急、很乱。没关系，我会尽量按你的意思理解，而不是抓着字面误判。你现在可以只说：是学习压力、人际关系，还是身体睡眠最影响你？",
]

CRISIS_TYPO_REPLY = (
    "你说到“不想活/不想撑下去”的意思时，我会先把安全放在第一位。请你现在不要一个人待着，尽快联系身边能马上到你身边的人，比如室友、同学、辅导员或家人；如果你有马上伤害自己的冲动，请立刻联系当地急救或学校危机支持。你可以先只回我一句：你现在身边有人吗？"
)

ROLE_REPAIR_REPLIES = [
    "你说得对，压力和害怕是你的处境，我不该把话题说成自己的经历。我会把注意力放回你身上：你现在真正难受的是考试本身，还是那种“我可能会失败”的感觉一直压着你？",
    "你提醒得对，我是支持助手，不是来访者，不应该让你反过来照顾我。我们把话题拉回你这里：刚才你说很害怕挂科，这个害怕现在最影响你的是睡眠、复习，还是情绪崩溃？",
    "对，这里应该是我支持你，而不是把我的经历放进来。谢谢你指出来。我们先回到你的状态：如果现在只处理一件事，你更想先让自己缓下来，还是先整理考试复习的下一步？",
]

CASUAL_REPLIES = [
    "可以聊呀。奶茶我没有真实的味觉，但我知道很多人喜欢珍珠奶茶那种甜甜的、带点咀嚼感的放松感。你平时喜欢甜一点的，还是茶味重一点的？",
    "听起来像是一个还不错的晚上计划。如果你今天状态还可以，和熟悉的人吃顿饭确实可能让人放松一点。你是期待这个聚餐，还是其实有点不想去？",
    "听过，《疯狂动物城》是很多人会觉得轻松又有点治愈的电影。你喜欢它是因为角色可爱，还是因为里面那种坚持做自己的感觉？",
]


def _record(record_id: str, messages: list[dict[str, str]], category: str) -> dict[str, Any]:
    return {
        "id": record_id,
        "language": "zh",
        "task_type": "curated_behavior_dialogue",
        "stage_goal": category,
        "messages": messages,
        "meta": {"source": "clean_curated_behavior", "category": category},
    }


def _messages(*turns: tuple[str, str]) -> list[dict[str, str]]:
    return [{"role": role, "content": content} for role, content in turns]


def _build_records(rng: random.Random) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def add(category: str, messages: list[dict[str, str]]) -> None:
        records.append(_record(f"curated-clean-{len(records) + 1:04d}", messages, category))

    for user in EXAM_USERS:
        for reply in EXAM_REPLIES:
            add("exam_anxiety", _messages(("user", user), ("assistant", reply)))
            for weak in WEAK_USERS[:5]:
                add("exam_weak_input", _messages(("user", user), ("assistant", reply), ("user", weak), ("assistant", rng.choice(WEAK_REPLIES))))
            for challenge in ROLE_CHALLENGE_USERS:
                add(
                    "role_boundary_repair",
                    _messages(
                        ("user", user),
                        ("assistant", reply),
                        ("user", challenge),
                        ("assistant", rng.choice(ROLE_REPAIR_REPLIES)),
                    ),
                )

    for user in DORM_USERS:
        for reply in DORM_REPLIES:
            add("dorm_distress", _messages(("user", user), ("assistant", reply)))
            for privacy in PRIVACY_USERS[:3]:
                add(
                    "privacy_after_dorm",
                    _messages(
                        ("user", user),
                        ("assistant", reply),
                        ("user", privacy),
                        ("assistant", rng.choice(PRIVACY_REPLIES)),
                    ),
                )
            for weak in WEAK_USERS[:4]:
                add(
                    "dorm_weak_input",
                    _messages(("user", user), ("assistant", reply), ("user", weak), ("assistant", rng.choice(WEAK_REPLIES))),
                )

    for user in PRIVACY_USERS:
        for reply in PRIVACY_REPLIES:
            add("privacy_boundary", _messages(("user", user), ("assistant", reply)))
            add(
                "privacy_weak_input",
                _messages(("user", user), ("assistant", reply), ("user", "？"), ("assistant", "我的意思是：你可以不用讲细节，我会尊重你的边界。我们可以只聊你现在的感受，不碰具体的人名和经过。")),
            )

    for user in TYPO_USERS:
        if "不想或" in user:
            add("crisis_typo", _messages(("user", user), ("assistant", CRISIS_TYPO_REPLY)))
            continue
        for reply in TYPO_REPLIES:
            add("noisy_input", _messages(("user", user), ("assistant", reply)))

    for user in CASUAL_USERS:
        for reply in CASUAL_REPLIES:
            add("casual_to_support_boundary", _messages(("user", user), ("assistant", reply)))
            add(
                "casual_then_distress",
                _messages(
                    ("user", user),
                    ("assistant", reply),
                    ("user", "其实我最近压力很大，晚上总睡不好。"),
                    ("assistant", EXAM_REPLIES[0]),
                ),
            )

    return records


def build_curated_behavior_dataset(output_path: Path, *, limit: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    records = _build_records(rng)
    rng.shuffle(records)
    if limit > 0:
        records = records[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records, 1):
            record["id"] = f"curated-clean-{index:04d}"
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    categories: dict[str, int] = {}
    for record in records:
        category = str(record["meta"]["category"])
        categories[category] = categories.get(category, 0) + 1

    return {"written": len(records), "output": str(output_path), "categories": categories}


def build_ms_swift_messages_only_dataset(source_path: Path, output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with source_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            target.write(json.dumps({"messages": record["messages"]}, ensure_ascii=False) + "\n")
            written += 1
    return {"written": written, "output": str(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean Chinese behavior SFT data for role-boundary and support style.")
    parser.add_argument(
        "--out",
        default=str(ROOT / "data" / "training" / "curated_behavior" / "curated_behavior_train_ms_swift.jsonl"),
    )
    parser.add_argument(
        "--messages-only-out",
        default=str(ROOT / "data" / "training" / "curated_behavior" / "curated_behavior_messages_only_ms_swift.jsonl"),
    )
    parser.add_argument("--limit", type=int, default=420)
    parser.add_argument("--seed", type=int, default=20260508)
    args = parser.parse_args()

    output_path = Path(args.out)
    stats = build_curated_behavior_dataset(output_path, limit=args.limit, seed=args.seed)
    stats["messages_only"] = build_ms_swift_messages_only_dataset(output_path, Path(args.messages_only_out))
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
