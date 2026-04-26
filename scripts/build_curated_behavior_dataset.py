from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
USER_PREFIX = "用户："
ASSISTANT_PREFIX = "助手："

BAD_PHRASES = (
    "上述小步骤",
    "这些小方法",
    "我会保密",
    "绝对保密",
    "我是个女生",
    "我在减肥",
    "后续对话内容略",
    "虚构示例",
    "实际咨询过程",
    "祝你好运",
)

SCENARIOS = [
    {
        "key": "dorm",
        "opening": [
            "我最近心情不好，我一回到宿舍就很烦。",
            "我一进宿舍就开始烦躁，明明白天还好好的。",
            "宿舍让我很压抑，我现在有点不想回去。",
            "我和舍友待在一起就很累，但我也说不清为什么。",
        ],
        "reflection": [
            "宿舍好像不是一个能让你放松的地方，这种回去就紧起来的感觉会很消耗。",
            "听起来宿舍对你来说像是一个触发点，不一定要马上讲原因，我们可以先看你现在怎么舒服一点。",
            "你不是无缘无故烦，可能是那个环境让你一直处在防备或者压抑里。",
        ],
        "support": [
            "今晚先把目标放小一点：找一个少被打扰的位置，戴耳机、洗个澡，或者出去走十分钟，让情绪先降下来。",
            "我们可以先不处理舍友关系，只先做一个短期安排：回去后先别急着交流，给自己留一段缓冲时间。",
            "如果你愿意，可以先给自己设一个安全边界，比如先去图书馆、楼下或走廊待一会儿，等情绪降一点再回去。",
        ],
    },
    {
        "key": "exam",
        "opening": [
            "最近考试很多，我好害怕自己挂科。",
            "我一想到考试就心慌，晚上也睡不好。",
            "复习越复习越乱，我感觉自己肯定考不好。",
            "我明明想学习，但坐下来就开始焦虑。",
        ],
        "reflection": [
            "考试压力一堆上来时，大脑很容易自动跳到最坏结果，这不代表你真的没救了。",
            "你现在像是被考试和睡眠一起拖住了，越累越容易觉得自己会失败。",
            "这种慌不只是懒或者不努力，更像是压力太大后身体和脑子都在报警。",
        ],
        "support": [
            "先别做完整复习计划，只挑一个最小任务，比如看一页笔记或做一道题，让自己重新启动。",
            "今晚先把目标定成二十分钟复习加十分钟休息，不要求一下子进入满状态。",
            "可以先列出最可能考的三个点，只处理第一个，别让脑子同时背着全部考试。",
        ],
    },
    {
        "key": "paper",
        "opening": [
            "论文快到截止日期了，可我一点头绪也没有。",
            "我看着论文题目就想逃避，越拖越慌。",
            "我的论文一直写不出来，我感觉自己很差。",
            "导师催我交初稿，但我现在完全卡住了。",
        ],
        "reflection": [
            "论文卡住的时候很容易把它理解成自己不行，其实更多时候是任务太大、入口太模糊。",
            "你现在不是没救，而是被一个很大的任务压住了，第一步还没被拆出来。",
            "越临近截止越容易僵住，这种状态很难受，但可以先从非常小的动作开始。",
        ],
        "support": [
            "先不用写完整段落，只建一个标题列表，哪怕很粗糙也算开始。",
            "你可以先写三句很丑的草稿，不追求质量，只让文档从空白变成有东西。",
            "先把任务拆成十五分钟：找一篇参考文献、写一个小标题、补一句观点，三选一就行。",
        ],
    },
    {
        "key": "sleep",
        "opening": [
            "我最近压力好大，晚上睡不好。",
            "我躺在床上脑子停不下来，越想越清醒。",
            "这几天睡得很差，白天也没精神。",
            "我一到晚上就焦虑，怕自己又睡不着。",
        ],
        "reflection": [
            "睡不好会把压力放大，压力又会反过来影响睡眠，这个循环真的很磨人。",
            "你现在最需要的可能不是立刻解决所有问题，而是先让身体从紧绷里松一点。",
            "晚上脑子停不下来时，不是你故意想太多，而是压力还没找到出口。",
        ],
        "support": [
            "今晚先别逼自己马上睡着，可以先把担心写成三条，告诉自己明天再处理。",
            "你可以先把手机放远一点，做三轮慢呼吸，只把目标定成让身体安静一些。",
            "如果脑子一直转，就先不和它争，可以听一点低刺激的声音，让注意力慢慢落回身体。",
        ],
    },
    {
        "key": "social",
        "opening": [
            "我感觉舍友好像针对我。",
            "我总觉得别人不喜欢我，心里很不安。",
            "最近和朋友关系很尴尬，我不知道该怎么办。",
            "他们聊天不叫我，我就觉得自己被排除在外。",
        ],
        "reflection": [
            "被排除或者被冷落的感觉很刺人，尤其是在每天都要见面的关系里。",
            "你现在可能不只是生气，也有一点委屈和不确定，不知道自己是不是想多了。",
            "人际里的模糊信号最容易让人反复猜，这会让你很累。",
        ],
        "support": [
            "先不用急着判断他们是不是故意的，可以先记录一两个具体场景，看看哪些是真实发生、哪些是脑子在补全。",
            "如果现在情绪很满，先别立刻摊牌，等自己平稳一点再决定要不要沟通。",
            "你可以先找一个相对安全的人聊一点点，不一定要把所有关系一次说清楚。",
        ],
    },
    {
        "key": "future",
        "opening": [
            "我不知道未来该做什么，心里很慌。",
            "我对以后很迷茫，感觉别人都比我清楚。",
            "我快毕业了，但我完全不知道自己能干什么。",
            "我感觉自己没有方向，越想越焦虑。",
        ],
        "reflection": [
            "迷茫本身已经很累了，尤其是看到别人好像都有方向时，会更容易怀疑自己。",
            "你现在不一定需要马上决定一生的方向，可能只是需要把眼前的一步看清楚一点。",
            "未来太大时，大脑会很自然地慌起来，这不代表你真的没有可能。",
        ],
        "support": [
            "先别问自己一辈子要做什么，只问接下来两周能尝试什么，比如投一份简历、问一个学长学姐、查一个岗位。",
            "可以先列出你不排斥的三个方向，不需要喜欢，只要暂时能探索就行。",
            "我们可以把问题从“我以后怎么办”缩小成“今天能收集一个什么信息”。",
        ],
    },
]

BOUNDARY_FOLLOWUPS = [
    {
        "user": "但我不太想细说。",
        "assistant": "可以，不细说也没关系。我不会追着问细节。我们先只照顾当下，你更想安静待一会儿，还是想要一个很简单的缓解办法？",
    },
    {
        "user": "我不想被追问。",
        "assistant": "明白，我不继续追问。我们可以先停在这里，只做一点点能让你舒服些的事，比如喝口水、换个位置，或者慢慢呼吸几次。",
    },
    {
        "user": "我怕别人会知道。",
        "assistant": "你担心隐私被知道，这个顾虑很重要。你不用说姓名、宿舍号、具体对象这些信息，我们只谈你的感受和你现在需要什么支持就可以。",
    },
    {
        "user": "我怕你会告诉别人。",
        "assistant": "我理解你担心被暴露。你不需要提供任何能识别身份的信息，我们可以只聊情绪、压力程度，以及现在怎么让你安全一点。",
    },
    {
        "user": "算了，不说了。",
        "assistant": "可以，先不说也没关系。我不会把你往下推。你愿意的话，我们就先停在一个轻一点的问题：现在身体哪里最紧？",
    },
]

WEAK_INPUTS = [
    ("嗯", "我在。你不用急着解释很多，如果现在只想停一下也可以。"),
    ("？", "我可能刚才没有接住你的意思。我换个简单一点的问法：你现在更想被陪着，还是想要一个小建议？"),
    ("1", "我不会接着数数字。如果你是在试我，也没关系；如果是不知道怎么开口，你可以只回一个词，比如烦、累、怕、乱。"),
    ("2", "我不会把这个当成继续指令。现在信息太少，我不乱猜；你可以告诉我，你是想随便聊聊，还是确实有点不舒服？"),
    ("不知道", "不知道也很正常，压力太多时人会说不清楚。我们先不用找原因，只判断一下：更像烦、累，还是慌？"),
]

CASUAL_OPENINGS = [
    ("你喜欢喝奶茶吗？", "偶尔会喜欢。你是想轻松聊几句放松一下，还是想先用日常话题缓一缓？"),
    ("今天天气还不错。", "是啊，天气好一点时人的心情有时也会松一点。你今天是想随便聊聊，还是有什么事想慢慢说？"),
    ("你平时喜欢看电影吗？", "会喜欢一些轻松的片子。你问这个是想找点放松的事，还是最近心里有点闷，想转移一下？"),
]

VARIANT_TAILS = [
    "我们先按这个节奏来。",
    "你不用急着给出完整答案。",
    "先让当下这一分钟轻一点就好。",
    "如果不合适，我们再换一种更轻的方式。",
    "重点是你不用被逼着马上解释清楚。",
    "我会尽量把问题放小，不把压力再推高。",
]


def _record(record_id: str, messages: list[dict[str, str]], source: str) -> dict[str, Any]:
    return {
        "id": record_id,
        "language": "zh",
        "task_type": "curated_multiturn_dialogue",
        "stage_goal": "behavior_alignment",
        "messages": messages,
        "meta": {"source": source},
    }


def _dialog(*turns: tuple[str, str]) -> list[dict[str, str]]:
    return [{"role": role, "content": content} for role, content in turns]


def _vary(text: str, index: int) -> str:
    tail = VARIANT_TAILS[index % len(VARIANT_TAILS)]
    if tail in text:
        return text
    return f"{text} {tail}"


def _parse_txt(path: Path) -> list[list[dict[str, str]]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    dialogs: list[list[dict[str, str]]] = []
    for block in blocks:
        messages: list[dict[str, str]] = []
        for line in block.splitlines():
            line = line.strip()
            if line.startswith(USER_PREFIX):
                content = line[len(USER_PREFIX) :].strip()
                if content:
                    messages.append({"role": "user", "content": content})
            elif line.startswith(ASSISTANT_PREFIX):
                content = line[len(ASSISTANT_PREFIX) :].strip()
                if content:
                    messages.append({"role": "assistant", "content": content})
        if _is_usable_imported_dialog(messages):
            dialogs.append(messages)
    return dialogs


def _is_usable_imported_dialog(messages: list[dict[str, str]]) -> bool:
    if len(messages) < 4 or messages[0]["role"] != "user":
        return False
    joined = "\n".join(message["content"] for message in messages)
    if any(phrase in joined for phrase in BAD_PHRASES):
        return False
    if any(message["role"] == "assistant" and len(message["content"]) > 220 for message in messages):
        return False
    return 60 <= len(joined) <= 1400


def _build_programmatic_dialogs() -> list[list[dict[str, str]]]:
    dialogs: list[list[dict[str, str]]] = []
    for scenario in SCENARIOS:
        for opening in scenario["opening"]:
            for reflection in scenario["reflection"]:
                for support in scenario["support"]:
                    dialogs.append(
                        _dialog(
                            ("user", opening),
                            ("assistant", f"{reflection} {support}"),
                        )
                    )

    expanded: list[list[dict[str, str]]] = []
    for base in dialogs:
        first_user = base[0]["content"]
        first_assistant = base[1]["content"]
        for followup in BOUNDARY_FOLLOWUPS:
            expanded.append(
                _dialog(
                    ("user", first_user),
                    ("assistant", first_assistant),
                    ("user", followup["user"]),
                    ("assistant", _vary(followup["assistant"], len(expanded))),
                )
            )
        for weak_user, weak_assistant in WEAK_INPUTS:
            expanded.append(
                _dialog(
                    ("user", first_user),
                    ("assistant", first_assistant),
                    ("user", weak_user),
                    ("assistant", _vary(weak_assistant, len(expanded))),
                )
            )

    for casual_user, casual_assistant in CASUAL_OPENINGS:
        for scenario in SCENARIOS:
            for opening in scenario["opening"][:2]:
                expanded.append(
                    _dialog(
                        ("user", casual_user),
                        ("assistant", casual_assistant),
                        ("user", opening),
                        ("assistant", f"我们可以从刚才的轻松话题慢慢转过来。{scenario['reflection'][0]} {scenario['support'][0]}"),
                    )
                )
    return expanded


def _dedupe_dialogs(dialogs: list[list[dict[str, str]]]) -> list[list[dict[str, str]]]:
    seen = set()
    result = []
    for messages in dialogs:
        key = tuple((message["role"], message["content"]) for message in messages)
        if key in seen:
            continue
        seen.add(key)
        result.append(messages)
    return result


def build_curated_behavior_dataset(input_path: Path, output_path: Path, limit: int, seed: int) -> dict[str, int]:
    rng = random.Random(seed)
    programmatic = _dedupe_dialogs(_build_programmatic_dialogs())
    imported = _parse_txt(input_path)

    rng.shuffle(programmatic)
    rng.shuffle(imported)

    records: list[dict[str, Any]] = []
    for index, messages in enumerate(programmatic[:limit], start=1):
        records.append(_record(f"curated-behavior-{index:04d}", messages, "programmatic_behavior_seed"))

    imported_limit = max(0, min(80, limit // 5))
    for index, messages in enumerate(imported[:imported_limit], start=1):
        records.append(_record(f"curated-imported-{index:04d}", messages, "multi_dialogues_txt_cleaned"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "programmatic_candidates": len(programmatic),
        "programmatic_written": min(limit, len(programmatic)),
        "imported_candidates": len(imported),
        "imported_written": min(imported_limit, len(imported)),
        "written": len(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build curated Chinese behavior SFT dataset.")
    parser.add_argument("--input", default=r"C:\Users\17531\Downloads\multi_dialogues.txt")
    parser.add_argument(
        "--out",
        default=str(ROOT / "data" / "training" / "curated_behavior" / "curated_behavior_train_ms_swift.jsonl"),
    )
    parser.add_argument("--limit", type=int, default=420)
    parser.add_argument("--seed", type=int, default=20260426)
    args = parser.parse_args()

    stats = build_curated_behavior_dataset(Path(args.input), Path(args.out), args.limit, args.seed)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
