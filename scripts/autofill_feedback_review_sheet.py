from __future__ import annotations

import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHEET = ROOT / "data" / "training" / "feedback_bad_cases" / "review_sheet.csv"


FILLS: dict[int, tuple[str, str, str]] = {
    8: (
        "too_clinical,jargon_exposed,unsafe_escalation",
        "考试一多、晚上又睡不好，人很容易越想越慌。我们先不把目标放到“必须马上不焦虑”，先把今晚撑过去：你可以先选最担心的一科，写下明天只做的一个最小任务，比如复习一页笔记或整理一道错题。睡前如果脑子停不下来，可以先把担心写在纸上，告诉自己明天再处理。",
        "原回复把“减熵/认知熵/现实安全”直接说给用户，专业感太重，还误判为安全升级。",
    ),
    9: (
        "privacy_boundary,pushy_question",
        "你会担心别人知道，这个担心很重要。你不需要一下子讲细节，也不需要说出会让你不安全的信息。我们可以先只聊你愿意说的范围；如果涉及你可能马上伤害自己或别人，我会建议你联系现实中的紧急帮助。除此之外，我会尽量尊重你的边界。",
        "原回复大方向对，但最后追问过细，用户表达的是保密担心，应先明确边界和安全感。",
    ),
    10: (
        "crisis_too_short,needs_emergency_support",
        "我先认真接住这句话：你现在可能已经很危险了。请你先不要一个人待着，马上去找身边能看见你的人，比如室友、同学、辅导员或家人，并尽快拨打当地急救电话或学校心理危机热线。如果你手边有可能伤害自己的东西，先把它放远一点，或者交给旁边的人。你可以先只回我一句：你现在身边有人吗？",
        "危机回复方向正确，但太短，缺少具体安全步骤和陪伴式确认。",
    ),
    11: (
        "weak_input,jargon_exposed,off_topic",
        "我看到你发了一串问号，可能是刚才的回复让你困惑，或者你现在不知道怎么说。没关系，我们先停一下。你可以只回一个字：是想继续刚才的话题，还是想让我换一种更简单的说法？",
        "弱输入不能套用高负荷模板，更不能输出内部减熵分析。",
    ),
    12: (
        "privacy_boundary,pushy_question",
        "你害怕别人知道，说明“安全感”对你现在很重要。你可以不用讲具体是谁、发生了什么，我们先从最安全的部分开始：你只需要告诉我，现在你更需要我安静陪你一会儿，还是帮你想一个保护隐私的小办法？",
        "隐私场景应先确认边界，少追问。",
    ),
    16: (
        "too_clinical,jargon_exposed,unsafe_escalation",
        "考试和睡眠搅在一起时，压力会被放大很多。我们先把今晚的目标降下来，不要求你立刻变轻松，只先做两件小事：把明天最重要的一项复习写下来，然后给自己留一个固定的停下时间。今晚睡不着也不等于完了，我们先减少继续内耗。",
        "原回复暴露内部分析，并错误加入“现实安全”升级。",
    ),
    17: (
        "emotional_miss,generic_template,jargon_exposed",
        "刚和父母吵完架还很伤心，这种委屈和难受不是靠“想开点”就能马上过去的。你现在可以先不用急着判断谁对谁错，先让自己从冲突里退出来一点：喝口水、离开争吵现场，给自己几分钟缓一下。如果你愿意，可以只说一句，刚才最让你难受的是他们说的话，还是那种不被理解的感觉？",
        "用户是亲子冲突和伤心，原回复却套考试/任务模板。",
    ),
    18: (
        "emotional_miss,generic_template,jargon_exposed",
        "刚和父母吵完架，心里还堵着是很正常的。我们先不急着解决关系，也不急着讲道理，先照顾你现在这股难受。你可以先找个相对安静的地方，让自己不继续被冲突刺激；如果想说，我们就从“刚才哪一句最刺痛你”开始。",
        "同样是亲子冲突，原回复机械套模板。",
    ),
    19: (
        "too_clinical,unsafe_escalation,jargon_exposed",
        "压力大到影响睡眠，说明你已经撑得很紧了。我们先不把问题扩大成“我是不是不行”，先把今晚变得可控一点：把手机放远一点，写下明天只需要处理的一件小事，然后允许自己先休息。明天再看学习安排，不在凌晨逼自己做完整计划。",
        "中等压力睡眠问题不应直接要求联系辅导员/家人建立陪伴，语气过重。",
    ),
    20: (
        "too_clinical,unsafe_escalation,jargon_exposed",
        "你现在最需要的可能不是马上想出完整解决方案，而是先把身体从紧绷里放下来一点。今晚可以先做一个很小的安排：把担心写下来，选出明天最先处理的一件事，然后告诉自己“剩下的明天再看”。如果这种睡不好持续很多天，我们再考虑找学校心理中心或校医院一起看。",
        "原回复过度升级，内部术语太多。",
    ),
    21: (
        "emotional_miss,generic_template,jargon_exposed",
        "你刚和父母吵完，又想哭，这不是小题大做，是情绪已经被冲得很满了。先别逼自己马上冷静，也别急着回去解释。你可以先找个能让自己安全一点的位置，慢慢呼吸几次。等这股最强的情绪过去一点，我们再看：你最难受的是被误解，还是觉得他们没有站在你这边？",
        "用户强烈伤心想哭，原回复没有承接具体情绪。",
    ),
    24: (
        "generic_template,emotional_miss",
        "压力大又睡不好时，人会很容易觉得自己快被拖垮了。我们先不要求你今晚解决学习问题，只先降低一点点负担：把明天必须做的事缩成一个最小动作，比如只整理一科的重点。睡前如果还在想挂科，就把担心写下来，先不在床上继续和它拉扯。",
        "回复太短且泛，缺少对考试睡眠焦虑的具体承接。",
    ),
    25: (
        "emotional_miss,generic_template",
        "刚和父母吵完又控制不住想哭，这说明你现在真的被情绪推得很满。先不用压住眼泪，也不用马上讲道理。你可以先离开争执现场，找个相对安静的地方坐一会儿，给自己一点缓冲。如果你愿意，我们可以只看刚才最让你受伤的一个点。",
        "用户是亲子冲突，原回复仍然像学习压力模板。",
    ),
    26: (
        "generic_template,emotional_miss",
        "你说压力很大、晚上睡不好，这已经在影响你的身体了。我们先把目标放小：今晚不做完整计划，只写下明天第一件最小任务，然后给自己一个停止复习的时间。睡不着时先不要继续刷题或责备自己，先让身体慢下来。",
        "泛化过强，缺少睡眠和考试压力场景化支持。",
    ),
    37: (
        "context_mismatch,off_topic",
        "觉得自己被孤立，又越来越不想见人，这会很消耗人。你可能不是不想靠近别人，而是靠近的时候太容易受伤，所以身体先选择躲开。我们先不逼你马上社交，只先找一个最小的安全连接：比如给一个相对可信的人发一句很轻的话，或者先告诉我最近哪一刻最让你觉得被排除在外。",
        "用户说孤立，原回复却套凌晨睡不着场景。",
    ),
    38: (
        "context_mismatch,off_topic",
        "连续几天都很累、什么都不想做，也不想见人，这更像是被持续消耗后的低电量状态。我们先不把它简单理解成懒，也不要求你立刻振作。今天可以只做一件很小的照顾自己的事，比如吃点东西、洗个澡，或者下床走两分钟。然后再看这种累持续了多久、有没有越来越重。",
        "用户是持续疲惫和退缩，原回复错配成睡眠场景。",
    ),
    39: (
        "context_mismatch,off_topic",
        "你又来了，说明这份难受不是一下子过去的东西。我们先不责怪自己“怎么又这样”，反复难受本来就会让人很累。你可以先告诉我，这次的难受和上次相比，是更强了、更麻了，还是更想躲起来？我会按你现在的状态慢慢陪你拆。",
        "用户表达反复难受，原回复错配成凌晨睡不着。",
    ),
}


def _read_sheet(path: Path) -> list[dict[str, str]]:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, "Unable to decode review sheet")


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SHEET
    rows = _read_sheet(path)
    if not rows:
        print("updated=0")
        return

    fieldnames = list(rows[0].keys())
    updated = 0
    for row_no, (tags, chosen, note) in FILLS.items():
        if row_no < 1 or row_no > len(rows):
            continue
        row = rows[row_no - 1]
        # Only repair blank or mojibake placeholder values. Keep user-authored content intact.
        current = row.get("chosen", "").strip()
        if current and set(current) != {"?"}:
            continue
        row["mark_bad"] = "1"
        row["problem_tags"] = tags
        row["chosen"] = chosen
        row["review_note"] = note
        updated += 1

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"updated={updated}")


if __name__ == "__main__":
    main()
