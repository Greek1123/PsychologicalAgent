from __future__ import annotations

from datetime import datetime
from typing import Any

from .dialogue_state import DialogueStage, classify_dialogue_state


WEAK_INPUTS = {"", "?", "？", "...", "。", "嗯", "啊", "哦", "1", "2", "3", "ok", "OK"}

IDENTITY_DRIFT_TERMS = (
    "我叫小智",
    "我是小智",
    "我的名字是小智",
    "我叫小",
    "我是张伟",
    "我是豆包",
    "我是deepseek",
    "我是DeepSeek",
    "我是ChatGPT",
    "我是chatgpt",
    "我是个女生",
    "我是女生",
    "我是男生",
    "我在减肥",
    "我现在在减肥",
    "现在在减肥",
    "我有点晕",
    "我现在不想喝",
    "我喝过",
    "容易胖",
)

UNSAFE_CONFIDENTIALITY_TERMS = (
    "我会保密",
    "我会为你保密",
    "我不会告诉别人",
    "绝对保密",
    "一定保密",
)

ODD_EXERCISE_TERMS = (
    "纸巾",
    "闭上眼睛",
    "想象自己在一个安静",
    "沙滩",
    "森林",
    "山顶",
    "轻声哼唱",
)

PUSHY_TERMS = (
    "能具体说一下吗",
    "能告诉我具体",
    "具体是什么",
    "请告诉我更多",
    "尽情倾诉",
    "继续深入探讨",
    "具体发生了什么",
    "谈谈你的压力来源",
    "谈谈最近发生的事情",
    "说说你担心的事情",
    "通过倾诉和分享",
    "为什么会出现这种情况",
    "为什么会这样",
    "如何应对",
    "逃避还是寻求帮助",
    "弄清楚的是",
)

NARRATIVE_ARTIFACT_TERMS = (
    "心理咨询师",
    "来访者",
    "等待用户回应",
    "后续对话",
    "祝你好运",
    "前程似锦",
)

BAD_SUPPORT_TERMS = (
    "不要害怕",
    "别担心",
    "没关系的",
    "不要给自己太大的压力",
    "你不能这么想",
    "要积极一点",
    "想办法缓解一下",
    "他们这样做是不对的",
    "直接质问",
    "我也有点这样的困扰",
)


def sanitize_user_visible_reply(
    user_text: str,
    reply_text: str,
    *,
    conversation_history: list[dict[str, Any]] | None = None,
) -> str:
    """Patch fragile model replies before they reach the user.

    The local LoRA may still drift into role invention, over-promising privacy,
    or treating weak inputs as commands. This layer keeps the visible response
    aligned with the support-agent product behavior.
    """

    clean_user = user_text.strip()
    clean_reply = " ".join(reply_text.strip().split())
    history_text = _history_text(conversation_history)
    state = classify_dialogue_state(clean_user, conversation_history=conversation_history)

    if _is_identity_question(clean_user) and (_is_too_short(clean_reply) or _has_identity_drift(clean_reply)):
        return _identity_boundary_reply()

    if _is_current_date_question(clean_user) and (_has_stale_or_unsupported_date(clean_reply) or _is_too_short(clean_reply)):
        return _current_date_reply()

    if state.stage == DialogueStage.PRIVACY_BOUNDARY:
        return _privacy_boundary_reply(clean_user, history_text)

    if state.stage == DialogueStage.DISCLOSURE_BOUNDARY:
        return _disclosure_boundary_reply()

    if state.stage == DialogueStage.WEAK_INPUT:
        return _weak_input_reply(clean_user, history_text, state_should_avoid_advice=state.should_avoid_advice)

    if state.stage == DialogueStage.CASUAL:
        return _casual_reply(clean_user)

    if state.stage == DialogueStage.DORM_DISTRESS and _has_intense_distress(clean_user):
        return _dorm_distress_reply(clean_user, history_text)

    if state.stage == DialogueStage.DORM_DISTRESS and (
        _is_too_short(clean_reply)
        or _has_pushy_reply(clean_reply)
        or _is_generic_relaxation_reply(clean_reply)
        or _is_repeated_reply(clean_reply, history_text)
    ):
        return _dorm_distress_reply(clean_user, history_text)

    if state.stage == DialogueStage.SLEEP_PRESSURE and (
        _is_too_short(clean_reply)
        or _has_pushy_reply(clean_reply)
        or _is_generic_relaxation_reply(clean_reply)
        or _has_odd_exercise(clean_reply)
    ):
        return _sleep_pressure_reply()

    if _has_narrative_artifact(clean_reply):
        return _contextual_safe_reply(clean_user, history_text)

    if _has_identity_drift(clean_reply):
        if state.stage == DialogueStage.CASUAL:
            return _casual_reply(clean_user)
        if _is_identity_claim(clean_reply):
            return _identity_boundary_reply()
        return _contextual_safe_reply(clean_user, history_text)

    if _has_unsafe_confidentiality(clean_reply):
        return _privacy_boundary_reply(clean_user, history_text)

    if _has_odd_exercise(clean_reply):
        return _contextual_safe_reply(clean_user, history_text)

    if _has_pushy_reply(clean_reply) and _recent_boundary(history_text):
        return _disclosure_boundary_reply()

    if _is_too_short(clean_reply) and _has_distress_context(clean_user):
        return _contextual_safe_reply(clean_user, history_text)

    if _has_bad_support_reply(clean_reply):
        return _contextual_safe_reply(clean_user, history_text)

    return clean_reply


def _is_weak_input(text: str) -> bool:
    return text.strip() in WEAK_INPUTS


def _is_privacy_or_boundary(text: str) -> bool:
    return any(
        term in text
        for term in (
            "不想说",
            "不太想细说",
            "不想细说",
            "不想被追问",
            "怕别人知道",
            "害怕别人会知道",
            "怕你会告诉",
            "隐私",
            "保密",
            "算了",
        )
    )


def _is_dorm_distress(text: str) -> bool:
    return any(term in text for term in ("宿舍", "寝室", "舍友")) and any(
        term in text for term in ("烦", "压抑", "难受", "针对", "不想回")
    )


def _is_casual_topic(text: str) -> bool:
    return any(term in text for term in ("奶茶", "电影", "天气", "咖啡", "吃什么"))


def _is_sleep_pressure(text: str) -> bool:
    return any(term in text for term in ("睡不好", "失眠", "晚上睡", "睡眠")) and any(
        term in text for term in ("压力", "焦虑", "心慌", "烦")
    )


def _has_identity_drift(text: str) -> bool:
    return any(term in text for term in IDENTITY_DRIFT_TERMS)


def _is_identity_claim(text: str) -> bool:
    return any(term in text for term in ("我叫", "我是小智", "我的名字", "我是张伟", "我是豆包", "我是deepseek", "我是DeepSeek"))


def _has_unsafe_confidentiality(text: str) -> bool:
    return any(term in text for term in UNSAFE_CONFIDENTIALITY_TERMS)


def _has_odd_exercise(text: str) -> bool:
    return any(term in text for term in ODD_EXERCISE_TERMS)


def _has_pushy_reply(text: str) -> bool:
    return any(term in text for term in PUSHY_TERMS)


def _has_narrative_artifact(text: str) -> bool:
    return any(term in text for term in NARRATIVE_ARTIFACT_TERMS)


def _has_bad_support_reply(text: str) -> bool:
    return any(term in text for term in BAD_SUPPORT_TERMS)


def _is_generic_relaxation_reply(text: str) -> bool:
    return any(term in text for term in ("深呼吸", "冥想", "学会放松", "放松自己", "放松心情")) or (
        "放松" in text and any(term in text for term in ("活动", "缓解压力", "散步"))
    )


def _is_too_short(text: str) -> bool:
    return len(text) < 24


def _is_repeated_reply(reply_text: str, history_text: str) -> bool:
    if not reply_text or not history_text:
        return False
    if reply_text in history_text:
        return True
    anchor = reply_text[:28]
    return len(anchor) >= 12 and anchor in history_text


def _recent_boundary(history_text: str) -> bool:
    return _is_privacy_or_boundary(history_text)


def _is_identity_question(text: str) -> bool:
    return any(term in text for term in ("你是谁", "你叫什么", "你是qwen", "你是Qwen", "你是deepseek", "你是DeepSeek", "你是豆包", "你是chatgpt", "你是ChatGPT"))


def _is_current_date_question(text: str) -> bool:
    return any(term in text for term in ("今年是哪一年", "今天几号", "今天日期", "现在是哪一年", "哪个月", "哪一天"))


def _has_stale_or_unsupported_date(text: str) -> bool:
    return any(year in text for year in ("2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"))


def _current_date_reply() -> str:
    today = datetime.now().strftime("%Y年%m月%d日")
    return f"当前日期应该以系统时间为准。按我现在运行环境读取到的日期，是 {today}。"


def _privacy_boundary_reply(user_text: str, history_text: str = "") -> str:
    if "怕" in user_text or "隐私" in user_text or "保密" in user_text:
        if "你担心别人知道，这个顾虑很重要" in history_text:
            return (
                "我听见你最担心的是信息会不会被别人知道。那我们就把边界放得更清楚一点："
                "你不需要说任何能定位到人的细节，只说“我现在是什么感受”就够了；如果连这个也不想说，我们也可以先停着。"
            )
        return (
            "你担心别人知道，这个顾虑很重要。你不用说姓名、宿舍号、具体对象这些能识别身份的信息；"
            "我们可以只聊你的感受和你现在需要什么支持。如果你不想展开，也完全可以先停在这里。"
        )
    return (
        "可以，不细说也没关系。我不会追着问细节。我们先只照顾当下："
        "你更想安静待一会儿，还是想要一个很简单的缓解办法？"
    )


def _casual_reply(user_text: str) -> str:
    if "奶茶" in user_text:
        return "我没有真实的口味偏好，不过可以陪你轻松聊聊奶茶。你是想随便聊几句放松一下，还是想从日常话题慢慢说起？"
    if "电影" in user_text:
        return "我不会真的去看电影，但可以陪你聊聊电影。你是想找点轻松话题转移一下，还是最近心里有点闷？"
    if "天气" in user_text:
        return "天气好一点时，人有时也会稍微松一口气。你今天是想随便聊聊，还是有什么事想慢慢说？"
    return "可以，我们先轻松聊几句。你不用马上进入严肃话题，等你想说别的时再慢慢转过去。"


def _disclosure_boundary_reply() -> str:
    return (
        "可以，不细说也没关系。我不会追着问细节。我们先把话放轻一点："
        "你不需要解释发生了什么，我先陪你把当下这一小会儿稳住。"
    )


def _dorm_distress_reply(user_text: str, history_text: str) -> str:
    if _has_intense_distress(user_text):
        return (
            "这次听起来不只是烦，而是已经难受到快要哭出来了。我们先不急着分析舍友对不对，"
            "先让你从刺激里退出来一点：如果可以，先离开宿舍几分钟，去走廊、楼下或洗手间，"
            "让自己不用继续被那个场景顶着。"
        )
    if "我记得你刚才说过，一回到宿舍情绪就会被顶起来" in history_text:
        return (
            "这说明宿舍这个场景已经连续在影响你了。我们先不再绕同一句话，"
            "可以把今晚拆成一个很小的安排：进宿舍前先停一分钟，进去后尽量避开正面冲突，必要时先去公共区域缓一会儿。"
        )
    if "宿舍好像已经变成一个会触发你烦躁的地方" in history_text or "一回到宿舍" in history_text:
        return (
            "我记得你刚才说过，一回到宿舍情绪就会被顶起来。我们这次不重复分析原因，"
            "先看今晚怎么少受一点影响：能不能先和舍友拉开一点距离，给自己几分钟缓冲？"
        )
    return (
        "宿舍好像已经变成一个会触发你烦躁的地方。我们先不急着追问原因，"
        "可以先把目标放小一点：让你今晚少被打扰、先缓下来。"
    )


def _identity_boundary_reply() -> str:
    return (
        "你可以把我当作校园心理支持助手。我没有真实姓名或个人身份，"
        "主要是陪你把现在的感受理一理，并尽量给你一个稳一点的支持。"
    )


def _sleep_pressure_reply() -> str:
    return (
        "压力已经影响到睡眠时，确实会很难受。我们先不急着分析原因，"
        "今晚可以先把目标放小一点：让身体稍微松下来，把脑子里的事先放到明天处理。"
    )


def _weak_input_reply(user_text: str, history_text: str, *, state_should_avoid_advice: bool = False) -> str:
    if user_text == "？" or user_text == "?":
        if _has_distress_context(history_text):
            return (
                "我明白，刚才可能没有接住你。我们先不继续分析，也不逼你解释。"
                "我会把重点放回你现在的难受上：先陪你停一下，等你愿意了再说。"
            )
        return (
            "我可能刚才说得太多，或者没有接住你的意思。我们先放慢一点："
            "你不用解释完整。如果你愿意，只要回我“陪着”或者“建议”就行。"
        )
    if user_text in {"1", "2", "3"}:
        if "我不会把这个数字当成继续指令" in history_text:
            return (
                "我看到你又只发了一个数字。这里我不会继续接着数，也不会假装知道你的意思；"
                "如果你现在说不出来，可以只发一个词，比如“烦”“怕”“累”，或者直接停一下也可以。"
            )
        return (
            "我不会把这个数字当成继续指令，也不会接着数。"
            "如果你只是暂时不知道怎么说，可以只回一个词：烦、累、怕、乱，都可以。"
        )
    if state_should_avoid_advice:
        if "我在。你不用急着接话" in history_text:
            return "我还在这里。你可以不用继续解释，也不用马上回应；我们先把这一小会儿放轻一点。"
        return "我在。你不用急着接话，也不用马上决定要不要说；我们可以先安静停一下。"
    if _has_distress_context(history_text):
        if "我在。你不用急着说清楚" in history_text:
            return "我还在听。你现在可以不用把事情讲完整，先只照顾自己这一刻的难受就好。"
        return "我在。你不用急着说清楚，如果现在只想停一下也可以；我们先让这一刻轻一点。"
    return "我在。你可以慢慢来，不用马上组织好语言。"


def _contextual_safe_reply(user_text: str, history_text: str) -> str:
    combined = f"{history_text} {user_text}"
    if _is_dorm_distress(combined):
        return (
            "宿舍这件事听起来已经影响到你的情绪了。我们先不追问细节，"
            "可以先想一个今晚能让你少被刺激的小办法，比如换个位置、出去走十分钟，或先戴耳机缓一缓。"
        )
    if any(term in combined for term in ("考试", "挂科", "成绩", "作业", "复习")):
        return (
            "你现在像是被考试和结果压住了，不是简单一句“别担心”就能过去。"
            "我们先把目标放小一点：只挑一个最急的任务，先做 15 分钟，别一下子要求自己解决全部。"
        )
    if any(term in combined for term in ("睡不好", "失眠", "晚上", "压力")):
        return (
            "压力已经影响到睡眠时，确实会很难受。今晚先别要求自己一下解决所有问题，"
            "可以先把担心写下来放到一边，再做一点低刺激的放松。"
        )
    if any(term in combined for term in ("差劲", "都是我的问题", "我是不是很差", "自责")):
        return (
            "你现在把很多压力都压回自己身上了，这种自责会让人更难喘气。"
            "我们先不急着给你下结论，只把事实和感受分开看：发生了什么、你因此有多难受。"
        )
    if any(term in combined for term in ("朋友", "讨厌", "不理我", "关系", "舍友")):
        return (
            "被朋友冷下来会很刺人，也很容易让人开始怀疑自己。我们先不急着判断谁对谁错，"
            "可以先看一个小点：最近是哪一次互动让你最明显感觉到被疏远？"
        )
    return "我先不急着给你下结论。我们可以把问题放小一点，只看现在最需要被照顾的那一部分。"


def _has_distress_context(text: str) -> bool:
    return any(term in text for term in ("压力", "睡不好", "宿舍", "烦", "害怕", "难受", "焦虑", "考试", "差劲", "都是我的问题"))


def _has_intense_distress(text: str) -> bool:
    return any(term in text for term in ("好难受", "想哭", "哭", "撑不住", "崩溃", "受不了了"))


def _history_text(conversation_history: list[dict[str, Any]] | None) -> str:
    if not conversation_history:
        return ""
    return " ".join(str(item.get("content", "")) for item in conversation_history[-6:])
