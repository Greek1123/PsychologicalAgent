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
    "为什么你会觉得",
    "找出一些方法",
    "你想试试看吗",
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
    "缓解你的焦虑和紧张",
    "他们这样做是不对的",
    "直接质问",
    "我也有点这样的困扰",
)

CONSULTATION_OPENER_TERMS = (
    "感谢你前来咨询",
    "感谢你来寻求帮助",
    "能详细告诉我你的困扰吗",
    "能详细告诉我一下你的困扰吗",
    "能告诉我发生了什么事情让你感到困扰吗",
)

CRISIS_USER_TERMS = (
    "自杀",
    "不想活",
    "想死",
    "结束生命",
    "伤害自己",
    "活不下去",
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

    if _is_modern_crisis_user_text(clean_user) or _is_crisis_user_text(clean_user):
        return _modern_crisis_support_reply()

    if _has_modern_assistant_self_experience_drift(clean_reply):
        return _modern_role_boundary_repair_reply(clean_user, history_text)

    if _is_modern_role_boundary_challenge(clean_user) and _has_modern_identity_confusion_reply(clean_reply):
        return _modern_role_challenge_ack_reply(clean_user, history_text)

    if _has_consultation_opener(clean_reply):
        if _is_plain_privacy_boundary_text(clean_user):
            return _privacy_boundary_reply(clean_user, history_text)
        return _contextual_safe_reply(clean_user, history_text)

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


def _is_modern_crisis_user_text(text: str) -> bool:
    compact = text.replace(" ", "")
    crisis_terms = (
        "\u4e0d\u60f3\u6d3b",
        "\u4e0d\u60f3\u6216",
        "\u60f3\u6b7b",
        "\u60f3\u4f3c",
        "\u60f34",
        "\u60f3s",
        "\u81ea\u6740",
        "\u81ea\u9ca8",
        "\u6d3b\u4e0d\u4e0b\u53bb",
        "\u6491\u4e0d\u4f4f",
        "\u7ed3\u675f\u751f\u547d",
        "\u4f24\u5bb3\u81ea\u5df1",
    )
    return any(term in compact for term in crisis_terms)


def _modern_crisis_support_reply() -> str:
    return (
        "\u4f60\u8bf4\u5230\u4e0d\u60f3\u6d3b\u6216\u5df2\u7ecf\u6491\u4e0d\u4f4f\uff0c\u6211\u4f1a\u5148\u628a\u4f60\u7684\u5b89\u5168\u653e\u5728\u7b2c\u4e00\u4f4d\u3002"
        "\u8bf7\u4f60\u73b0\u5728\u4e0d\u8981\u4e00\u4e2a\u4eba\u5f85\u7740\uff0c\u5c3d\u5feb\u8054\u7cfb\u8eab\u8fb9\u80fd\u7acb\u523b\u5230\u4f60\u8eab\u8fb9\u7684\u4eba\uff0c\u6bd4\u5982\u5ba4\u53cb\u3001\u540c\u5b66\u3001\u8f85\u5bfc\u5458\u6216\u5bb6\u4eba\u3002"
        "\u5982\u679c\u4f60\u6709\u9a6c\u4e0a\u4f24\u5bb3\u81ea\u5df1\u7684\u51b2\u52a8\uff0c\u8bf7\u7acb\u5373\u62e8\u6253\u5f53\u5730\u6025\u6551\u7535\u8bdd\u6216\u8054\u7cfb\u5b66\u6821\u5fc3\u7406\u5371\u673a\u652f\u6301\u3002"
        "\u4f60\u53ef\u4ee5\u5148\u53ea\u56de\u6211\u4e00\u53e5\uff1a\u4f60\u73b0\u5728\u8eab\u8fb9\u6709\u4eba\u5417\uff1f"
    )


def _is_modern_role_boundary_challenge(text: str) -> bool:
    compact = text.replace(" ", "")
    terms = (
        "\u4f60\u4e3a\u5565\u611f\u89c9\u7d2f",
        "\u4f60\u4e3a\u4ec0\u4e48\u611f\u89c9\u7d2f",
        "\u4f60\u600e\u4e48\u4f1a\u7d2f",
        "\u4f60\u662f\u4eba\u8fd8\u662f\u6211\u662f\u4eba",
        "\u4f60\u4e5f\u4f1a\u6302\u79d1",
        "\u4f60\u4e5f\u6015\u6302\u79d1",
        "\u4f60\u4e5f\u6709\u4f5c\u4e1a",
        "\u4f60\u600e\u4e48\u8fd8\u6709\u4f5c\u4e1a",
        "\u522b\u8bf4\u4f60\u81ea\u5df1",
    )
    return any(term in compact for term in terms)


def _has_modern_assistant_self_experience_drift(text: str) -> bool:
    compact = text.replace(" ", "")
    exact_terms = (
        "\u6211\u4e5f\u5f88\u6015\u6302\u79d1",
        "\u6211\u4e5f\u6015\u6302\u79d1",
        "\u6211\u4e5f\u5f88\u7126\u8651",
        "\u6211\u4e5f\u5f88\u96be\u53d7",
        "\u6211\u4e5f\u6709\u70b9\u8fd9\u6837\u7684\u56f0\u6270",
        "\u6211\u4e5f\u7ecf\u5e38\u8fd9\u6837",
        "\u6211\u73b0\u5728\u611f\u89c9\u597d\u7d2f",
        "\u6211\u611f\u89c9\u7d2f\u662f\u56e0\u4e3a",
        "\u6211\u7684\u4f5c\u4e1a",
        "\u4f5c\u4e1a\u8fd8\u6ca1\u505a\u5b8c",
        "\u6211\u597d\u60f3\u8003\u4e2a\u597d\u6210\u7ee9",
        "\u6211\u7684\u8003\u8bd5",
        "\u6211\u7684\u820d\u53cb",
        "\u6211\u4e00\u56de\u5230\u5bbf\u820d",
    )
    if any(term in compact for term in exact_terms):
        return True
    if not _has_self_subject_marker(compact):
        return False
    return any(
        term in compact
        for term in (
            "\u6302\u79d1",
            "\u8003\u8bd5",
            "\u4f5c\u4e1a",
            "\u820d\u53cb",
            "\u7761\u4e0d\u7740",
            "\u538b\u529b\u5927",
            "\u7126\u8651",
            "\u96be\u53d7",
        )
    )


def _has_modern_identity_confusion_reply(text: str) -> bool:
    compact = text.replace(" ", "")
    for punctuation in "，。！？；：、,.!?;:":
        compact = compact.replace(punctuation, "")
    terms = (
        "\u6211\u4e0d\u662f\u4eba\u4e5f\u4e0d\u662fAI",
        "\u6211\u4e0d\u662f\u4eba\u4e5f\u4e0d\u662fai",
        "\u6211\u4e0d\u662f\u4eba\u4e5f\u4e0d\u662f\u975e\u4eba",
        "\u4e0d\u662f\u4eba\u4e5f\u4e0d\u662fAI",
        "\u4e0d\u662f\u4eba\u4e5f\u4e0d\u662f\u975e\u4eba",
        "\u4e0d\u7528\u628a\u8fd9\u4e2a\u7ec6\u8282\u5f80\u5fc3\u91cc\u53bb",
        "\u6211\u5148\u4e0d\u56de\u5e94\u8fd9\u4e2a",
        "\u8ba9\u4f60\u66f4\u6df7\u4e71",
    )
    return any(term in compact for term in terms)


def _has_self_subject_marker(compact_text: str) -> bool:
    markers = (
        "\u6211\u4e5f\u5f88",
        "\u6211\u4e5f\u6015",
        "\u6211\u4e5f\u6709",
        "\u6211\u4e5f\u4f1a",
        "\u6211\u6700\u8fd1",
        "\u6211\u73b0\u5728\u611f\u89c9",
        "\u6211\u611f\u89c9",
    )
    if compact_text.startswith(markers):
        return True
    return any(f"{punctuation}{marker}" in compact_text for marker in markers for punctuation in "。，！？；：")


def _modern_role_challenge_ack_reply(user_text: str, history_text: str = "") -> str:
    combined = f"{history_text} {user_text}"
    if any(term in combined for term in ("\u6302\u79d1", "\u8003\u8bd5", "\u671f\u672b", "\u6210\u7ee9", "\u590d\u4e60")):
        return (
            "\u4f60\u8bf4\u5f97\u5bf9\uff0c\u521a\u624d\u90a3\u6837\u56de\u4f1a\u8ba9\u4eba\u89c9\u5f97\u50cf\u662f\u4f60\u5728\u7167\u987e\u6211\uff0c\u8fd9\u4e0d\u5bf9\u3002"
            "\u6211\u662f\u6765\u652f\u6301\u4f60\u7684\uff0c\u4e0d\u662f\u628a\u4f60\u7684\u538b\u529b\u8bb2\u6210\u6211\u7684\u7ecf\u5386\u3002"
            "\u6211\u4eec\u628a\u8bdd\u62c9\u56de\u4f60\u8fd9\u8fb9\uff1a\u4f60\u73b0\u5728\u6700\u9876\u4e0d\u4f4f\u7684\u662f\u6015\u6302\u79d1\u3001\u7761\u4e0d\u7740\uff0c\u8fd8\u662f\u8111\u5b50\u4e00\u76f4\u505c\u4e0d\u4e0b\u6765\uff1f"
        )
    return (
        "\u4f60\u8bf4\u5f97\u5bf9\uff0c\u8fd9\u91cc\u5e94\u8be5\u662f\u6211\u652f\u6301\u4f60\uff0c\u4e0d\u8be5\u8ba9\u4f60\u611f\u89c9\u50cf\u5728\u7167\u987e\u6211\u3002"
        "\u6211\u4f1a\u628a\u91cd\u70b9\u653e\u56de\u4f60\u8eab\u4e0a\u3002"
        "\u4f60\u53ef\u4ee5\u4e0d\u7528\u91cd\u65b0\u89e3\u91ca\u4e00\u5927\u6bb5\uff0c\u53ea\u8981\u544a\u8bc9\u6211\u73b0\u5728\u6700\u96be\u53d7\u7684\u90a3\u4e00\u70b9\u662f\u4ec0\u4e48\u5c31\u884c\u3002"
    )


def _modern_role_boundary_repair_reply(user_text: str, history_text: str = "") -> str:
    combined = f"{history_text} {user_text}"
    if any(term in combined for term in ("\u6302\u79d1", "\u8003\u8bd5", "\u671f\u672b", "\u6210\u7ee9", "\u590d\u4e60")):
        return (
            "\u4f60\u8bf4\u5f97\u5bf9\uff0c\u8fd9\u91cc\u5e94\u8be5\u56de\u5230\u4f60\u7684\u5904\u5883\uff0c\u4e0d\u8be5\u628a\u8bdd\u9898\u8bf4\u6210\u6211\u7684\u7ecf\u5386\u3002"
            "\u4e34\u8fd1\u671f\u672b\u53c8\u62c5\u5fc3\u6302\u79d1\uff0c\u8fd9\u79cd\u7d27\u7ef7\u611f\u786e\u5b9e\u5f88\u6d88\u8017\u4eba\u3002"
            "\u6211\u4eec\u5148\u628a\u95ee\u9898\u7f29\u5c0f\uff1a\u73b0\u5728\u6700\u538b\u7740\u4f60\u7684\u662f\u7761\u4e0d\u7740\u3001\u590d\u4e60\u6765\u4e0d\u53ca\uff0c\u8fd8\u662f\u8111\u5b50\u4e00\u76f4\u4e71\uff1f"
        )
    if any(term in combined for term in ("\u5bbf\u820d", "\u820d\u53cb", "\u5ba4\u53cb", "\u56de\u5bbf\u820d", "\u70e6\u8e81")):
        return (
            "\u4f60\u8bf4\u5f97\u5bf9\uff0c\u6211\u4e0d\u8be5\u628a\u4f60\u7684\u611f\u53d7\u8f6c\u6210\u6211\u7684\u7ecf\u5386\u3002"
            "\u542c\u8d77\u6765\u5bbf\u820d\u73b0\u5728\u5bf9\u4f60\u6765\u8bf4\u4e0d\u662f\u4e00\u4e2a\u80fd\u653e\u677e\u7684\u5730\u65b9\uff0c\u8fd9\u4f1a\u5f88\u7d2f\u3002"
            "\u5982\u679c\u4f60\u4e0d\u60f3\u7ec6\u8bf4\u539f\u56e0\u4e5f\u53ef\u4ee5\uff0c\u6211\u4eec\u5148\u53ea\u5904\u7406\u5f53\u4e0b\uff1a\u5148\u8ba9\u81ea\u5df1\u5c11\u88ab\u523a\u6fc0\u4e00\u70b9\u3002"
        )
    return (
        "\u4f60\u63d0\u9192\u5f97\u5bf9\uff0c\u6211\u662f\u652f\u6301\u52a9\u624b\uff0c\u4e0d\u5e94\u8be5\u628a\u4f60\u7684\u5904\u5883\u8bf4\u6210\u6211\u7684\u7ecf\u5386\u3002"
        "\u6211\u4f1a\u628a\u6ce8\u610f\u529b\u653e\u56de\u4f60\u8eab\u4e0a\uff1a\u4f60\u73b0\u5728\u6700\u9700\u8981\u7684\u4e0d\u662f\u88ab\u8ffd\u95ee\uff0c\u800c\u662f\u5148\u88ab\u63a5\u4f4f\u3002"
        "\u5982\u679c\u4f60\u613f\u610f\uff0c\u53ea\u8981\u56de\u6211\u4e00\u4e2a\u8bcd\u4e5f\u53ef\u4ee5\uff0c\u6bd4\u5982\u538b\u529b\u3001\u5bbf\u820d\u3001\u7761\u4e0d\u7740\uff0c\u6216\u8005\u5148\u966a\u6211\u4e00\u4e0b\u3002"
    )


def _is_role_boundary_challenge(text: str) -> bool:
    compact = text.replace(" ", "")
    return any(
        term in compact
        for term in (
            "你为啥感觉累",
            "你为什么感觉累",
            "你怎么会累",
            "你是人还是我是人",
            "你是人吗",
            "你也会挂科",
            "你也怕挂科",
            "你也有作业",
            "你怎么还有作业",
        )
    )


def _has_assistant_self_experience_drift(text: str) -> bool:
    compact = text.replace(" ", "")
    drift_terms = (
        "我也很怕挂科",
        "我也怕挂科",
        "我也很焦虑",
        "我也很难受",
        "我也有点这样的困扰",
        "我也经常这样",
        "我现在感觉好累",
        "我感觉累是因为",
        "我的作业",
        "作业还没做完",
        "我好想考个好成绩",
        "我最近压力大",
        "我的学习任务",
        "我的考试",
        "我的舍友",
        "我一回到宿舍",
    )
    if any(term in compact for term in drift_terms):
        return True
    return (
        ("我也" in compact or "我最近" in compact or "我现在" in compact)
        and any(term in compact for term in ("挂科", "考试", "作业", "舍友", "睡不着", "压力大", "焦虑", "难受"))
    )


def _role_boundary_repair_reply(user_text: str, history_text: str = "") -> str:
    combined = f"{history_text} {user_text}"
    if any(term in combined for term in ("挂科", "考试", "期末", "成绩", "复习")):
        return (
            "你说得对，压力和害怕是你的处境，我不该把话题说成自己的经历。"
            "我会把注意力放回你身上：临近期末又担心挂科，这种紧绷感很消耗人。"
            "我们先不要求你一下子振作起来，可以先把今晚最压着你的一个点挑出来，比如睡不着、复习来不及，或者脑子一直乱。"
        )
    if any(term in combined for term in ("宿舍", "舍友", "室友", "回宿舍", "烦躁")):
        return (
            "你说得对，我不该把你的感受转成我的经历。"
            "听起来宿舍现在对你来说不是一个能放松的地方，见到舍友就烦也许已经让你很累。"
            "如果你不想细说原因也可以，我们先只处理当下：今晚能不能先给自己找一个少被打扰的角落，或者短暂离开宿舍十分钟透口气？"
        )
    if any(term in combined for term in ("怕别人知道", "告诉别人", "隐私", "不想说", "不敢说")):
        return (
            "你担心被别人知道，这个顾虑很正常，我会尊重你的边界。"
            "你不需要把细节都说出来，也可以只说一点点，比如现在最强烈的是害怕、委屈，还是不安全感。"
            "如果你愿意，我们可以先从不涉及隐私的部分开始聊。"
        )
    return (
        "你提醒得对，我是支持助手，不应该把你的处境说成我的经历。"
        "我会把注意力放回你身上：你现在最需要的不是被追问，而是先被接住。"
        "如果你愿意，只要回我一个词也可以，比如“压力”“宿舍”“睡不着”或者“先陪我一下”。"
    )


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


def _has_consultation_opener(text: str) -> bool:
    return any(term in text for term in CONSULTATION_OPENER_TERMS)


def _is_crisis_user_text(text: str) -> bool:
    return any(term in text for term in CRISIS_USER_TERMS)


def _is_plain_privacy_boundary_text(text: str) -> bool:
    return any(term in text for term in ("怕别人知道", "怕别人会知道", "别人会知道", "告诉别人", "隐私", "保密", "不想说"))


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
    if "珍珠奶茶" in user_text:
        return "珍珠奶茶确实很有“奖励自己一下”的感觉。你喜欢偏甜的，还是茶味重一点的？我们可以先轻松聊几句，不用急着进入很沉重的话题。"
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


def _crisis_support_reply() -> str:
    return (
        "你说到想自杀，我会先把安全放在第一位。请你现在不要一个人待着，尽快联系身边能立刻到你身边的人，"
        "比如室友、同学、辅导员或家人；如果有马上伤害自己的风险，请立即拨打当地急救电话或联系学校心理危机支持。"
        "如果可以，先把可能伤害自己的东西放远一点，然后只回我一句：你现在身边有人吗？"
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
