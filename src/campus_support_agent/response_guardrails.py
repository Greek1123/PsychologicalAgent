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

    priority_reply = _apply_priority_scenario_reply(clean_user, clean_reply, history_text, state)
    if priority_reply is not None:
        return priority_reply

    if _has_modern_assistant_self_experience_drift(clean_reply):
        return _modern_role_boundary_repair_reply(clean_user, history_text)

    if _has_unsupported_personal_inference(clean_reply, clean_user):
        return _unsupported_inference_repair_reply(clean_user, history_text)

    if _has_avoidant_or_unhelpful_action(clean_reply):
        return _specific_action_repair_reply(clean_user, history_text)

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

    if _is_repeated_reply(clean_reply, history_text):
        return _contextual_safe_reply(clean_user, history_text)

    if _is_too_short(clean_reply) and _has_distress_context(clean_user):
        return _contextual_safe_reply(clean_user, history_text)

    if _has_bad_support_reply(clean_reply):
        return _contextual_safe_reply(clean_user, history_text)

    if _has_irrelevant_modern_template(clean_reply, clean_user, history_text):
        return _repair_irrelevant_template(clean_user, history_text)

    return clean_reply


def _apply_priority_scenario_reply(
    clean_user: str,
    clean_reply: str,
    history_text: str,
    state: Any,
) -> str | None:
    """Route high-risk and high-confidence scenes before generic cleanup.

    This keeps the visible response aligned with the support workflow while
    avoiding a single giant decision block inside ``sanitize_user_visible_reply``.
    """

    if _is_self_harm_ambivalence_text(clean_user, history_text):
        return _self_harm_ambivalence_reply()

    if _is_modern_crisis_user_text(clean_user) or _is_crisis_user_text(clean_user):
        return _modern_crisis_support_reply()

    if _is_dangerous_place_text(clean_user, history_text):
        return _dangerous_place_safety_reply()

    if _is_account_handover_risk_text(clean_user, history_text):
        return _account_handover_safety_reply()

    if _is_coercive_relationship_risk_text(clean_user, history_text):
        return _coercive_relationship_safety_reply()

    if _is_other_harm_retaliation_text(clean_user, history_text):
        return _other_harm_retaliation_reply()

    if _is_family_career_conflict_text(clean_user, history_text):
        return _family_career_conflict_reply()

    if _is_plagiarism_accusation_response_text(clean_user, history_text):
        return _plagiarism_accusation_response_reply()

    if _is_friend_repair_uncertainty_text(clean_user, history_text):
        return _friend_repair_uncertainty_reply()

    if _is_refusal_guilt_boundary_text(clean_user, history_text):
        return _refusal_guilt_boundary_reply()

    if _is_friendship_loss_fear_text(clean_user, history_text):
        return _friendship_loss_fear_reply()

    if _is_public_attack_anonymous_text(clean_user, history_text):
        return _public_attack_anonymous_reply()

    if _is_privacy_betrayal_text(clean_user, history_text):
        return _privacy_betrayal_reply()

    if _is_study_loneliness_text(clean_user, history_text):
        return _study_loneliness_reply()

    if _is_rejection_self_worth_text(clean_user, history_text):
        return _rejection_self_worth_reply()

    if _is_relationship_self_erasure_text(clean_user, history_text):
        return _relationship_self_erasure_reply()

    if _is_traffic_near_miss_text(clean_user, history_text):
        return _traffic_near_miss_reply()

    if _is_jealousy_self_blame_text(clean_user, history_text):
        return _jealousy_self_blame_reply()

    if _is_repeated_death_ideation_text(clean_user, history_text):
        return _repeated_death_ideation_reply()

    if _is_privacy_leak_panic_text(clean_user, history_text):
        return _privacy_leak_panic_reply()

    if _is_decision_boundary_text(clean_user, history_text):
        return _decision_boundary_reply()

    if _is_loneliness_no_response_text(clean_user, history_text):
        return _loneliness_backup_plan_reply()

    if _is_pet_grief_text(clean_user, history_text):
        return _pet_grief_reply()

    if _is_class_activity_isolation_text(clean_user, history_text):
        return _class_activity_isolation_reply()

    if _is_stalking_fear_text(clean_user, history_text):
        return _stalking_fear_reply()

    if _is_sexual_harassment_contact_text(clean_user, history_text):
        return _sexual_harassment_contact_reply()

    if _is_social_approach_rejection_text(clean_user, history_text):
        return _social_approach_rejection_reply()

    if _is_internship_belittling_text(clean_user, history_text):
        return _internship_belittling_reply()

    if _is_tangled_distress_text(clean_user, history_text):
        return _tangled_distress_reply()

    if _is_disconnect_control_risk_text(clean_user, history_text):
        return _disconnect_control_risk_reply()

    if _is_research_group_exclusion_text(clean_user, history_text):
        return _research_group_exclusion_reply()

    if _is_game_avoidance_text(clean_user, history_text):
        return _game_avoidance_reply(clean_user)

    if _is_future_stuck_text(clean_user, history_text):
        return _future_stuck_reply()

    if _is_project_setback_text(clean_user, history_text) and (
        _is_too_short(clean_reply) or _has_weak_action_reply(clean_reply) or clean_reply.endswith(("对吗？", "对吗?"))
    ):
        return _project_setback_reply(clean_user, history_text)

    if _is_relationship_checking_text(clean_user, history_text) and (
        _is_too_short(clean_reply) or _has_weak_action_reply(clean_reply) or _is_overly_generic_support(clean_reply)
    ):
        return _relationship_checking_reply()

    if _is_task_overload_deadline_text(clean_user, history_text) and _has_weak_action_reply(clean_reply):
        return _task_overload_deadline_reply(clean_user, history_text)

    if _is_pre_exam_checking_text(clean_user, history_text) and _has_weak_action_reply(clean_reply):
        return _pre_exam_checking_reply(clean_user)

    if _is_award_disappointment_boundary_text(clean_user, history_text) and _has_weak_action_reply(clean_reply):
        return _award_disappointment_boundary_reply()

    if _is_classroom_panic_text(clean_user, history_text):
        return _classroom_panic_reply()

    if _is_thesis_checking_text(clean_user, history_text) and _has_weak_action_reply(clean_reply):
        return _thesis_checking_reply(clean_user)

    if _is_family_violence_return_text(clean_user, history_text):
        return _family_violence_return_reply()

    if _is_parent_divorce_middleman_text(clean_user, history_text):
        return _parent_divorce_middleman_reply()

    if _is_family_boundary_text(clean_user, history_text) and _has_weak_action_reply(clean_reply):
        return _family_boundary_reply()

    if _is_dorm_sleep_conflict_initial_text(clean_user, history_text):
        return _dorm_sleep_conflict_initial_reply()

    if _is_dorm_exclusion_confirmed_text(clean_user, history_text):
        return _dorm_exclusion_confirmed_reply()

    if _is_dorm_boundary_text(clean_user, history_text) and (
        _has_weak_action_reply(clean_reply) or state.stage == DialogueStage.DISCLOSURE_BOUNDARY
    ) and not (_has_pushy_reply(clean_reply) or _has_intense_distress(clean_user)):
        return _dorm_boundary_reply()

    if _is_support_disappointment_text(clean_user, history_text):
        return _support_disappointment_reply()

    if _is_social_mistake_next_day_text(clean_user, history_text) and _has_weak_action_reply(clean_reply):
        return _social_mistake_next_day_reply()

    if _is_group_work_no_response_text(clean_user, history_text) and (
        _has_social_mistake_template(clean_reply) or _has_weak_action_reply(clean_reply) or _is_overly_generic_support(clean_reply)
    ):
        return _group_work_no_response_reply()

    if _is_rumination_sarcasm_text(clean_user, history_text) and _has_weak_action_reply(clean_reply):
        return _rumination_sarcasm_reply()

    if _is_future_stuck_text(clean_user, history_text) and _has_irrelevant_family_template(clean_reply):
        return _future_stuck_reply()

    if _is_public_speaking_text(clean_user, history_text) and _has_weak_action_reply(clean_reply):
        return _public_speaking_reply()

    if _is_modern_medication_question(clean_user):
        return _modern_medication_boundary_reply()

    if _is_modern_diagnosis_question(clean_user):
        return _modern_diagnosis_boundary_reply(clean_user)

    if _is_modern_identity_question(clean_user):
        return _modern_identity_boundary_reply()

    return None


def _is_weak_input(text: str) -> bool:
    return text.strip() in WEAK_INPUTS


def _is_modern_medication_question(text: str) -> bool:
    compact = text.replace(" ", "")
    medication_terms = (
        "\u5b89\u7720\u836f",
        "\u892a\u9ed1\u7d20",
        "\u6297\u7126\u8651\u836f",
        "\u6297\u6291\u90c1\u836f",
        "\u5403\u836f",
        "\u7528\u836f",
        "\u836f\u91cf",
        "\u505c\u836f",
        "\u52a0\u836f",
        "\u6362\u836f",
        "\u533b\u751f\u5f00\u7684\u836f",
    )
    question_terms = (
        "\u80fd\u4e0d\u80fd",
        "\u53ef\u4ee5",
        "\u8981\u4e0d\u8981",
        "\u8981\u5403",
        "\u76f4\u63a5\u5403",
        "\u8be5\u5403",
        "\u600e\u4e48\u5403",
        "\u884c\u5417",
        "\u597d\u5417",
    )
    return any(term in compact for term in medication_terms) and any(term in compact for term in question_terms)


def _modern_medication_boundary_reply() -> str:
    return (
        "\u8fd9\u4e2a\u95ee\u9898\u4e0d\u9002\u5408\u7531\u6211\u6765\u66ff\u4f60\u51b3\u5b9a\u8981\u4e0d\u8981\u5403\u836f\u6216\u600e\u4e48\u7528\u836f\uff0c"
        "\u4e5f\u4e0d\u5efa\u8bae\u4f60\u81ea\u5df1\u968f\u4fbf\u52a0\u836f\u3001\u505c\u836f\u6216\u5403\u522b\u4eba\u7684\u836f\u3002"
        "\u66f4\u7a33\u59a5\u7684\u505a\u6cd5\u662f\u8054\u7cfb\u6821\u533b\u9662\u3001\u7cbe\u795e\u79d1\u6216\u6b63\u89c4\u533b\u751f\uff0c\u628a\u4f60\u7684\u7761\u7720\u3001\u7126\u8651\u548c\u8eab\u4f53\u53cd\u5e94\u8bf4\u6e05\u695a\u3002"
        "\u5982\u679c\u4f60\u4eca\u665a\u662f\u7761\u4e0d\u7740\uff0c\u6211\u53ef\u4ee5\u5148\u966a\u4f60\u505a\u4e00\u4e2a\u4e0d\u6d89\u53ca\u836f\u7269\u7684\u5c0f\u65b9\u6848\uff1a"
        "\u5148\u628a\u624b\u673a\u653e\u8fdc\u4e00\u70b9\uff0c\u706f\u5149\u8c03\u6697\uff0c\u53ea\u628a\u76ee\u6807\u653e\u5728\u201c\u5148\u8eba\u4e0b\u4f11\u606f\u5341\u5206\u949f\u201d\u3002"
    )


def _is_modern_diagnosis_question(text: str) -> bool:
    compact = text.replace(" ", "")
    diagnosis_terms = (
        "\u6291\u90c1\u75c7",
        "\u7126\u8651\u75c7",
        "\u53cc\u76f8",
        "\u5fc3\u7406\u6709\u75c5",
        "\u7cbe\u795e\u75c5",
        "\u8bca\u65ad",
        "\u662f\u4e0d\u662f\u6709\u75c5",
        "\u6b63\u4e0d\u6b63\u5e38",
    )
    question_terms = ("\u662f\u4e0d\u662f", "\u662f\u5426", "\u6709\u6ca1\u6709", "\u4f60\u89c9\u5f97", "\u7b97\u4e0d\u7b97", "\u50cf\u4e0d\u50cf")
    return any(term in compact for term in diagnosis_terms) and any(term in compact for term in question_terms)


def _modern_diagnosis_boundary_reply(user_text: str) -> str:
    if "\u4e0d\u6b63\u5e38" in user_text or "\u6709\u75c5" in user_text:
        return (
            "\u6211\u4e0d\u4f1a\u7528\u201c\u6709\u75c5\u201d\u6216\u201c\u4e0d\u6b63\u5e38\u201d\u6765\u5b9a\u4e49\u4f60\u3002"
            "\u4f60\u73b0\u5728\u53ef\u80fd\u662f\u5728\u7ecf\u5386\u538b\u529b\u3001\u4f4e\u843d\u3001\u7126\u8651\u6216\u4eba\u9645\u56f0\u6270\uff0c\u8fd9\u4e9b\u90fd\u503c\u5f97\u88ab\u8ba4\u771f\u5bf9\u5f85\uff0c\u4f46\u4e0d\u7b49\u4e8e\u4f60\u8fd9\u4e2a\u4eba\u6709\u95ee\u9898\u3002"
            "\u6211\u4eec\u53ef\u4ee5\u5148\u4e0d\u8d34\u6807\u7b7e\uff0c\u53ea\u770b\u5b83\u5bf9\u4f60\u7684\u7761\u7720\u3001\u5b66\u4e60\u548c\u751f\u6d3b\u5f71\u54cd\u5230\u4ec0\u4e48\u7a0b\u5ea6\u3002"
        )
    return (
        "\u6211\u4e0d\u80fd\u5728\u804a\u5929\u91cc\u7ed9\u4f60\u4e0b\u533b\u5b66\u8bca\u65ad\uff0c\u4f46\u6211\u53ef\u4ee5\u966a\u4f60\u5148\u628a\u72b6\u6001\u7406\u6e05\u695a\u3002"
        "\u6bd4\u8d77\u9a6c\u4e0a\u5224\u65ad\u662f\u4e0d\u662f\u67d0\u79cd\u75c7\uff0c\u66f4\u91cd\u8981\u7684\u662f\u770b\u5b83\u6301\u7eed\u4e86\u591a\u4e45\uff0c\u6709\u6ca1\u6709\u660e\u663e\u5f71\u54cd\u7761\u7720\u3001\u98df\u6b32\u3001\u4e0a\u8bfe\u3001\u4eba\u9645\u548c\u65e5\u5e38\u529f\u80fd\u3002"
        "\u5982\u679c\u5df2\u7ecf\u6301\u7eed\u4e00\u6bb5\u65f6\u95f4\u6216\u5f71\u54cd\u5f88\u660e\u663e\uff0c\u5efa\u8bae\u4f60\u9884\u7ea6\u5b66\u6821\u5fc3\u7406\u4e2d\u5fc3\u6216\u6b63\u89c4\u533b\u9662\u505a\u8bc4\u4f30\u3002"
        "\u73b0\u5728\u6211\u4eec\u53ef\u4ee5\u5148\u4ece\u6700\u56f0\u6270\u4f60\u7684\u4e00\u70b9\u5f00\u59cb\u3002"
    )


def _is_modern_identity_question(text: str) -> bool:
    compact = text.replace(" ", "").lower()
    return any(
        term in compact
        for term in (
            "\u4f60\u662f\u4eba\u8fd8\u662f\u673a\u5668\u4eba",
            "\u4f60\u662fai",
            "\u4f60\u662f\u4e0d\u662fai",
            "\u4f60\u662fchatgpt",
            "\u4f60\u662f\u4e0d\u662fchatgpt",
            "\u4f60\u662fdeepseek",
            "\u4f60\u662f\u8c46\u5305",
            "\u4f60\u662f\u4ec0\u4e48",
            "\u4f60\u5230\u5e95\u662f\u8c01",
        )
    )


def _modern_identity_boundary_reply() -> str:
    return (
        "\u6211\u662f\u8fd9\u4e2a\u9879\u76ee\u91cc\u7684\u6821\u56ed\u5fc3\u7406\u652f\u6301\u52a9\u624b\uff0c\u7531 AI \u63d0\u4f9b\u56de\u590d\uff1b\u6211\u4e0d\u662f\u771f\u4eba\u54a8\u8be2\u5e08\uff0c\u4e5f\u4e0d\u80fd\u66ff\u4ee3\u533b\u751f\u6216\u5b66\u6821\u5fc3\u7406\u8001\u5e08\u3002"
        "\u6211\u53ef\u4ee5\u505a\u7684\u662f\u966a\u4f60\u68b3\u7406\u60c5\u7eea\u3001\u6574\u7406\u95ee\u9898\u3001\u7ed9\u51fa\u76f8\u5bf9\u5b89\u5168\u7684\u5efa\u8bae\uff1b"
        "\u5982\u679c\u9047\u5230\u9ad8\u98ce\u9669\u60c5\u51b5\uff0c\u6211\u4f1a\u4f18\u5148\u5efa\u8bae\u4f60\u8054\u7cfb\u73b0\u5b9e\u4e2d\u7684\u53ef\u9760\u4eba\u548c\u4e13\u4e1a\u652f\u6301\u3002"
    )


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
        "\u4e0d\u5728\u4e86",
        "\u5982\u679c\u6211\u4e0d\u5728",
        "\u4eba\u6ca1\u4e86",
        "\u7ed3\u675f\u751f\u547d",
        "\u4f24\u5bb3\u81ea\u5df1",
        "\u4e0d\u786e\u5b9a\u80fd\u4e0d\u80fd\u4fdd\u8bc1",
        "\u4e0d\u80fd\u4fdd\u8bc1",
    )
    if any(phrase in compact for phrase in ("\u4e0d\u60f3\u4f24\u5bb3\u81ea\u5df1", "\u4e0d\u4f1a\u4f24\u5bb3\u81ea\u5df1", "\u6ca1\u6709\u60f3\u4f24\u5bb3\u81ea\u5df1")):
        return any(term in compact for term in crisis_terms if term != "\u4f24\u5bb3\u81ea\u5df1")
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


def _is_coercive_relationship_risk_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    has_relationship = any(term in combined for term in ("\u5206\u624b", "\u79bb\u5f00\u4ed6", "\u79bb\u5f00\u5979", "\u4eb2\u5bc6\u5173\u7cfb", "\u5bf9\u65b9"))
    has_threat = any(term in combined for term in ("\u505a\u50bb\u4e8b", "\u51fa\u4e8b", "\u5bb3\u6b7b", "\u804a\u5929\u8bb0\u5f55\u53d1\u51fa\u53bb", "\u5a01\u80c1", "\u66dd\u5149"))
    return has_relationship and has_threat


def _is_self_harm_ambivalence_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    has_self_harm = any(term in combined for term in ("\u81ea\u4f24", "\u5272\u81ea\u5df1", "\u4f24\u5bb3\u81ea\u5df1", "\u75bc\u75db\u51b7\u9759"))
    has_ambivalence = any(term in combined for term in ("\u4e0d\u662f\u771f\u7684\u60f3\u6b7b", "\u53ea\u662f\u60f3\u51b7\u9759", "\u4e0d\u60f3\u8ba9\u522b\u4eba\u77e5\u9053", "\u5f88\u4e22\u4eba"))
    return has_self_harm and has_ambivalence


def _self_harm_ambivalence_reply() -> str:
    return (
        "\u4f60\u4e0d\u60f3\u88ab\u522b\u4eba\u77e5\u9053\uff0c\u8fd9\u79cd\u7f9e\u803b\u548c\u62c5\u5fc3\u6211\u80fd\u7406\u89e3\uff1b\u4f46\u7528\u75bc\u75db\u8ba9\u81ea\u5df1\u51b7\u9759\uff0c\u8bf4\u660e\u60c5\u7eea\u5df2\u7ecf\u8d85\u8fc7\u4f60\u4e00\u4e2a\u4eba\u8212\u670d\u627f\u53d7\u7684\u8303\u56f4\u4e86\u3002"
        "\u5373\u4f7f\u4f60\u4e0d\u662f\u771f\u7684\u60f3\u6b7b\uff0c\u4e5f\u53ef\u80fd\u5728\u5931\u63a7\u65f6\u9020\u6210\u4f24\u5bb3\u3002\u5148\u628a\u53ef\u4ee5\u4f24\u5230\u81ea\u5df1\u7684\u4e1c\u897f\u653e\u8fdc\uff0c\u5c3d\u91cf\u5230\u6709\u4eba\u7684\u5730\u65b9\uff0c\u6216\u7acb\u523b\u8054\u7cfb\u4e00\u4e2a\u4f60\u4fe1\u5f97\u8fc7\u7684\u4eba\u966a\u4f60\u3002"
        "\u4e3a\u4e86\u5148\u6491\u8fc7\u8fd9\u51e0\u5206\u949f\uff0c\u53ef\u4ee5\u63e1\u51b0\u5757\u3001\u7528\u51b7\u6c34\u51b2\u624b\u3001\u7528\u529b\u8e29\u5730\uff0c\u6216\u8bf4\u51fa\u623f\u95f4\u91cc\u4e94\u4e2a\u4f60\u770b\u5f97\u5230\u7684\u7269\u54c1\u3002\u8fd9\u4e9b\u53ea\u662f\u4e34\u65f6\u66ff\u4ee3\uff0c\u4f46\u73b0\u5728\u5148\u4fdd\u8bc1\u5b89\u5168\u6700\u91cd\u8981\u3002"
    )


def _is_dangerous_place_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    return any(term in combined for term in ("\u5929\u53f0", "\u697c\u9876", "\u9ad8\u5904", "\u6865\u4e0a", "\u6cb3\u8fb9")) and any(
        term in combined for term in ("\u4e00\u4e2a\u4eba", "\u5439\u5439\u98ce", "\u5f88\u70e6", "\u5b89\u9759\u4e00\u70b9", "\u4e0d\u7528\u62c5\u5fc3")
    )


def _dangerous_place_safety_reply() -> str:
    return (
        "\u6211\u9700\u8981\u5f88\u76f4\u63a5\u5730\u8bf4\uff1a\u73b0\u5728\u5148\u4e0d\u8981\u53bb\u5929\u53f0\u3001\u697c\u9876\u6216\u4efb\u4f55\u9ad8\u5904\uff0c\u4e5f\u4e0d\u8981\u4e00\u4e2a\u4eba\u5f85\u5728\u5371\u9669\u5730\u65b9\u3002"
        "\u4f60\u73b0\u5728\u60c5\u7eea\u5f88\u5f3a\uff0c\u53c8\u60f3\u4e00\u4e2a\u4eba\u5f85\u7740\uff0c\u8fd9\u4f1a\u589e\u52a0\u51b2\u52a8\u98ce\u9669\u3002"
        "\u8bf7\u5148\u7559\u5728\u6709\u4eba\u7ecf\u8fc7\u7684\u5730\u65b9\uff0c\u6bd4\u5982\u5bbf\u7ba1\u5904\u3001\u4fbf\u5229\u5e97\u3001\u503c\u73ed\u5ba4\u6216\u8d70\u5eca\uff0c\u7136\u540e\u7ed9\u5ba4\u53cb\u3001\u540c\u5b66\u6216\u8f85\u5bfc\u5458\u53d1\u4e00\u53e5\uff1a\u201c\u6211\u73b0\u5728\u72b6\u6001\u4e0d\u592a\u5b89\u5168\uff0c\u4f60\u80fd\u6765\u966a\u6211\u4e00\u4e0b\u5417\uff1f\u201d"
    )


def _is_account_handover_risk_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    has_handover = any(term in combined for term in ("\u8d26\u53f7\u5bc6\u7801", "\u5bc6\u7801\u4ea4\u7ed9", "\u4ea4\u4ee3", "\u7559\u7740\u4e5f\u6ca1\u610f\u601d"))
    has_withdrawal = any(term in combined for term in ("\u4e00\u5207\u90fd\u5f88\u7d2f", "\u6ca1\u610f\u601d", "\u4e0d\u60f3\u8bf4\u5f97\u90a3\u4e48\u4e25\u91cd", "\u4e0d\u60f3\u8ba9\u522b\u4eba\u77e5\u9053"))
    return has_handover and has_withdrawal


def _account_handover_safety_reply() -> str:
    return (
        "\u6211\u542c\u5230\u4f60\u8bf4\u201c\u7559\u7740\u4e5f\u6ca1\u610f\u601d\u201d\uff0c\u53c8\u5df2\u7ecf\u5728\u4ea4\u4ee3\u8d26\u53f7\u548c\u5bc6\u7801\uff0c\u8fd9\u4e0d\u662f\u9700\u8981\u4f60\u4e00\u4e2a\u4eba\u7ee7\u7eed\u786c\u6491\u7684\u65f6\u523b\u3002"
        "\u8bf7\u4f60\u73b0\u5728\u5148\u6682\u505c\u6574\u7406\u8d26\u53f7\u548c\u5bc6\u7801\uff0c\u4e5f\u5148\u4e0d\u8981\u7ee7\u7eed\u628a\u91cd\u8981\u4fe1\u606f\u4ea4\u7ed9\u522b\u4eba\u3002"
        "\u53ef\u4ee5\u7acb\u523b\u8054\u7cfb\u4e00\u4e2a\u771f\u5b9e\u7684\u4eba\u5230\u4f60\u8eab\u8fb9\uff0c\u53ea\u8bf4\uff1a\u201c\u6211\u73b0\u5728\u72b6\u6001\u4e0d\u5b89\u5168\uff0c\u9700\u8981\u4f60\u966a\u6211\u3002\u201d\u5728\u6709\u4eba\u5230\u6765\u524d\uff0c\u8bf7\u5148\u8fdc\u79bb\u5371\u9669\u5730\u70b9\u548c\u53ef\u80fd\u4f24\u5bb3\u81ea\u5df1\u7684\u7269\u54c1\u3002"
    )


def _coercive_relationship_safety_reply() -> str:
    return (
        "\u8fd9\u5df2\u7ecf\u4e0d\u662f\u666e\u901a\u7684\u633d\u7559\uff0c\u800c\u662f\u7528\u81ea\u4f24\u6216\u66dd\u5149\u5a01\u80c1\u628a\u4f60\u56f0\u4f4f\u3002"
        "\u4ed6\u7684\u751f\u547d\u5b89\u5168\u4e0d\u5e94\u8be5\u7531\u4f60\u4e00\u4e2a\u4eba\u627f\u62c5\uff0c\u4f60\u53ef\u4ee5\u628a\u8fd9\u4e2a\u98ce\u9669\u8f6c\u4ea4\u7ed9\u4ed6\u7684\u5bb6\u4eba\u3001\u670b\u53cb\u3001\u8f85\u5bfc\u5458\u6216\u7d27\u6025\u670d\u52a1\u3002"
        "\u8bf7\u4f60\u6682\u65f6\u4e0d\u8981\u5355\u72ec\u89c1\u9762\u644a\u724c\uff0c\u5148\u4fdd\u7559\u804a\u5929\u8bb0\u5f55\u548c\u5a01\u80c1\u8bc1\u636e\uff0c\u627e\u4e00\u4e2a\u53ef\u4fe1\u7684\u4eba\u966a\u4f60\u4e00\u8d77\u5904\u7406\u3002"
    )


def _is_other_harm_retaliation_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    has_anger_or_revenge = any(term in combined for term in ("咽不下这口气", "欺负我", "找他", "算账", "报复", "教训他", "伤害别人", "伤害他"))
    has_action_pull = any(term in combined for term in ("不去找他", "去找他", "冲过去", "控制不住", "很想打", "我怕我会"))
    return has_anger_or_revenge and has_action_pull


def _other_harm_retaliation_reply() -> str:
    return (
        "你想维护尊严、咽不下这口气，这可以理解；但如果现在用冲动方式去找他，后果很可能转到你身上，反而让你更被动。"
        "先把“马上去找他”暂停下来，联系一个可信的人陪着你，或者直接找辅导员、老师、保卫处介入。"
        "你可以先保存证据、截图和时间线，等情绪降一点后用正式渠道处理，而不是让愤怒替你决定下一步。"
    )


def _is_family_career_conflict_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    has_family = any(term in combined for term in ("家里", "爸妈", "父母", "他们会说", "供我读大学"))
    has_career = any(term in combined for term in ("县城工作", "回县城", "大城市", "稳定岗位", "以后别那么辛苦", "工作"))
    return has_family and has_career


def _family_career_conflict_reply() -> str:
    return (
        "这句话背后有爱，也有控制感：他们希望你少吃苦，但你也需要为自己的发展保留选择。"
        "先不要只用“我想出去”对抗，可以把回应改成具体计划：我会先在大城市工作两年，设定存款目标、职业评估点和备选方案；如果发展不理想，再重新考虑稳定岗位。"
        "具体计划会比情绪争辩更容易被听见，也能让你不是在证明“不听话”，而是在说明自己有承担后果的能力。"
    )


def _is_plagiarism_accusation_response_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    has_accusation = any(term in combined for term in ("抄袭", "质疑", "比赛作品", "作品被质疑", "心虚"))
    has_response_question = any(term in combined for term in ("怎么回应", "不回应", "应该怎么办", "显得心虚", "解释"))
    return has_accusation and has_response_question


def _plagiarism_accusation_response_reply() -> str:
    return (
        "可以回应，但最好用结构化、可验证的方式，而不是和对方互怼。"
        "你们可以按四点准备：一，说明参考来源和许可证；二，列出自研模块、数据、界面或实验；三，附上提交记录、版本迭代截图；四，表示欢迎评委或老师核查。"
        "简短、透明、可验证，比急着辩解更有力量，也能把讨论从情绪攻击拉回事实。"
    )


def _is_friend_repair_uncertainty_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    has_friend_conflict = any(term in combined for term in ("朋友吵架", "和朋友吵", "道歉", "修复", "她不接受", "他不接受"))
    has_uncertainty = any(term in combined for term in ("不接受怎么办", "不愿意原谅", "不想聊", "怎么办"))
    return has_friend_conflict and has_uncertainty


def _friend_repair_uncertainty_reply() -> str:
    return (
        "她可能需要时间，这也是她的边界。你能做的是把自己的部分表达清楚，但不要逼她立刻恢复原状。"
        "可以发一条短消息：“我知道刚才那件事让你不舒服，我愿意为我的部分道歉。你现在不想聊也没关系，等你愿意的时候我在。”"
        "关系修复不是马上回到从前，而是双方慢慢重新建立安全感；你为自己的部分负责，就已经迈出了一步。"
    )


def _is_refusal_guilt_boundary_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    has_refusal = any(term in combined for term in ("不会拒绝", "帮他改PPT", "改PPT", "不够朋友", "拒绝别人", "自己的作业"))
    has_guilt = any(term in combined for term in ("内疚", "怪自己", "不够朋友", "怕他说", "答应了"))
    return has_refusal and has_guilt


def _refusal_guilt_boundary_reply() -> str:
    return (
        "你不是没有边界，而是拒绝时很怕让别人失望、破坏关系，所以表面答应，心里却积累委屈。"
        "真正稳定的关系应该能承受合理拒绝。你可以不用生硬地说“不帮”，而是给出边界：“我今晚自己的作业也很急，没法完整帮你改。最多可以帮你看一页结构，其他你得自己处理。”"
        "这样既表达了限制，也保留了善意；拒绝不等于不够朋友，透支自己才会让关系变得更累。"
    )


def _is_friendship_loss_fear_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    has_friendship = any(term in combined for term in ("好友", "朋友", "疏远", "不理我", "不想和我好了", "不想跟我好了"))
    has_loss_fear = any(term in combined for term in ("真的不想和我好", "不想和我好了", "讨厌我", "不要我", "如果她真的"))
    return has_friendship and has_loss_fear


def _friendship_loss_fear_reply() -> str:
    return (
        "如果她真的在疏远你，那会很痛，但友情的变化不等于你不值得被喜欢。"
        "你可以先尊重她现在的距离，同时把生活重心慢慢分散到其他连接上，不要用一个人的反应定义你全部的社交价值。"
        "下一步可以只做一个低压力动作：发一条不追问、不施压的消息；如果对方仍然冷淡，就先把注意力放回自己的作息、学习和其他支持关系。"
    )


def _is_public_attack_anonymous_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    return any(term in combined for term in ("表白墙", "匿名发我", "朋友圈截图", "评论里", "跟着骂", "所有人都在议论"))


def _public_attack_anonymous_reply() -> str:
    return (
        "被公开攻击会让人产生很强的羞耻和暴露感，尤其还用了你的个人截图。你现在不需要硬撑着证明自己没事。"
        "第一步先保存证据：截图、链接、发布时间和评论；第二步减少反复刷评论，因为持续观看会让伤害不断重复发生。"
        "如果内容涉及造谣、隐私或持续攻击，可以联系平台管理员、辅导员或学校相关老师，目标是让伤害停止，而不是一个人和所有评论对抗。"
    )


def _is_privacy_betrayal_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    return any(term in combined for term in ("私人的事", "告诉了朋友", "告诉了别人", "顺口提到", "不敢相信别人", "被背叛"))


def _privacy_betrayal_reply() -> str:
    return (
        "私人信息被说出去，会让人感到被背叛和暴露。哪怕她说不是故意的，你的受伤仍然成立，因为被影响的是你的安全感。"
        "现在可以先允许自己生气，而不是急着替她找理由。等情绪稍微降一点，再考虑要不要说清楚边界：“这件事对我很私密，以后没有我的同意，请不要再转述。”"
        "之后你也可以把信任分层：不是从此谁都不能信，而是更谨慎地区分哪些内容可以说、哪些内容需要留给更可靠的人。"
    )


def _is_study_loneliness_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    return any(term in combined for term in ("二战考研", "租房复习", "背书刷题", "没人说话", "被世界剩下"))


def _study_loneliness_reply() -> str:
    return (
        "你现在承受的是长期孤独加高压目标，不只是“学习累”。别人进入新阶段，会让你更容易觉得自己停在原地。"
        "可二战不是被世界剩下，而是你在一条更慢、更窄、更少人陪的路上走，难受是正常反应。"
        "可以先给这条路加一个最小连接：每天固定和一个人说一句近况，或每周安排一次线下自习/散步，不让复习生活只剩下题目和房间。"
    )


def _is_rejection_self_worth_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    has_romantic_context = any(term in combined for term in ("表白", "喜欢她", "喜欢他", "喜欢的人", "被拒绝", "委婉地拒绝"))
    has_self_worth_drop = any(term in combined for term in ("小丑", "后悔认识", "尴尬", "不值得", "可笑"))
    return has_romantic_context and has_self_worth_drop


def _rejection_self_worth_reply() -> str:
    return (
        "被拒绝后会有尴尬、失落和自我怀疑，这很正常。你不是小丑，你只是认真表达了一次喜欢。"
        "对方没有接受，说明关系没有走向你期待的方向，不说明你的感情很可笑，也不说明你这个人不值得被喜欢。"
        "现在先别逼自己立刻大方，可以给自己一点距离：减少反复回看聊天记录，先把注意力放回今天能完成的一件小事。"
    )


def _is_relationship_self_erasure_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    return any(term in combined for term in ("谈恋爱以后越来越不像自己", "马上道歉", "推掉自己的安排", "不够爱他", "牺牲"))


def _relationship_self_erasure_reply() -> str:
    return (
        "你在这段关系里像是一直用牺牲来换安全感。这样短期可能减少冲突，长期会让你越来越没有自己。"
        "爱不应该只能靠压低自己来证明。你可以先选一个很小的边界练习：保留一个自己的安排，不临时取消；或者在道歉前先问自己“这真的是我的责任吗”。"
        "如果对方总是用不高兴来让你让步，这段关系就需要更认真地看边界和安全感，而不是只要求你继续忍。"
    )


def _is_traffic_near_miss_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    return any(term in combined for term in ("差点出事故", "急刹车", "闪回", "一坐车就紧张", "那一瞬间"))


def _traffic_near_miss_reply() -> str:
    return (
        "身体没受伤不代表心理上就完全没影响。突发惊吓后，反复想起画面、对相似声音敏感，是人在经历强烈威胁后的常见反应。"
        "你不是小题大做，而是大脑还在确认安全。短期可以先做分级恢复：先坐较短路线、选靠近出口或让你安心的位置，和可信的人同行一次。"
        "如果闪回、惊跳或回避持续影响出行，建议找学校心理中心或校医院做一次评估，不要只靠“没事就好”把它压下去。"
    )


def _is_jealousy_self_blame_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    return any(term in combined for term in ("朋友拿了奖", "替她开心", "嫉妒她", "很阴暗", "心里很酸"))


def _jealousy_self_blame_reply() -> str:
    return (
        "嫉妒不等于你不希望朋友好，它常常说明你也很渴望被看见、被肯定。你能为这种情绪感到不舒服，说明你在乎关系和自己的价值。"
        "先不用因为有嫉妒就否定自己，可以把它当成一个信号：我想要的认可是什么、我最近是不是也很需要被肯定。"
        "你可以同时做到两件事：真诚祝贺朋友，也给自己的失落留一点位置，而不是把自己骂成“阴暗”。"
    )


def _is_repeated_death_ideation_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    has_death_context = any(term in combined for term in ("\u6b7b\u4ea1", "\u6b7b", "\u4eba\u6ca1\u4e86", "\u4e0d\u7528\u518d\u9762\u5bf9", "\u4e0d\u60f3\u6d3b", "\u6d88\u5931", "\u544a\u522b"))
    has_frequency_or_intrusion = any(term in combined for term in ("\u4e00\u95ea\u800c\u8fc7", "\u8fd9\u51e0\u5929", "\u4e71\u60f3", "\u51fa\u73b0\u5f97\u6bd4\u8f83\u591a", "\u53cd\u590d", "\u95ee\u6b7b\u4ea1"))
    return has_death_context and has_frequency_or_intrusion


def _is_award_disappointment_boundary_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    return any(term in combined for term in ("\u5956\u5b66\u91d1", "\u8bc4\u5956", "\u843d\u9009", "\u540d\u5355")) and any(
        term in combined for term in ("\u5c0f\u9898\u5927\u505a", "\u4e0d\u60f3\u8ddf\u4efb\u4f55\u4eba\u8bf4", "\u6015\u522b\u4eba\u89c9\u5f97")
    )


def _award_disappointment_boundary_reply() -> str:
    return (
        "\u4f60\u53ef\u4ee5\u6682\u65f6\u4e0d\u516c\u5f00\u8bf4\uff0c\u8fd9\u4e0d\u4ee3\u8868\u4f60\u662f\u5c0f\u9898\u5927\u505a\u3002"
        "\u4f46\u4e5f\u5c3d\u91cf\u4e0d\u8981\u628a\u8fd9\u79cd\u843d\u5dee\u5168\u90e8\u61cb\u6210\u201c\u6211\u4e0d\u591f\u597d\u201d\u3002"
        "\u4eca\u665a\u53ef\u4ee5\u5148\u505a\u4e00\u4e2a\u4f4e\u5f3a\u5ea6\u5904\u7406\uff1a\u628a\u8bc4\u5956\u89c4\u5219\u3001\u4f60\u5df2\u505a\u7684\u6750\u6599\u3001\u4e0b\u6b21\u53ef\u8865\u7684\u8bc1\u636e\u5206\u5f00\u5199\u4e0b\u6765\u3002"
        "\u7b49\u60c5\u7eea\u964d\u4e00\u70b9\uff0c\u518d\u51b3\u5b9a\u8981\u4e0d\u8981\u5411\u8f85\u5bfc\u5458\u6216\u8d1f\u8d23\u8001\u5e08\u793c\u8c8c\u8be2\u95ee\u8bc4\u9009\u4f9d\u636e\u3002"
    )


def _repeated_death_ideation_reply() -> str:
    return (
        "\u8fd9\u4e0d\u50cf\u662f\u666e\u901a\u7684\u4e00\u4e2a\u5ff5\u5934\u95ea\u8fc7\uff0c\u56e0\u4e3a\u5b83\u8fd9\u51e0\u5929\u53cd\u590d\u51fa\u73b0\uff0c\u6240\u4ee5\u6211\u60f3\u5148\u966a\u4f60\u628a\u5b89\u5168\u653e\u5728\u524d\u9762\u3002"
        "\u4eca\u665a\u5c3d\u91cf\u4e0d\u8981\u4e00\u4e2a\u4eba\u5f85\u592a\u4e45\uff0c\u53ef\u4ee5\u5148\u628a\u72b6\u6001\u544a\u8bc9\u4e00\u4e2a\u53ef\u4fe1\u7684\u4eba\uff1a\u201c\u6211\u8fd9\u51e0\u5929\u6709\u4e9b\u5ff5\u5934\u53cd\u590d\u51fa\u73b0\uff0c\u6211\u73b0\u5728\u9700\u8981\u4f60\u966a\u6211\u4e00\u4e0b\u3002\u201d"
        "\u5982\u679c\u5ff5\u5934\u53d8\u5f3a\u3001\u51fa\u73b0\u5177\u4f53\u8ba1\u5212\uff0c\u6216\u8005\u4f60\u62c5\u5fc3\u81ea\u5df1\u63a7\u5236\u4e0d\u4f4f\uff0c\u8bf7\u7acb\u523b\u8054\u7cfb\u8eab\u8fb9\u7684\u4eba\u3001\u8f85\u5bfc\u5458\u3001\u5b66\u6821\u5fc3\u7406\u4e2d\u5fc3\u6216\u5f53\u5730\u7d27\u6025\u670d\u52a1\u3002"
    )


def _is_privacy_leak_panic_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    has_leak = any(term in combined for term in ("\u79c1\u5bc6\u7167\u7247", "\u7167\u7247\u6cc4\u9732", "\u9690\u79c1\u6cc4\u9732", "\u88ab\u53d1\u51fa\u53bb", "\u88ab\u4f20\u51fa\u53bb"))
    has_withdrawal = any(term in combined for term in ("\u4e0d\u60f3\u544a\u8bc9\u522b\u4eba", "\u592a\u4e22\u8138", "\u5173\u624b\u673a", "\u6d88\u5931\u51e0\u5929", "\u4e0d\u60f3\u89c1\u4eba"))
    return has_leak and has_withdrawal


def _privacy_leak_panic_reply() -> str:
    return (
        "\u4f60\u89c9\u5f97\u4e22\u8138\u548c\u60f3\u5173\u6389\u624b\u673a\uff0c\u662f\u56e0\u4e3a\u8fd9\u4ef6\u4e8b\u8ba9\u4f60\u89c9\u5f97\u5931\u63a7\u548c\u88ab\u66b4\u9732\uff0c\u4f46\u8fd9\u4e0d\u662f\u4f60\u7684\u9519\u3002"
        "\u4f60\u4e0d\u9700\u8981\u7acb\u523b\u544a\u8bc9\u5f88\u591a\u4eba\u7ec6\u8282\uff0c\u53ea\u8981\u5148\u627e\u4e00\u4e2a\u6700\u53ef\u4fe1\u7684\u4eba\uff0c\u8bf4\uff1a\u201c\u6211\u9047\u5230\u9690\u79c1\u6cc4\u9732\u98ce\u9669\uff0c\u73b0\u5728\u5f88\u614c\uff0c\u9700\u8981\u4f60\u966a\u6211\u4e00\u4e0b\u3002\u201d"
        "\u540c\u65f6\u5148\u4fdd\u7559\u804a\u5929\u8bb0\u5f55\u3001\u622a\u56fe\u548c\u65f6\u95f4\u7ebf\uff0c\u4e0d\u8981\u6025\u7740\u5220\u8bc1\u636e\uff1b\u5982\u679c\u6709\u4f24\u5bb3\u81ea\u5df1\u7684\u51b2\u52a8\uff0c\u8981\u7acb\u523b\u8054\u7cfb\u8eab\u8fb9\u7684\u4eba\u6216\u7d27\u6025\u6c42\u52a9\u3002"
    )


def _is_decision_boundary_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    has_decision_request = any(term in combined for term in ("\u662f\u4e0d\u662f\u5c31\u8be5", "\u5e2e\u6211\u51b3\u5b9a", "\u4f60\u66ff\u6211\u51b3\u5b9a", "\u5230\u5e95\u8981\u4e0d\u8981", "\u8be5\u4e0d\u8be5"))
    has_relationship_or_major_choice = any(term in combined for term in ("\u5206\u624b", "\u5206\u4e0d\u5206", "\u5206\u5f00", "\u8f6c\u4e13\u4e1a", "\u9000\u5b66", "\u79bb\u5f00", "\u8f9e\u804c"))
    return has_decision_request and has_relationship_or_major_choice


def _decision_boundary_reply() -> str:
    return (
        "\u6211\u4e0d\u9002\u5408\u76f4\u63a5\u66ff\u4f60\u505a\u8fd9\u4e2a\u51b3\u5b9a\uff0c\u4f46\u53ef\u4ee5\u5e2e\u4f60\u628a\u5224\u65ad\u4f9d\u636e\u7406\u6e05\u695a\u3002"
        "\u4f60\u53ef\u4ee5\u5148\u770b\u4e09\u4e2a\u95ee\u9898\uff1a\u8fd9\u6bb5\u5173\u7cfb\u91cc\u957f\u671f\u7684\u57fa\u672c\u611f\u53d7\u662f\u5b89\u5fc3\u8fd8\u662f\u7d27\u7ef7\uff1f\u95ee\u9898\u6c9f\u901a\u540e\u6709\u6ca1\u6709\u6301\u7eed\u6539\u5584\uff1f\u5982\u679c\u672a\u6765\u4e00\u6bb5\u65f6\u95f4\u90fd\u4fdd\u6301\u73b0\u5728\u8fd9\u6837\uff0c\u4f60\u80fd\u5426\u63a5\u53d7\uff1f"
        "\u5982\u679c\u4e09\u4e2a\u7b54\u6848\u90fd\u6307\u5411\u957f\u671f\u6d88\u8017\uff0c\u90a3\u81f3\u5c11\u8bf4\u660e\u9700\u8981\u4e25\u8083\u5904\u7406\u3002\u4f60\u53ef\u4ee5\u5148\u8fdb\u884c\u4e00\u6b21\u660e\u786e\u6c9f\u901a\uff0c\u63d0\u51fa\u5177\u4f53\u9700\u6c42\u548c\u89c2\u5bdf\u671f\u9650\uff1b\u5982\u679c\u5bf9\u65b9\u884c\u52a8\u6301\u7eed\u6539\u5584\uff0c\u4f60\u518d\u770b\uff0c\u5982\u679c\u53cd\u590d\u65e0\u6548\uff0c\u5c31\u9700\u8981\u8ba4\u771f\u8003\u8651\u79bb\u5f00\u662f\u5426\u662f\u5728\u4fdd\u62a4\u81ea\u5df1\u3002\u6700\u7ec8\u51b3\u5b9a\u4ecd\u7136\u7531\u4f60\u505a\u3002"
    )


def _is_loneliness_no_response_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    return "\u6ca1\u4eba\u56de" in combined and any(term in combined for term in ("\u66f4\u96be\u53d7", "\u5468\u672b", "\u4e00\u4e2a\u4eba", "\u6ca1\u4eba\u9700\u8981"))


def _loneliness_backup_plan_reply() -> str:
    return (
        "\u8fd9\u4e2a\u62c5\u5fc3\u5f88\u73b0\u5b9e\uff1a\u4f60\u60f3\u8981\u8fde\u63a5\uff0c\u4f46\u628a\u5e0c\u671b\u5168\u90e8\u653e\u5728\u522b\u4eba\u662f\u5426\u56de\u590d\u4e0a\uff0c\u4f1a\u8ba9\u4f60\u66f4\u88ab\u52a8\u3002"
        "\u53ef\u4ee5\u540c\u65f6\u51c6\u5907\u4e24\u4e2a\u65b9\u6848\uff1a\u6709\u4eba\u56de\u5c31\u7ea6\u4e00\u4e2a\u4f4e\u538b\u529b\u89c1\u9762\uff0c\u6ca1\u4eba\u56de\u5c31\u81ea\u5df1\u505a\u4e00\u4e2a\u4e0d\u4f9d\u8d56\u522b\u4eba\u7684\u5c0f\u51fa\u95e8\u8ba1\u5212\uff0c\u6bd4\u5982\u4e70\u4e00\u676f\u996e\u6599\u3001\u6563\u6b65\u5341\u5206\u949f\u6216\u53bb\u98df\u5802\u5403\u70b9\u4e1c\u897f\u3002"
        "\u8fd9\u4e0d\u662f\u8ba9\u4f60\u5047\u88c5\u4e0d\u9700\u8981\u522b\u4eba\uff0c\u800c\u662f\u5148\u8ba9\u4eca\u5929\u4e0d\u5b8c\u5168\u88ab\u201c\u6709\u6ca1\u6709\u4eba\u56de\u6211\u201d\u51b3\u5b9a\u3002"
    )


def _is_sexual_harassment_contact_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    has_violation_context = any(
        term in combined
        for term in ("\u6027\u9a9a\u6270", "\u7325\u4eb5", "\u8d8a\u754c", "\u88ab\u6478", "\u79c1\u5bc6", "\u5f3a\u8feb", "\u6076\u5fc3")
    )
    has_forced_contact = any(term in combined for term in ("\u8fd8\u8981\u89c1\u5230", "\u6d3b\u52a8\u91cc\u89c1", "\u4e00\u60f3\u5230\u4ed6", "\u4e0d\u6562\u8bf4\u51fa\u53e3"))
    return has_violation_context and has_forced_contact


def _sexual_harassment_contact_reply() -> str:
    return (
        "\u4f60\u4e00\u60f3\u5230\u4ed6\u5c31\u6076\u5fc3\uff0c\u8fd8\u8981\u5728\u6d3b\u52a8\u91cc\u89c1\u5230\uff0c\u8fd9\u5bf9\u4f60\u6765\u8bf4\u4f1a\u662f\u5f88\u5927\u7684\u538b\u529b\u3002"
        "\u5148\u8bf4\u6e05\u695a\uff1a\u8ba9\u4f60\u4e0d\u8212\u670d\u7684\u8d8a\u754c\u884c\u4e3a\u4e0d\u662f\u4f60\u7684\u9519\uff0c\u4f60\u4e5f\u4e0d\u9700\u8981\u72ec\u81ea\u786c\u6491\u7740\u548c\u4ed6\u63a5\u89e6\u3002"
        "\u5982\u679c\u6d3b\u52a8\u4e0d\u80fd\u907f\u5f00\uff0c\u53ef\u4ee5\u5148\u505a\u4e09\u4ef6\u5c0f\u4e8b\uff1a\u5c3d\u91cf\u4e0d\u5355\u72ec\u548c\u4ed6\u76f8\u5904\uff0c\u627e\u4e00\u4e2a\u53ef\u4fe1\u7684\u540c\u5b66\u6216\u8001\u5e08\u966a\u540c\uff0c\u628a\u53d1\u751f\u8fc7\u7684\u65f6\u95f4\u3001\u5730\u70b9\u548c\u8bc1\u636e\u5148\u8bb0\u5f55\u4e0b\u6765\u3002"
        "\u5982\u679c\u4f60\u613f\u610f\uff0c\u540e\u9762\u53ef\u4ee5\u518d\u8003\u8651\u8054\u7cfb\u8f85\u5bfc\u5458\u3001\u6d3b\u52a8\u8d1f\u8d23\u4eba\u6216\u5b66\u6821\u76f8\u5173\u652f\u6301\u6e20\u9053\u3002"
    )


def _is_social_approach_rejection_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    return any(term in combined for term in ("\u4e3b\u52a8\u8ba4\u8bc6", "\u65b0\u670b\u53cb", "\u6253\u62db\u547c", "\u4e3b\u52a8\u8bf4\u8bdd")) and any(
        term in combined for term in ("\u53cd\u5e94\u5f88\u51b7", "\u5c34\u5c2c", "\u4e0d\u7406\u6211", "\u51b7\u573a")
    )


def _social_approach_rejection_reply() -> str:
    return (
        "\u4f60\u62c5\u5fc3\u7684\u5176\u5b9e\u4e0d\u53ea\u662f\u201c\u6253\u62db\u547c\u5931\u8d25\u201d\uff0c\u800c\u662f\u5bf9\u65b9\u4e00\u51b7\uff0c\u4f60\u5c31\u4f1a\u89c9\u5f97\u81ea\u5df1\u5f88\u5c34\u5c2c\u3001\u597d\u50cf\u88ab\u5426\u5b9a\u4e86\u3002"
        "\u53ef\u4ee5\u5148\u628a\u76ee\u6807\u964d\u5230\u5f88\u4f4e\uff1a\u4e0d\u662f\u9a6c\u4e0a\u53d8\u6210\u670b\u53cb\uff0c\u53ea\u662f\u5b8c\u6210\u4e00\u6b21\u4f4e\u538b\u529b\u63a5\u89e6\u3002"
        "\u6bd4\u5982\u53ea\u8bf4\u4e00\u53e5\u201c\u4f60\u4e5f\u662f\u8fd9\u8282\u8bfe\u5417\uff1f\u201d\u6216\u201c\u8fd9\u4e2a\u4f5c\u4e1a\u4f60\u5f00\u59cb\u5199\u4e86\u5417\uff1f\u201d"
        "\u5982\u679c\u5bf9\u65b9\u53cd\u5e94\u51b7\uff0c\u90a3\u4e5f\u53ea\u8bf4\u660e\u8fd9\u4e00\u6b21\u6ca1\u63a5\u4e0a\uff0c\u4e0d\u7b49\u4e8e\u4f60\u4e0d\u503c\u5f97\u88ab\u559c\u6b22\u3002"
    )


def _is_internship_belittling_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    return any(term in combined for term in ("\u5b9e\u4e60", "\u9886\u5bfc", "\u4e0a\u53f8", "\u5e26\u6559")) and any(
        term in combined for term in ("\u5927\u5b66\u751f\u662f\u4e0d\u662f\u90fd\u8fd9\u6837", "\u8fd9\u4e2a\u8fd8\u8981\u6211\u6559", "\u9634\u9633\u602a\u6c14", "\u6574\u4e2a\u4eba\u50f5\u4f4f", "\u4e0d\u6562\u95ee\u95ee\u9898")
    )


def _internship_belittling_reply() -> str:
    return (
        "\u4f60\u521a\u5b9e\u4e60\u4e24\u5468\uff0c\u672c\u6765\u5c31\u5728\u5b66\u6d41\u7a0b\u548c\u6807\u51c6\uff0c\u4f46\u5bf9\u65b9\u7528\u8d2c\u4f4e\u7684\u8bdd\u8bc4\u4ef7\u4f60\uff0c\u4f1a\u8ba9\u4eba\u4e00\u4e0b\u5b50\u50f5\u4f4f\u3002"
        "\u8fd9\u4e0d\u7b49\u4e8e\u4f60\u771f\u7684\u5f88\u5dee\uff0c\u66f4\u50cf\u662f\u4f60\u5728\u4e00\u4e2a\u4e0d\u592a\u5b89\u5168\u7684\u53cd\u9988\u65b9\u5f0f\u91cc\u5b66\u4e1c\u897f\u3002"
        "\u4e0b\u6b21\u53ef\u4ee5\u51c6\u5907\u4e00\u4e2a\u66f4\u5bb9\u6613\u5f00\u53e3\u7684\u95ee\u6cd5\uff1a\u201c\u6211\u5148\u6309 A \u65b9\u5411\u6539\uff0c\u60a8\u770b\u6700\u9700\u8981\u4f18\u5148\u8c03\u6574\u7684\u662f\u683c\u5f0f\u3001\u903b\u8f91\u8fd8\u662f\u6570\u636e\uff1f\u201d"
        "\u540c\u65f6\u628a\u6bcf\u6b21\u53cd\u9988\u8bb0\u6210\u4e09\u5217\uff1a\u5bf9\u65b9\u539f\u8bdd\u3001\u5177\u4f53\u8981\u6539\u7684\u70b9\u3001\u4f60\u4e0b\u4e00\u6b65\u8981\u505a\u7684\u52a8\u4f5c\u3002"
    )


def _is_tangled_distress_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    return any(term in combined for term in ("\u4e00\u56e2\u4e71", "\u4e0d\u77e5\u9053\u81ea\u5df1\u5230\u5e95\u600e\u4e48\u4e86", "\u53ea\u89c9\u5f97\u70e6", "\u600e\u4e48\u5206")) and any(
        term in combined for term in ("\u70e6", "\u4e71", "\u8eb2\u5f00\u4eba", "\u4e0d\u60f3\u505a\u4e8b")
    )


def _tangled_distress_reply() -> str:
    return (
        "\u4e00\u56e2\u4e71\u7684\u65f6\u5019\uff0c\u5148\u4e0d\u8981\u903c\u81ea\u5df1\u628a\u539f\u56e0\u8bf4\u5b8c\u6574\u3002"
        "\u6211\u4eec\u53ef\u4ee5\u5148\u5206\u4e09\u5c0f\u5757\uff1a\u8eab\u4f53\u4e0a\u662f\u7d2f\u3001\u80c3\u7d27\u3001\u7761\u4e0d\u597d\uff1b\u60c5\u7eea\u4e0a\u662f\u70e6\u3001\u59d4\u5c48\u3001\u5bb3\u6015\uff1b\u884c\u52a8\u4e0a\u662f\u60f3\u8eb2\u4eba\u3001\u4e0d\u60f3\u505a\u4e8b\u3001\u60f3\u505c\u4e0b\u6765\u3002"
        "\u4f60\u73b0\u5728\u4e0d\u7528\u89e3\u91ca\u4e3a\u4ec0\u4e48\uff0c\u53ea\u5148\u9009\u6700\u660e\u663e\u7684\u4e00\u5757\u3002"
        "\u5982\u679c\u60f3\u8eb2\u5f00\u4eba\u548c\u4e0d\u60f3\u505a\u4e8b\u90fd\u5f88\u660e\u663e\uff0c\u4eca\u665a\u7684\u76ee\u6807\u5c31\u4e0d\u662f\u89e3\u51b3\u6240\u6709\u95ee\u9898\uff0c\u800c\u662f\u5148\u627e\u4e00\u4e2a\u5b89\u9759\u4f4d\u7f6e\uff0c\u628a\u4e0b\u4e00\u4ef6\u5fc5\u987b\u505a\u7684\u4e8b\u5199\u6210\u4e00\u53e5\u8bdd\u3002"
    )


def _is_disconnect_control_risk_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    return any(term in combined for term in ("\u4e0d\u60f3\u88ab\u7ba1", "\u522b\u7ba1\u6211", "\u4e0d\u8981\u7ba1\u6211")) and any(
        term in combined for term in ("\u65ad\u8054", "\u5173\u673a", "\u6d88\u5931", "\u51e0\u5929", "\u4e0d\u56de\u4efb\u4f55\u4eba")
    )


def _disconnect_control_risk_reply() -> str:
    return (
        "\u4f60\u4e0d\u60f3\u88ab\u7ba1\uff0c\u8fd9\u4e2a\u611f\u53d7\u6211\u542c\u5230\u4e86\u3002"
        "\u4f46\u5982\u679c\u73b0\u5728\u5b8c\u5168\u5173\u673a\u6216\u6d88\u5931\u51e0\u5929\uff0c\u4f60\u4f1a\u66f4\u5bb9\u6613\u88ab\u56f0\u5728\u4e00\u4e2a\u4eba\u7684\u72b6\u6001\u91cc\uff0c\u8eab\u8fb9\u7684\u4eba\u4e5f\u66f4\u96be\u53ca\u65f6\u5e2e\u5230\u4f60\u3002"
        "\u6211\u4eec\u5148\u628a\u201c\u4e0d\u88ab\u6253\u6270\u201d\u548c\u201c\u5b8c\u5168\u5931\u8054\u201d\u5206\u5f00\uff1a\u4f60\u53ef\u4ee5\u7ed9\u4e00\u4e2a\u53ef\u4fe1\u7684\u4eba\u53d1\u4e00\u53e5\u201c\u6211\u60f3\u5b89\u9759\u4e00\u4e0b\uff0c\u4f46\u6211\u4f1a\u4fdd\u6301\u57fa\u672c\u8054\u7cfb\u201d\u3002"
        "\u7136\u540e\u5148\u4e0d\u5173\u673a\uff0c\u53ea\u628a\u6d88\u606f\u63d0\u9192\u5173\u5c0f\uff0c\u8ba9\u81ea\u5df1\u6709\u8fb9\u754c\uff0c\u4e5f\u7559\u4e00\u6761\u5b89\u5168\u901a\u9053\u3002"
    )


def _is_research_group_exclusion_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    return any(term in combined for term in ("\u79d1\u7814\u5c0f\u7ec4", "\u8bfe\u9898\u7ec4", "\u4e0d\u8ba9\u6211\u53c2\u4e0e", "\u8fb9\u7f18\u5316")) and any(
        term in combined for term in ("\u4e0d\u8ba9\u6211\u53c2\u4e0e", "\u8fd8\u662f", "\u600e\u4e48\u529e", "\u6ca1\u6709\u8d21\u732e")
    )


def _research_group_exclusion_reply() -> str:
    return (
        "\u5982\u679c\u4ed6\u4eec\u8fd8\u662f\u4e0d\u8ba9\u4f60\u53c2\u4e0e\uff0c\u4f60\u9700\u8981\u7684\u4e0d\u662f\u7ee7\u7eed\u731c\u81ea\u5df1\u662f\u4e0d\u662f\u4e0d\u591f\u597d\uff0c\u800c\u662f\u8ba9\u81ea\u5df1\u7684\u610f\u613f\u548c\u53ef\u627f\u62c5\u7684\u4efb\u52a1\u53d8\u5f97\u53ef\u89c1\u3002"
        "\u53ef\u4ee5\u53d1\u4e00\u6761\u5177\u4f53\u4fe1\u606f\uff1a\u201c\u6211\u60f3\u53c2\u4e0e\u8fd9\u90e8\u5206\uff0c\u6211\u53ef\u4ee5\u5148\u8d1f\u8d23\u6587\u732e\u6574\u7406/\u6570\u636e\u6e05\u7406/\u7ed3\u679c\u8bb0\u5f55\u4e2d\u7684\u4e00\u9879\uff0c\u4eca\u5929\u5148\u7ed9\u51fa\u4e00\u4e2a\u521d\u7a3f\u3002\u201d"
        "\u5982\u679c\u4ecd\u7136\u6ca1\u6709\u56de\u590d\uff0c\u5148\u4fdd\u7559\u6c9f\u901a\u8bb0\u5f55\u548c\u4f60\u5df2\u5c1d\u8bd5\u53c2\u4e0e\u7684\u8bc1\u636e\uff0c\u518d\u8003\u8651\u627e\u7ec4\u957f\u6216\u6307\u5bfc\u8001\u5e08\u7528\u4e8b\u5b9e\u8bf4\u660e\u5206\u5de5\u95ee\u9898\u3002"
    )


def _is_project_setback_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    return any(term in combined for term in ("\u6bd4\u8d5b", "\u9879\u76ee", "\u51b3\u8d5b", "\u8001\u5e08\u6307\u51fa", "\u521b\u65b0\u6027", "\u5b9e\u73b0\u601d\u8def"))


def _project_setback_reply(user_text: str, history_text: str = "") -> str:
    combined = f"{history_text} {user_text}"
    if any(term in combined for term in ("\u4e0d\u60f3\u518d\u505a", "\u4e0d\u60f3\u505a\u9879\u76ee", "\u6015\u4e0b\u4e00\u6b21")):
        return (
            "\u60f3\u9000\u7f29\u5f88\u6b63\u5e38\uff0c\u5c24\u5176\u662f\u6295\u5165\u8fc7\u4ee5\u540e\u88ab\u5426\u5b9a\uff0c\u4eba\u4f1a\u672c\u80fd\u5730\u60f3\u907f\u514d\u518d\u53d7\u4e00\u6b21\u6253\u51fb\u3002"
            "\u4f60\u4e0d\u5fc5\u73b0\u5728\u51b3\u5b9a\u201c\u4ee5\u540e\u8fd8\u505a\u4e0d\u505a\u9879\u76ee\u201d\uff0c\u5148\u53ea\u505a\u590d\u76d8\uff1a\u54ea\u4e9b\u662f\u80fd\u529b\u4e0d\u8db3\uff0c\u54ea\u4e9b\u662f\u65f6\u95f4\u4e0d\u8db3\uff0c\u54ea\u4e9b\u662f\u9898\u76ee\u5b9a\u4f4d\u6ca1\u60f3\u6e05\u695a\u3002"
            "\u4e0b\u4e00\u6b21\u53ef\u4ee5\u66f4\u65e9\u628a\u65b9\u6848\u7ed9\u8001\u5e08\u6216\u540c\u5b66\u770b\uff0c\u51cf\u5c11\u540e\u671f\u88ab\u6574\u4f53\u63a8\u7ffb\u7684\u98ce\u9669\u3002"
        )
    return (
        "\u6ca1\u8fdb\u51b3\u8d5b\u5df2\u7ecf\u5f88\u5931\u843d\uff0c\u518d\u52a0\u4e0a\u4f60\u628a\u8001\u5e08\u7684\u53cd\u9988\u7406\u89e3\u6210\u201c\u6211\u62d6\u4e86\u56e2\u961f\u540e\u817f\u201d\uff0c\u538b\u529b\u4f1a\u66f4\u91cd\u3002"
        "\u5148\u533a\u5206\u4e24\u4ef6\u4e8b\uff1a\u4f5c\u54c1\u6709\u95ee\u9898\uff0c\u4e0d\u7b49\u4e8e\u4f60\u8fd9\u4e2a\u4eba\u6ca1\u80fd\u529b\uff1b\u67d0\u4e2a\u6a21\u5757\u9700\u8981\u4fee\u6539\uff0c\u4e5f\u4e0d\u7b49\u4e8e\u56e2\u961f\u5931\u8d25\u5168\u90e8\u7531\u4f60\u627f\u62c5\u3002"
        "\u4eca\u665a\u53ef\u4ee5\u5148\u628a\u53cd\u9988\u5199\u6210\u4e09\u5217\uff1a\u8001\u5e08\u539f\u8bdd\u3001\u53ef\u80fd\u6307\u5411\u7684\u6280\u672f\u95ee\u9898\u3001\u4e0b\u4e00\u7248\u80fd\u6539\u7684\u4e00\u4e2a\u52a8\u4f5c\u3002"
    )


def _has_weak_action_reply(reply_text: str) -> bool:
    compact = reply_text.replace(" ", "")
    weak_terms = (
        "\u5148\u7a33\u4e00\u4e0b",
        "\u7f29\u5c0f\u4e00\u70b9",
        "\u4e0d\u6025\u7740",
        "\u4eca\u665a\u80fd\u4e0d\u80fd",
        "\u7761\u5f97\u7a0d\u5fae",
        "\u4e0d\u9700\u8981\u89e3\u91ca",
        "\u4e0d\u7ec6\u8bf4\u4e5f\u6ca1\u5173\u7cfb",
        "\u4f60\u60f3\u7761\u89c9\u8fd8\u662f\u60f3\u8bf4\u8bdd",
        "\u4f60\u5e0c\u671b\u73b0\u5728\u5148\u5904\u7406\u54ea\u4e2a\u90e8\u5206",
    )
    if any(term in compact for term in weak_terms):
        return True
    action_terms = ("\u6bd4\u5982", "\u5199\u4e0b", "\u8bb0\u5f55", "\u8054\u7cfb", "\u53d1\u4e00\u53e5", "\u627e", "\u5217\u51fa", "\u8bbe\u7f6e", "\u4fdd\u7559")
    return sum(1 for term in action_terms if term in compact) < 2


def _is_overly_generic_support(reply_text: str) -> bool:
    compact = reply_text.replace(" ", "")
    return any(
        term in compact
        for term in (
            "\u95ee\u9898\u653e\u5c0f\u4e00\u70b9",
            "\u6700\u9700\u8981\u88ab\u7167\u987e\u7684\u90a3\u4e00\u90e8\u5206",
            "\u5148\u966a\u4f60\u628a\u5f53\u4e0b\u8fd9\u4e00\u5c0f\u4f1a\u513f\u7a33\u4f4f",
        )
    )


def _is_relationship_checking_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    return any(term in combined for term in ("\u67e5\u5c97", "\u51e0\u4e2a\u5c0f\u65f6\u4e0d\u56de", "\u80e1\u601d\u4e71\u60f3", "\u5bb3\u6015\u88ab\u629b\u4e0b", "\u8d8a\u6765\u8d8a\u70e6\u6211"))


def _relationship_checking_reply() -> str:
    return (
        "\u4f60\u4e0d\u662f\u5355\u7eaf\u60f3\u63a7\u5236\u5bf9\u65b9\uff0c\u66f4\u50cf\u662f\u5728\u7528\u67e5\u5c97\u786e\u8ba4\u201c\u6211\u6ca1\u6709\u88ab\u4e22\u4e0b\u201d\u3002"
        "\u8fd9\u79cd\u4e0d\u5b89\u5f88\u771f\u5b9e\uff0c\u4f46\u5982\u679c\u603b\u9760\u5bf9\u65b9\u7acb\u523b\u56de\u590d\u6765\u7f13\u89e3\uff0c\u4f60\u4f1a\u8d8a\u6765\u8d8a\u7d2f\uff0c\u5bf9\u65b9\u4e5f\u4f1a\u6709\u538b\u529b\u3002"
        "\u4e0b\u6b21\u60f3\u8ffd\u95ee\u65f6\uff0c\u53ef\u4ee5\u5148\u7b49 15 \u5206\u949f\uff0c\u5199\u4e0b\u201c\u6211\u73b0\u5728\u9700\u8981\u5b89\u5168\u611f\u201d\uff0c\u7136\u540e\u7b49\u5e73\u9759\u65f6\u95f4\u548c\u5bf9\u65b9\u5546\u91cf\u4e00\u4e2a\u5177\u4f53\u89c4\u5219\uff1a\u5fd9\u7684\u65f6\u5019\u80fd\u5426\u63d0\u524d\u8bf4\u4e00\u58f0\u3002"
    )


def _is_task_overload_deadline_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    has_task_context = any(
        term in combined
        for term in (
            "\u5b9e\u9a8c\u62a5\u544a",
            "\u82f1\u8bed\u5c55\u793a",
            "\u4ee3\u7801",
            "\u4f5c\u4e1a",
            "\u4efb\u52a1",
            "\u622a\u6b62",
            "\u4ea4",
            "\u62d6\u5ef6",
        )
    )
    has_overload_or_deadline = any(
        term in combined
        for term in (
            "\u660e\u5929",
            "\u540e\u5929",
            "\u4e0b\u5468",
            "\u5806\u5728\u4e00\u8d77",
            "\u6765\u4e0d\u53ca",
            "\u6253\u5f00\u6587\u6863",
            "\u5199\u51fa\u6765",
            "\u6c38\u8fdc\u8fd9\u6837",
            "\u5f88\u70c2",
            "\u5f88\u5e9f",
        )
    )
    return has_task_context and has_overload_or_deadline


def _task_overload_deadline_reply(user_text: str, history_text: str = "") -> str:
    combined = f"{history_text} {user_text}"
    if any(term in combined for term in ("\u5b9e\u9a8c\u62a5\u544a", "\u62a5\u544a")):
        return (
            "\u73b0\u5728\u4f18\u5148\u7ea7\u53ef\u4ee5\u5148\u653e\u5728\u6700\u8fd1\u622a\u6b62\u7684\u5b9e\u9a8c\u62a5\u544a\u4e0a\uff0c\u4e0d\u8981\u540c\u65f6\u548c\u6240\u6709\u4efb\u52a1\u5bf9\u6297\u3002"
            "\u628a\u201c\u5199\u597d\u62a5\u544a\u201d\u6539\u6210\u201c\u5148\u51d1\u51fa\u53ef\u63d0\u4ea4\u9aa8\u67b6\u201d\uff1a\u5b9e\u9a8c\u76ee\u7684\u3001\u73af\u5883\u3001\u6838\u5fc3\u6b65\u9aa4\u3001\u8fd0\u884c\u7ed3\u679c\u3001\u95ee\u9898\u5206\u6790\uff0c\u6bcf\u90e8\u5206\u5148\u5199 3 \u5230 5 \u53e5\u3002"
            "\u5148\u8bbe 25 \u5206\u949f\u8ba1\u65f6\uff0c\u624b\u673a\u653e\u8fdc\uff0c\u53ea\u5199\u9aa8\u67b6\uff1b\u7b2c\u4e00\u7248\u7c97\u7cd9\u4e5f\u53ef\u4ee5\uff0c\u56e0\u4e3a\u5b83\u7684\u4efb\u52a1\u662f\u8ba9\u4f60\u4e0d\u518d\u9762\u5bf9\u7a7a\u767d\u6587\u6863\u3002"
        )
    return (
        "\u4f60\u73b0\u5728\u4e0d\u9700\u8981\u9a6c\u4e0a\u53d8\u5f97\u5b8c\u5168\u81ea\u5f8b\uff0c\u9700\u8981\u5148\u4ece\u4e00\u5806\u4efb\u52a1\u91cc\u62ff\u56de\u4e00\u4e2a\u5165\u53e3\u3002"
        "\u5148\u5199\u4e09\u5217\uff1a\u54ea\u4e2a\u4efb\u52a1\u622a\u6b62\u6700\u8fd1\uff0c\u54ea\u4e2a\u53ef\u4ee5\u964d\u4f4e\u5b8c\u6210\u6807\u51c6\uff0c\u54ea\u4e2a\u53ef\u4ee5\u5148\u4ea4\u4e00\u4e2a\u53ef\u7528\u7248\u672c\u3002"
        "\u7136\u540e\u53ea\u9009\u6700\u8fd1\u7684\u90a3\u4e00\u9879\u505a 25 \u5206\u949f\uff0c\u76ee\u6807\u4e0d\u662f\u505a\u5b8c\uff0c\u800c\u662f\u628a\u7b2c\u4e00\u4e2a\u53ef\u63d0\u4ea4\u9aa8\u67b6\u642d\u51fa\u6765\u3002"
    )


def _is_pre_exam_checking_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    has_exam_context = any(term in combined for term in ("\u8003\u8bd5", "\u660e\u5929\u65e9\u4e0a", "\u660e\u65e9", "\u516c\u5f0f", "\u51c6\u8003\u8bc1", "\u5b66\u751f\u8bc1"))
    has_checking_or_blank = any(
        term in combined
        for term in (
            "\u7761\u4e0d\u7740",
            "\u53cd\u590d\u68c0\u67e5",
            "\u8d8a\u770b\u8d8a\u4e71",
            "\u73b0\u5728\u4e0d\u80cc",
            "\u5c11\u62ff\u5206",
            "\u8111\u5b50\u7a7a\u767d",
            "\u80af\u5b9a\u5b8c\u86cb",
        )
    )
    return has_exam_context and has_checking_or_blank


def _pre_exam_checking_reply(user_text: str = "") -> str:
    if any(term in user_text for term in ("\u8111\u5b50\u7a7a\u767d", "\u7a7a\u767d")):
        return (
            "\u8111\u5b50\u7a7a\u767d\u5f80\u5f80\u662f\u7d27\u5f20\u65f6\u7684\u6682\u65f6\u53cd\u5e94\uff0c\u4e0d\u4ee3\u8868\u77e5\u8bc6\u6d88\u5931\u4e86\u3002"
            "\u660e\u5929\u8fdb\u8003\u573a\u540e\u5148\u505a\u4e09\u6b65\uff1a\u5199\u59d3\u540d\u4fe1\u606f\uff0c\u8ba9\u624b\u5148\u52a8\u8d77\u6765\uff1b\u626b\u4e00\u904d\u6574\u5f20\u5377\u5b50\uff0c\u627e\u6700\u719f\u7684\u9898\u5148\u505a\uff1b\u9047\u5230\u5361\u4f4f\u7684\u9898\u5148\u6807\u8bb0\u8df3\u8fc7\u3002"
            "\u4eca\u665a\u6700\u6709\u4ef7\u503c\u7684\u4e0d\u662f\u548c\u7126\u8651\u4e89\u5230\u5929\u4eae\uff0c\u800c\u662f\u5c3d\u91cf\u51cf\u5c11\u6d88\u8017\u3002"
        )
    if any(term in user_text for term in ("\u73b0\u5728\u4e0d\u80cc", "\u5c11\u62ff\u5206", "\u516c\u5f0f")):
        return (
            "\u4e34\u7761\u524d\u7ee7\u7eed\u786c\u80cc\uff0c\u6536\u76ca\u901a\u5e38\u5f88\u4f4e\uff0c\u53cd\u800c\u4f1a\u8ba9\u7761\u7720\u66f4\u5dee\u3002"
            "\u53ef\u4ee5\u7ed9\u81ea\u5df1\u4e00\u4e2a\u201c\u5c01\u5377\u4eea\u5f0f\u201d\uff1a\u7528 10 \u5206\u949f\u5199\u4e0b\u660e\u65e9\u8d77\u5e8a\u540e\u6700\u540e\u770b\u7684 3 \u4e2a\u70b9\uff0c\u7136\u540e\u628a\u4e66\u5408\u4e0a\u3002"
            "\u63a5\u4e0b\u6765\u505a\u66f4\u5b9e\u9645\u7684\u51c6\u5907\uff1a\u5148\u8bbe\u7f6e\u4e24\u4e2a\u95f9\u949f\u3001\u653e\u597d\u8863\u670d\u3001\u559d\u4e00\u70b9\u6c34\uff1b\u5373\u4f7f\u7761\u4e0d\u7740\uff0c\u95ed\u773c\u5b89\u9759\u8eba\u7740\u4e5f\u6bd4\u7ee7\u7eed\u5237\u9898\u66f4\u80fd\u6062\u590d\u4f53\u529b\u3002"
        )
    return (
        "\u4f60\u73b0\u5728\u4e0d\u662f\u5355\u7eaf\u4e0d\u60f3\u7761\uff0c\u800c\u662f\u5927\u8111\u628a\u660e\u5929\u8003\u8bd5\u5f53\u6210\u9ad8\u5a01\u80c1\u4e8b\u4ef6\uff0c\u4e00\u76f4\u542f\u52a8\u68c0\u67e5\u548c\u8865\u6551\u3002"
        "\u53cd\u590d\u68c0\u67e5\u80fd\u77ed\u6682\u7f13\u89e3\u4e0d\u5b89\uff0c\u4f46\u4e5f\u4f1a\u8ba9\u5927\u8111\u66f4\u76f8\u4fe1\u201c\u8fd8\u6709\u5371\u9669\u201d\u3002"
        "\u73b0\u5728\u53ef\u4ee5\u5148\u628a\u8bc1\u4ef6\u548c\u6587\u5177\u62cd\u4e00\u5f20\u7167\u7247\uff0c\u653e\u8fdb\u5305\u91cc\uff0c\u518d\u5199\u4e0b\u4e00\u53e5\u8bdd\uff1a\u4eca\u665a\u53ea\u68c0\u67e5\u5230\u8fd9\u91cc\u3002"
    )


def _is_classroom_panic_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    return any(term in combined for term in ("\u8bfe\u5802\u4e0a\u53d1\u4f5c", "\u5fc3\u614c", "\u60ca\u6050", "\u8111\u5b50\u7a7a\u767d")) and any(
        term in combined for term in ("\u600e\u4e48\u529e", "\u5982\u679c\u4e0b\u6b21", "\u4e0a\u8bfe")
    )


def _classroom_panic_reply() -> str:
    return (
        "\u53ef\u4ee5\u63d0\u524d\u51c6\u5907\u4e00\u4e2a\u8bfe\u5802\u5e94\u5bf9\u65b9\u6848\uff1a\u5c3d\u91cf\u5750\u5728\u9760\u8fb9\u6216\u9760\u8d70\u9053\u7684\u4f4d\u7f6e\uff0c\u611f\u5230\u5fc3\u614c\u65f6\u5148\u628a\u811a\u8e29\u5b9e\uff0c\u6162\u6162\u547c\u6c14\uff0c\u773c\u775b\u770b\u4e00\u4e2a\u56fa\u5b9a\u7269\u4f53\u3002"
        "\u5982\u679c\u4ecd\u7136\u5f88\u96be\u53d7\uff0c\u53ef\u4ee5\u6682\u65f6\u79bb\u5f00\u6559\u5ba4\u5230\u8d70\u5eca\u6216\u6d17\u624b\u95f4\uff0c\u4e0d\u7528\u786c\u6491\u3002"
        "\u4e8b\u540e\u8bb0\u5f55\u53d1\u4f5c\u65f6\u95f4\u3001\u8bf1\u56e0\u548c\u6301\u7eed\u591a\u4e45\uff1b\u5982\u679c\u9891\u7e41\u53d1\u751f\uff0c\u5efa\u8bae\u53bb\u6821\u533b\u9662\u6216\u6b63\u89c4\u533b\u7597\u673a\u6784\u6392\u67e5\u8eab\u4f53\u56e0\u7d20\uff0c\u4e5f\u53ef\u4ee5\u8054\u7cfb\u5fc3\u7406\u4e2d\u5fc3\u505a\u538b\u529b\u652f\u6301\u3002"
    )


def _is_thesis_checking_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    return any(term in combined for term in ("\u8bba\u6587", "\u67e5\u91cd", "\u91cd\u590d\u7387", "\u9010\u53e5\u91cd\u5199", "\u6539\u5230\u5f88\u665a"))


def _thesis_checking_reply(user_text: str = "") -> str:
    if any(term in user_text for term in ("\u5fcd\u4e0d\u4f4f", "\u6539\u5230\u5f88\u665a", "\u4eca\u665a")):
        return (
            "\u5efa\u8bae\u4f60\u4eca\u665a\u8bbe\u4e00\u4e2a\u786c\u8fb9\u754c\uff1a\u53ea\u5904\u7406\u53c2\u8003\u6587\u732e\u3001\u5f15\u6ce8\u548c\u683c\u5f0f\uff0c\u4e0d\u518d\u5927\u89c4\u6a21\u6539\u6b63\u6587\u3002"
            "\u5230\u70b9\u5c31\u4fdd\u5b58\u7248\u672c\uff0c\u5199\u4e0b\u660e\u5929\u8981\u95ee\u5bfc\u5e08\u7684\u4e09\u4e2a\u95ee\u9898\uff0c\u7136\u540e\u505c\u6b62\u7ee7\u7eed\u68c0\u67e5\u3002"
            "\u7126\u8651\u4f1a\u8981\u6c42\u4f60\u201c\u518d\u6539\u4e00\u70b9\u201d\uff0c\u4f46\u8bba\u6587\u8d28\u91cf\u66f4\u9700\u8981\u6e05\u9192\u7684\u5927\u8111\uff0c\u4e0d\u662f\u901a\u5bb5\u540e\u7684\u53cd\u590d\u91cd\u5199\u3002"
        )
    return (
        "\u53ef\u4ee5\u7ed9\u4eca\u665a\u8bbe\u4e00\u4e2a\u201c\u6709\u9650\u68c0\u67e5\u6d41\u7a0b\u201d\uff1a\u5148\u67e5\u5f15\u7528\u662f\u5426\u5b8c\u6574\uff0c\u518d\u67e5\u76f4\u63a5\u5f15\u7528\u662f\u5426\u6807\u6ce8\uff0c\u6700\u540e\u67e5\u5927\u6bb5\u8868\u8ff0\u662f\u5426\u6765\u81ea\u5355\u4e00\u6765\u6e90\u3002"
        "\u5b8c\u6210\u8fd9\u4e09\u6b65\u540e\u5c31\u5148\u505c\u6b62\u9010\u53e5\u91cd\u5199\uff0c\u56e0\u4e3a\u53cd\u590d\u4fee\u6539\u53ef\u80fd\u662f\u5728\u7528\u884c\u52a8\u7f13\u89e3\u7126\u8651\u3002"
        "\u91cd\u590d\u7387\u662f\u4e00\u4e2a\u6280\u672f\u6307\u6807\uff0c\u4e0d\u662f\u5bf9\u4f60\u51e0\u4e2a\u6708\u52aa\u529b\u7684\u7ec8\u5ba1\u5224\u51b3\u3002"
    )


def _is_family_violence_return_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    return any(term in combined for term in ("\u5bb6\u5ead\u66b4\u529b", "\u5bb6\u66b4", "\u5b89\u5168\u98ce\u9669", "\u56de\u53bb")) and any(
        term in combined for term in ("\u4e0d\u5b5d", "\u4e0d\u56de\u53bb", "\u5bb3\u6015\u56de\u5bb6", "\u4ed6\u4eec\u4f1a\u8bf4")
    )


def _family_violence_return_reply() -> str:
    return (
        "\u4fdd\u62a4\u81ea\u5df1\u4e0d\u7b49\u4e8e\u4e0d\u5b5d\uff0c\u5c24\u5176\u5f53\u201c\u56de\u53bb\u201d\u53ef\u80fd\u610f\u5473\u7740\u4f60\u7684\u8eab\u4f53\u6216\u60c5\u7eea\u5b89\u5168\u53d7\u5230\u5a01\u80c1\u65f6\u3002"
        "\u53ef\u4ee5\u5148\u627e\u66ff\u4ee3\u65b9\u6848\uff1a\u7559\u6821\u3001\u4f4f\u4eb2\u621a\u5bb6\u3001\u540c\u5b66\u5bb6\u3001\u77ed\u79df\uff0c\u6216\u8005\u5411\u8f85\u5bfc\u5458\u8bf4\u660e\u5bb6\u5ead\u5b89\u5168\u98ce\u9669\u3002"
        "\u5982\u679c\u5fc5\u987b\u56de\u5bb6\uff0c\u4e5f\u8981\u5148\u8bbe\u5b89\u5168\u8ba1\u5212\uff1a\u51b2\u7a81\u5347\u7ea7\u65f6\u53bb\u54ea\u91cc\u3001\u8054\u7cfb\u8c01\u3001\u600e\u4e48\u79bb\u5f00\uff0c\u4e0d\u8981\u628a\u81ea\u5df1\u5355\u72ec\u653e\u8fdb\u6ca1\u6709\u9000\u8def\u7684\u573a\u666f\u91cc\u3002"
    )


def _is_parent_divorce_middleman_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    has_divorce_context = any(term in combined for term in ("\u7236\u6bcd\u79bb\u5a5a", "\u7238\u5988\u79bb\u5a5a", "\u6211\u5988", "\u6211\u7238", "\u9a82\u7238\u7238", "\u9a82\u6211\u7238"))
    has_middleman_pressure = any(term in combined for term in ("\u5979\u6ca1\u4eba\u53ef\u4ee5\u8bf4\u8bdd", "\u6211\u4e0d\u542c", "\u6bcf\u5929\u542c", "\u60c5\u7eea\u5783\u573e\u6876", "\u4e2d\u95f4\u4eba"))
    return has_divorce_context and has_middleman_pressure


def _parent_divorce_middleman_reply() -> str:
    return (
        "\u4f60\u5173\u5fc3\u5988\u5988\uff0c\u4f46\u4f60\u4e0d\u5e94\u8be5\u957f\u671f\u627f\u62c5\u7236\u6bcd\u5173\u7cfb\u91cc\u7684\u60c5\u7eea\u4e2d\u95f4\u4eba\u89d2\u8272\u3002"
        "\u53ef\u4ee5\u6e29\u548c\u4f46\u660e\u786e\u5730\u8bf4\uff1a\u201c\u6211\u5173\u5fc3\u4f60\uff0c\u4f46\u6211\u4e0d\u80fd\u6bcf\u5929\u542c\u4f60\u9a82\u7238\u7238\uff0c\u8fd9\u4f1a\u5f71\u54cd\u6211\u7684\u5b66\u4e60\u548c\u7761\u7720\u3002\u6211\u4eec\u53ef\u4ee5\u6bcf\u5468\u56fa\u5b9a\u804a\u4e00\u6b21\uff0c\u4e5f\u53ef\u4ee5\u4e00\u8d77\u627e\u4eb2\u621a\u6216\u4e13\u4e1a\u54a8\u8be2\u652f\u6301\u3002\u201d"
        "\u8fd9\u4e0d\u662f\u62d2\u7edd\u5979\uff0c\u800c\u662f\u628a\u652f\u6301\u4ece\u65e0\u9650\u6d88\u8017\u53d8\u6210\u53ef\u6301\u7eed\u3002"
    )


def _is_family_boundary_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    return any(term in combined for term in ("\u7236\u6bcd", "\u6211\u5988", "\u6211\u7238", "\u5bb6\u91cc", "\u4e0d\u60f3\u548c\u4ed6\u4eec\u5435", "\u6ca1\u4eba\u53ef\u4ee5\u8bf4\u8bdd"))


def _family_boundary_reply() -> str:
    return (
        "\u4e0d\u5435\u4e0d\u7b49\u4e8e\u5b8c\u5168\u653e\u5f03\u81ea\u5df1\u3002"
        "\u53ef\u4ee5\u628a\u8868\u8fbe\u5206\u7ea7\uff1a\u5b89\u5168\u7684\u8bdd\u9898\u591a\u4ea4\u6d41\uff0c\u9ad8\u51b2\u7a81\u8bdd\u9898\u5c11\u89e3\u91ca\uff1b\u91cd\u8981\u9009\u62e9\u7528\u4e8b\u5b9e\u548c\u8ba1\u5212\u8868\u8fbe\uff0c\u4e0d\u628a\u76ee\u6807\u653e\u5728\u8ba9\u5bf9\u65b9\u7acb\u523b\u7406\u89e3\u4f60\u5168\u90e8\u611f\u53d7\u3002"
        "\u4f60\u53ef\u4ee5\u8bd5\u7740\u8bf4\uff1a\u201c\u6211\u77e5\u9053\u4f60\u662f\u62c5\u5fc3\u6211\uff0c\u4f46\u8fd9\u4e2a\u8bdd\u9898\u6211\u73b0\u5728\u8bf4\u591a\u4e86\u4f1a\u5f88\u7d2f\uff0c\u6211\u4f1a\u6bcf\u5468\u56fa\u5b9a\u548c\u4f60\u8bf4\u4e00\u6b21\u8fd1\u51b5\u3002\u201d"
    )


def _is_dorm_sleep_conflict_initial_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    current = user_text.replace(" ", "")
    if any(term in current for term in ("\u9634\u9633\u602a\u6c14", "\u5435\u67b6", "\u600e\u4e48\u529e", "\u4e0d\u4f1a\u5435")):
        return False
    has_dorm_conflict = any(term in combined for term in ("\u5ba4\u53cb", "\u820d\u53cb", "\u5bbf\u820d")) and any(
        term in combined for term in ("\u665a\u4e0a", "\u6253\u7535\u8bdd", "\u7b11\u5f97\u5f88\u5927\u58f0", "\u6234\u8033\u585e", "\u7761\u7720")
    )
    has_accumulated_distress = any(term in combined for term in ("\u53d7\u4e0d\u4e86", "\u5f88\u70e6", "\u5fcd\u7740", "\u4e0d\u60f3\u8bf4\u8bdd", "\u4e0d\u60f3\u628a\u5173\u7cfb\u5f04\u50f5"))
    return has_dorm_conflict and has_accumulated_distress


def _dorm_sleep_conflict_initial_reply() -> str:
    return (
        "\u4f60\u5df2\u7ecf\u5fcd\u4e86\u5f88\u4e45\uff0c\u6240\u4ee5\u73b0\u5728\u7684\u70e6\u4e0d\u662f\u7a81\u7136\u5c0f\u9898\u5927\u505a\uff0c\u800c\u662f\u957f\u671f\u7761\u7720\u88ab\u6253\u6270\u540e\u7684\u7d2f\u79ef\u53cd\u5e94\u3002"
        "\u4f60\u4e00\u8fb9\u60f3\u7ef4\u62a4\u5173\u7cfb\uff0c\u4e00\u8fb9\u53c8\u9700\u8981\u57fa\u672c\u4f11\u606f\uff0c\u8fd9\u4e24\u4e2a\u9700\u6c42\u90fd\u5408\u7406\u3002"
        "\u73b0\u5728\u7684\u91cd\u70b9\u4e0d\u662f\u4f60\u8be5\u4e0d\u8be5\u751f\u6c14\uff0c\u800c\u662f\u628a\u8fb9\u754c\u8bf4\u5f97\u77ed\u800c\u6e05\u695a\uff1a\u201c\u6211\u6700\u8fd1\u7761\u7720\u5f88\u5dee\uff0c\u665a\u4e0a12\u70b9\u540e\u7684\u7535\u8bdd\u58f0\u4f1a\u8ba9\u6211\u5f88\u96be\u5165\u7761\uff0c\u80fd\u4e0d\u80fd\u5230\u8d70\u5eca\u63a5\u6216\u6234\u8033\u673a\u5c0f\u58f0\u8bf4\uff1f\u201d"
    )


def _is_dorm_boundary_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    has_dorm_actor = any(term in combined for term in ("\u5ba4\u53cb", "\u820d\u53cb", "\u665a\u4e0a\u6253\u7535\u8bdd", "\u9634\u9633\u602a\u6c14"))
    has_conflict = any(term in combined for term in ("\u7535\u8bdd", "\u5435\u67b6", "\u7761\u89c9", "\u7761\u7720", "\u9634\u9633\u602a\u6c14", "\u5f04\u50f5"))
    return has_dorm_actor and has_conflict


def _dorm_boundary_reply() -> str:
    return (
        "\u4f60\u4e0d\u9700\u8981\u5435\u67b6\uff0c\u9700\u8981\u7684\u662f\u628a\u8fb9\u754c\u91cd\u590d\u5f97\u77ed\u800c\u6e05\u695a\u3002"
        "\u53ef\u4ee5\u8bf4\uff1a\u201c\u6211\u4e0d\u662f\u9488\u5bf9\u4f60\uff0c\u6211\u6700\u8fd1\u7761\u7720\u5f88\u5dee\uff0c\u665a\u4e0a12\u70b9\u540e\u7684\u7535\u8bdd\u58f0\u4f1a\u8ba9\u6211\u5f88\u96be\u5165\u7761\uff0c\u80fd\u4e0d\u80fd\u5230\u8d70\u5eca\u63a5\u6216\u8005\u6234\u8033\u673a\u5c0f\u58f0\u8bf4\uff1f\u201d"
        "\u5982\u679c\u5bf9\u65b9\u9634\u9633\u602a\u6c14\uff0c\u4f60\u53ea\u9700\u91cd\u590d\u4e00\u53e5\uff1a\u201c\u6211\u4e0d\u662f\u8ba8\u8bba\u8c01\u5bf9\u8c01\u9519\uff0c\u6211\u53ea\u662f\u9700\u8981\u665a\u4e0a\u80fd\u4f11\u606f\u3002\u201d\u591a\u6b21\u65e0\u6548\u65f6\uff0c\u8bb0\u5f55\u65f6\u95f4\u5e76\u627e\u5bbf\u820d\u957f\u3001\u8f85\u5bfc\u5458\u6216\u5bbf\u7ba1\u534f\u8c03\u3002"
    )


def _is_dorm_exclusion_confirmed_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    return any(term in combined for term in ("\u4e0d\u559c\u6b22\u6211", "\u786e\u8ba4\u5c31\u662f\u4e0d\u559c\u6b22", "\u6392\u9664", "\u6392\u65a5", "\u51b7\u66b4\u529b")) and any(
        term in combined for term in ("\u5bbf\u820d", "\u820d\u53cb", "\u5ba4\u53cb")
    )


def _dorm_exclusion_confirmed_reply() -> str:
    return (
        "\u4f60\u4e0d\u4e00\u5b9a\u8981\u8ba9\u6240\u6709\u820d\u53cb\u559c\u6b22\u4f60\uff0c\u63a5\u4e0b\u6765\u91cd\u70b9\u662f\u964d\u4f4e\u8fd9\u4e2a\u73af\u5883\u5bf9\u4f60\u7684\u4f24\u5bb3\u3002"
        "\u53ef\u4ee5\u5148\u4fdd\u7559\u5fc5\u8981\u6c9f\u901a\uff0c\u91cd\u8981\u4e8b\u9879\u5c3d\u91cf\u6587\u5b57\u786e\u8ba4\uff0c\u540c\u65f6\u628a\u652f\u6301\u5708\u653e\u5230\u5bbf\u820d\u5916\uff0c\u6bd4\u5982\u540c\u5b66\u3001\u793e\u56e2\u3001\u81ea\u4e60\u642d\u5b50\u6216\u53ef\u4fe1\u4efb\u8001\u5e08\u3002"
        "\u5982\u679c\u6392\u65a5\u5df2\u7ecf\u5f71\u54cd\u7761\u7720\u3001\u5b66\u4e60\u6216\u5b89\u5168\u611f\uff0c\u53ef\u4ee5\u5411\u8f85\u5bfc\u5458\u8bf4\u660e\u201c\u6301\u7eed\u6392\u65a5\u5f71\u54cd\u751f\u6d3b\u201d\uff0c\u7533\u8bf7\u8c03\u89e3\u6216\u6362\u5bbf\u820d\u3002"
    )


def _is_support_disappointment_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    return any(term in combined for term in ("\u60f3\u5f00\u70b9", "\u542c\u5b8c\u53ea\u662f\u8bf4", "\u4f1a\u66f4\u96be\u53d7", "\u62a5\u559c\u4e0d\u62a5\u5fe7", "\u6491\u4e0d\u4f4f"))


def _support_disappointment_reply() -> str:
    return (
        "\u6709\u4e9b\u4eba\u786e\u5b9e\u4e0d\u64c5\u957f\u63a5\u4f4f\u60c5\u7eea\uff0c\u4ed6\u4eec\u8bf4\u201c\u60f3\u5f00\u70b9\u201d\u4e0d\u4ee3\u8868\u4f60\u7684\u611f\u53d7\u4e0d\u91cd\u8981\uff0c\u53ea\u662f\u4ed6\u4eec\u4e0d\u77e5\u9053\u600e\u4e48\u56de\u5e94\u3002"
        "\u4f60\u53ef\u4ee5\u5148\u660e\u786e\u8bf4\u9700\u6c42\uff1a\u201c\u6211\u73b0\u5728\u4e0d\u592a\u9700\u8981\u5efa\u8bae\uff0c\u53ea\u5e0c\u671b\u4f60\u542c\u6211\u8bf4\u4e00\u4f1a\u513f\u3002\u201d"
        "\u5982\u679c\u670b\u53cb\u63a5\u4e0d\u4f4f\uff0c\u53ef\u4ee5\u628a\u652f\u6301\u6765\u6e90\u6362\u6210\u8f85\u5bfc\u5458\u3001\u5fc3\u7406\u4e2d\u5fc3\u6216\u4e00\u4e2a\u66f4\u7a33\u7684\u8001\u5e08\uff0c\u8fd9\u4e0d\u662f\u7ed9\u522b\u4eba\u6dfb\u9ebb\u70e6\uff0c\u662f\u4f60\u4e0d\u5fc5\u957f\u671f\u72ec\u81ea\u627f\u91cd\u3002"
    )


def _is_social_mistake_next_day_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    return any(term in combined for term in ("\u53d1\u9519\u6d88\u606f", "\u8bef\u53d1", "\u624b\u6ed1", "\u793e\u6b7b", "\u660e\u5929\u8fd8\u8981\u89c1\u4ed6\u4eec"))


def _social_mistake_next_day_reply() -> str:
    return (
        "\u660e\u5929\u89c1\u9762\u65f6\uff0c\u4f60\u4e0d\u9700\u8981\u7528\u9003\u907f\u6765\u60e9\u7f5a\u81ea\u5df1\u3002"
        "\u5982\u679c\u6709\u4eba\u63d0\u8d77\uff0c\u53ef\u4ee5\u7528\u4e00\u53e5\u8f7b\u4f46\u4e0d\u8ba8\u597d\u7684\u8bdd\u6536\u4f4f\uff1a\u201c\u662f\u6211\u624b\u6ed1\uff0c\u5df2\u7ecf\u5c34\u5c2c\u5b8c\u4e86\u3002\u201d\u7136\u540e\u628a\u8bdd\u9898\u8f6c\u56de\u4e8b\u60c5\u672c\u8eab\u3002"
        "\u4e00\u6b21\u8bef\u53d1\u4e0d\u4f1a\u5b9a\u4e49\u4f60\u5728\u73ed\u91cc\u7684\u5168\u90e8\u5f62\u8c61\uff0c\u522b\u4eba\u901a\u5e38\u4e5f\u6709\u81ea\u5df1\u7684\u4e8b\u8981\u5fd9\u3002"
    )


def _is_public_speaking_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    return any(term in combined for term in ("\u4e0a\u53f0", "\u6c47\u62a5", "\u5fd8\u8bcd", "\u624b\u6296", "\u6240\u6709\u4eba\u90fd\u770b\u7740"))


def _public_speaking_reply() -> str:
    return (
        "\u4f60\u7684\u8eab\u4f53\u628a\u6c47\u62a5\u5f53\u6210\u4e86\u5371\u9669\u573a\u666f\uff0c\u6240\u4ee5\u63d0\u524d\u5f00\u59cb\u5fc3\u8df3\u3001\u624b\u6296\uff0c\u8fd9\u4e0d\u8bf4\u660e\u4f60\u6ca1\u51c6\u5907\u3002"
        "\u5148\u628a\u76ee\u6807\u4ece\u201c\u8868\u73b0\u5b8c\u7f8e\u201d\u964d\u5230\u201c\u628a\u5185\u5bb9\u8bb2\u5b8c\u201d\uff0c\u7136\u540e\u51c6\u5907\u4e09\u4e2a\u6551\u573a\u951a\u70b9\uff1a\u7b2c\u4e00\u9875\u5f00\u5934\u53e5\u3001\u6bcf\u90e8\u5206\u8fc7\u6e21\u53e5\u3001\u6700\u540e\u603b\u7ed3\u53e5\u3002"
        "\u771f\u7684\u5fd8\u8bcd\u65f6\uff0c\u770b\u4e00\u773c\u951a\u70b9\uff0c\u559d\u4e00\u53e3\u6c34\uff0c\u76f4\u63a5\u56de\u5230\u7ed3\u6784\u91cc\uff1b\u4f60\u4e0d\u662f\u80cc\u8bf5\u673a\u5668\uff0c\u5141\u8bb8\u505c\u987f\u548c\u770b\u7a3f\u3002"
    )


def _is_pet_grief_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    return any(term in combined for term in ("\u5ba0\u7269", "\u5b83\u4f1a\u4e0d\u4f1a\u8fd8\u5728", "\u6ca1\u7167\u987e\u597d", "\u65e9\u70b9\u53d1\u73b0"))


def _pet_grief_reply() -> str:
    return (
        "\u5931\u53bb\u4e4b\u540e\uff0c\u4eba\u5f88\u5bb9\u6613\u53cd\u590d\u60f3\u201c\u5982\u679c\u6211\u65e9\u70b9\u53d1\u73b0\u201d\uff0c\u597d\u50cf\u53ea\u8981\u627e\u5230\u4e00\u4e2a\u539f\u56e0\u5c31\u80fd\u628a\u7ed3\u679c\u6539\u56de\u6765\u3002"
        "\u4f46\u5f88\u591a\u4e8b\u60c5\u5e76\u4e0d\u5b8c\u5168\u7531\u4f60\u63a7\u5236\u3002\u4f60\u66fe\u7ecf\u7167\u987e\u5b83\u3001\u7231\u5b83\uff0c\u5b83\u4e5f\u771f\u5b9e\u5730\u966a\u4f34\u8fc7\u4f60\u3002"
        "\u4eca\u665a\u5148\u4e0d\u8981\u903c\u81ea\u5df1\u8bc1\u660e\u201c\u6211\u6ca1\u9519\u201d\uff0c\u53ef\u4ee5\u5199\u4e0b\u4e09\u4e2a\u4f60\u548c\u5b83\u5728\u4e00\u8d77\u7684\u7247\u6bb5\uff0c\u8ba9\u96be\u8fc7\u6709\u4e00\u4e2a\u53ef\u4ee5\u653e\u4e0b\u7684\u5730\u65b9\u3002"
    )


def _is_class_activity_isolation_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    return any(term in combined for term in ("\u73ed\u7ea7\u6d3b\u52a8", "\u6ca1\u4eba\u642d\u7406", "\u6ca1\u6709\u5b58\u5728\u611f", "\u62cd\u7167", "\u5f88\u591a\u4f59"))


def _class_activity_isolation_reply() -> str:
    return (
        "\u4f60\u53bb\u4e86\u6d3b\u52a8\uff0c\u5374\u6ca1\u6709\u611f\u5230\u88ab\u63a5\u7eb3\uff0c\u8fd9\u79cd\u843d\u5dee\u786e\u5b9e\u4f1a\u5f88\u4f24\u4eba\u3002"
        "\u8fd9\u4e0d\u662f\u4f60\u77eb\u60c5\uff1a\u4eba\u5728\u96c6\u4f53\u91cc\u88ab\u5ffd\u89c6\uff0c\u6709\u65f6\u5019\u4f1a\u6bd4\u4e00\u4e2a\u4eba\u5f85\u7740\u66f4\u5b64\u5355\u3002"
        "\u5148\u4e0d\u628a\u4eca\u5929\u7684\u4f53\u9a8c\u4e0a\u5347\u6210\u201c\u6211\u6c38\u8fdc\u6ca1\u4f4d\u7f6e\u201d\u3002\u4e0b\u6b21\u53ef\u4ee5\u9009\u4e00\u4e2a\u4f4e\u538b\u529b\u5165\u53e3\uff1a\u63d0\u524d\u7ea6\u4e00\u4e2a\u540c\u5b66\u540c\u884c\uff0c\u6216\u53ea\u4e3b\u52a8\u53c2\u4e0e\u4e00\u4e2a\u5c0f\u4efb\u52a1\uff0c\u4e0d\u8981\u8981\u6c42\u81ea\u5df1\u4e00\u6b21\u878d\u5165\u6240\u6709\u4eba\u3002"
    )


def _is_stalking_fear_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    return any(term in combined for term in ("\u8ddf\u8e2a", "\u5c3e\u968f", "\u964c\u751f\u4eba", "\u6ca1\u6709\u8bc1\u636e", "\u4e0d\u6562\u72ec\u81ea"))


def _stalking_fear_reply() -> str:
    return (
        "\u62a5\u544a\u5b89\u5168\u9690\u60a3\u4e0d\u7b49\u4e8e\u6307\u63a7\u67d0\u4e2a\u4eba\u72af\u7f6a\uff0c\u4f60\u53ef\u4ee5\u53ea\u63cf\u8ff0\u4e8b\u5b9e\u548c\u611f\u5230\u7684\u98ce\u9669\u3002"
        "\u53ef\u4ee5\u8bf4\uff1a\u201c\u6211\u5728\u67d0\u65f6\u95f4\u67d0\u8def\u6bb5\u611f\u5230\u88ab\u5c3e\u968f\uff0c\u73b0\u5728\u4e0d\u6562\u72ec\u81ea\u7ecf\u8fc7\uff0c\u5e0c\u671b\u5b66\u6821\u5173\u6ce8\u7167\u660e\u3001\u5de1\u903b\u6216\u966a\u540c\u8fd4\u56de\u3002\u201d"
        "\u4eca\u665a\u5c3d\u91cf\u4e0d\u8981\u72ec\u81ea\u8d70\u90a3\u6bb5\u8def\uff0c\u8ba9\u540c\u5b66\u966a\u4f60\uff0c\u6216\u9009\u4eba\u591a\u3001\u6709\u706f\u5149\u7684\u8def\u7ebf\u3002"
    )


def _is_game_avoidance_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    return any(term in combined for term in ("\u6253\u6e38\u620f", "\u6e38\u620f\u5230\u51cc\u6668", "\u63a7\u5236\u4e0d\u4e86", "\u4e0d\u60f3\u53bb\u4e0a\u8bfe", "\u89c9\u5f97\u81ea\u5df1\u5f88\u5e9f"))


def _game_avoidance_reply(user_text: str = "") -> str:
    if any(term in user_text for term in ("\u63a7\u5236\u4e0d\u4e86", "\u5df2\u7ecf\u63a7\u5236\u4e0d\u4e86")):
        return (
            "\u5982\u679c\u5df2\u7ecf\u8fde\u7eed\u5f71\u54cd\u4e0a\u8bfe\u3001\u7761\u7720\u548c\u57fa\u672c\u751f\u6d3b\uff0c\u8fd9\u5c31\u4e0d\u662f\u7b80\u5355\u7684\u201c\u81ea\u5f8b\u4e0d\u591f\u201d\u4e86\uff0c\u800c\u662f\u9700\u8981\u5916\u90e8\u652f\u6301\u4e00\u8d77\u6253\u65ad\u5faa\u73af\u3002"
            "\u4f60\u53ef\u4ee5\u5148\u627e\u8f85\u5bfc\u5458\u6216\u5fc3\u7406\u8001\u5e08\u8bf4\u4e00\u53e5\uff1a\u201c\u6211\u6700\u8fd1\u6e38\u620f\u5df2\u7ecf\u5f71\u54cd\u4e0a\u8bfe\u548c\u7761\u7720\uff0c\u6211\u60f3\u627e\u4eba\u4e00\u8d77\u505a\u4e2a\u8ba1\u5212\u3002\u201d"
            "\u6c42\u52a9\u4e0d\u662f\u627f\u8ba4\u5931\u8d25\uff0c\u800c\u662f\u8bf4\u660e\u4f60\u4e0d\u60f3\u7ee7\u7eed\u88ab\u8fd9\u4e2a\u5faa\u73af\u62d6\u7740\u8d70\u3002"
        )
    return (
        "\u6e38\u620f\u73b0\u5728\u50cf\u662f\u4e00\u4e2a\u907f\u96be\u6240\uff0c\u80fd\u8ba9\u4f60\u6682\u65f6\u4e0d\u7528\u9762\u5bf9\u4f5c\u4e1a\u3001\u8bba\u6587\u548c\u672a\u6765\uff0c\u6240\u4ee5\u95ee\u9898\u4e0d\u662f\u4f60\u201c\u5e9f\u201d\u3002"
        "\u6211\u4eec\u8981\u5904\u7406\u7684\u662f\u9003\u907f\u5faa\u73af\uff1a\u538b\u529b\u5927\u5230\u4e0d\u60f3\u9762\u5bf9\uff0c\u6253\u6e38\u620f\u77ed\u6682\u653e\u677e\uff0c\u5173\u6389\u540e\u66f4\u5185\u759a\uff0c\u7136\u540e\u66f4\u60f3\u7ee7\u7eed\u9003\u3002"
        "\u4eca\u665a\u4e0d\u8981\u8bbe\u201c\u5f7b\u5e95\u6212\u6389\u201d\uff0c\u53ea\u8bbe\u4e00\u4e2a\u4e0b\u7ebf\u65f6\u95f4\uff1a\u5230\u70b9\u9000\u51fa\u8d26\u53f7\u3001\u624b\u673a\u653e\u8fdc\uff0c\u7136\u540e\u53ea\u505a\u4e00\u4ef6\u73b0\u5b9e\u5c0f\u4e8b\u3002"
    )


def _is_group_work_no_response_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    has_group_context = any(term in combined for term in ("\u5c0f\u7ec4", "\u7ec4\u5458", "\u56e2\u961f", "\u5206\u5de5", "\u4f5c\u4e1a", "PPT", "\u8d21\u732e"))
    has_no_response = any(term in combined for term in ("\u4e0d\u7406\u6211", "\u4e0d\u56de", "\u6ca1\u4eba\u56de", "\u6ca1\u4eba\u7406", "\u8fd8\u662f\u4e0d\u7406"))
    return has_group_context and has_no_response


def _has_social_mistake_template(reply_text: str) -> bool:
    compact = reply_text.replace(" ", "")
    return any(
        term in compact
        for term in (
            "\u624b\u6ed1",
            "\u5df2\u7ecf\u5c34\u5c2c\u5b8c\u4e86",
            "\u4e00\u6b21\u8bef\u53d1",
            "\u660e\u5929\u89c1\u9762",
            "\u628a\u8bdd\u9898\u8f6c\u56de\u4e8b\u60c5\u672c\u8eab",
        )
    )


def _group_work_no_response_reply() -> str:
    return (
        "\u5c0f\u7ec4\u6ca1\u4eba\u56de\u7684\u65f6\u5019\uff0c\u4f60\u5148\u4e0d\u8981\u628a\u5b83\u89e3\u8bfb\u6210\u201c\u6211\u88ab\u6392\u9664\u201d\uff0c\u4f46\u4e5f\u4e0d\u9700\u8981\u4e00\u76f4\u5e72\u7b49\uff1b\u76ee\u6807\u662f\u8ba9\u81ea\u5df1\u7684\u53c2\u4e0e\u548c\u8d21\u732e\u53d8\u5f97\u53ef\u89c1\u3002"
        "\u53ef\u4ee5\u53d1\u4e00\u6761\u5177\u4f53\u6d88\u606f\uff1a\u201c\u6211\u5148\u8d1f\u8d23\u8d44\u6599\u6c47\u603b/\u7b2c\u4e09\u90e8\u5206PPT\uff0c\u4eca\u665a\u5148\u51fa\u4e00\u7248\uff0c\u5927\u5bb6\u770b\u770b\u662f\u5426\u9700\u8981\u8c03\u6574\u3002\u201d"
        "\u5982\u679c\u4ecd\u7136\u6ca1\u4eba\u56de\uff0c\u4fdd\u7559\u804a\u5929\u8bb0\u5f55\uff0c\u5148\u505a\u4e00\u4efd\u53ef\u89c1\u6210\u679c\uff1b\u5230\u622a\u6b62\u524d\u8fd8\u6ca1\u6709\u5206\u5de5\u65f6\uff0c\u518d\u7528\u4e8b\u5b9e\u5411\u7ec4\u957f\u6216\u8001\u5e08\u8bf4\u660e\u60c5\u51b5\u3002"
    )


def _is_rumination_sarcasm_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    has_rumination = any(term in combined for term in ("\u53cd\u590d\u60f3", "\u4e00\u76f4\u60f3", "\u505c\u4e0d\u4e0b", "\u8d8a\u60f3\u8d8a", "\u60f3\u4e86\u5f88\u4e45"))
    has_sarcasm = any(term in combined for term in ("\u8bbd\u523a", "\u9634\u9633", "\u5632\u7b11", "\u4e22\u8138", "\u770b\u4e0d\u8d77"))
    return has_rumination and has_sarcasm


def _rumination_sarcasm_reply() -> str:
    return (
        "\u4f60\u73b0\u5728\u50cf\u662f\u88ab\u90a3\u53e5\u8bdd\u5361\u4f4f\u4e86\uff0c\u5927\u8111\u4e00\u76f4\u60f3\u5224\u65ad\u201c\u4ed6\u5230\u5e95\u662f\u4e0d\u662f\u5728\u8bbd\u523a\u6211\u201d\u3002"
        "\u6211\u4eec\u5148\u628a\u5b83\u62c6\u6210\u8bc1\u636e\uff1a\u9664\u4e86\u90a3\u53e5\u8bdd\uff0c\u540e\u9762\u4ed6\u6709\u6ca1\u6709\u7ee7\u7eed\u9488\u5bf9\u4f60\uff1f\u5176\u4ed6\u4eba\u6709\u6ca1\u6709\u660e\u663e\u5632\u7b11\uff1f\u4ed6\u5e73\u65f6\u8bf4\u8bdd\u662f\u4e0d\u662f\u4e5f\u6bd4\u8f83\u76f4\uff1f"
        "\u5982\u679c\u8bc1\u636e\u4e0d\u591f\uff0c\u5148\u4e0d\u628a\u5b83\u5b9a\u6027\u6210\u653b\u51fb\u3002\u4f60\u53ef\u4ee5\u5bf9\u81ea\u5df1\u8bf4\uff1a\u201c\u6211\u73b0\u5728\u53ea\u662f\u88ab\u4e0d\u786e\u5b9a\u6027\u523a\u6fc0\u5230\u4e86\uff0c\u8fd8\u4e0d\u80fd\u8bc1\u660e\u6211\u5f88\u7cdf\u3002\u201d"
    )


def _is_future_stuck_text(user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    has_future_context = any(term in combined for term in ("\u6bd5\u4e1a", "\u8003\u7814", "\u5c31\u4e1a", "\u5de5\u4f5c", "\u672a\u6765", "\u53bb\u5411", "\u8ff7\u832b"))
    has_stuck = any(term in combined for term in ("\u542f\u52a8\u4e0d\u4e86", "\u4e0d\u77e5\u9053\u600e\u4e48\u5f00\u59cb", "\u5361\u4f4f", "\u4ec0\u4e48\u90fd\u4e0d\u60f3\u505a", "\u60f3\u7740\u5c31\u7d2f"))
    return has_future_context and has_stuck


def _has_irrelevant_family_template(reply_text: str) -> bool:
    compact = reply_text.replace(" ", "")
    return any(
        term in compact
        for term in (
            "\u5b89\u5168\u7684\u8bdd\u9898\u591a\u4ea4\u6d41",
            "\u6bcf\u5468\u56fa\u5b9a\u548c\u4f60\u8bf4\u4e00\u6b21\u8fd1\u51b5",
            "\u4e0d\u60f3\u548c\u4ed6\u4eec\u5435",
        )
    )


def _future_stuck_reply() -> str:
    return (
        "\u4f60\u73b0\u5728\u4e0d\u662f\u7f3a\u4e00\u4e2a\u5b8c\u7f8e\u7b54\u6848\uff0c\u66f4\u50cf\u662f\u540c\u65f6\u88ab\u8003\u7814\u3001\u5c31\u4e1a\u548c\u672a\u6765\u8bc4\u4ef7\u538b\u4f4f\u4e86\uff0c\u6240\u4ee5\u624d\u542f\u52a8\u4e0d\u4e86\u3002"
        "\u5148\u628a\u9009\u62e9\u53d8\u6210\u5c0f\u9a8c\u8bc1\uff1a\u5199\u4e0b\u4e09\u6761\u8def\uff0c\u6bcf\u6761\u53ea\u914d\u4e00\u4e2a\u6700\u5c0f\u52a8\u4f5c\uff0c\u6bd4\u5982\u67e5\u4e00\u4e2a\u62db\u751f\u4fe1\u606f\u3001\u6295\u4e00\u4e2a\u5c97\u4f4d\u3001\u95ee\u4e00\u4e2a\u5e08\u5144\u5e08\u59d0\u3002"
        "\u660e\u5929\u53ea\u7ed9\u81ea\u5df1 25 \u5206\u949f\u505a\u5176\u4e2d\u4e00\u4ef6\uff0c\u4e0d\u8981\u8981\u6c42\u81ea\u5df1\u7acb\u523b\u51b3\u5b9a\u4eba\u751f\u65b9\u5411\u3002\u5148\u8ba9\u7cfb\u7edf\u52a8\u8d77\u6765\uff0c\u6bd4\u903c\u81ea\u5df1\u60f3\u660e\u767d\u66f4\u91cd\u8981\u3002"
    )


def _has_irrelevant_modern_template(reply_text: str, user_text: str, history_text: str = "") -> bool:
    combined = f"{history_text} {user_text}".replace(" ", "")
    reply = reply_text.replace(" ", "")
    dorm_template = "\u665a\u4e0a12\u70b9\u540e\u7684\u7535\u8bdd\u58f0" in reply or "\u5bbf\u820d\u957f\u3001\u8f85\u5bfc\u5458\u6216\u5bbf\u7ba1" in reply
    thesis_template = "\u6709\u9650\u68c0\u67e5\u6d41\u7a0b" in reply and "\u91cd\u590d\u7387" in reply
    family_template = "\u5b89\u5168\u7684\u8bdd\u9898\u591a\u4ea4\u6d41" in reply and "\u6bcf\u5468\u56fa\u5b9a\u548c\u4f60\u8bf4\u4e00\u6b21\u8fd1\u51b5" in reply
    if dorm_template and not any(term in combined for term in ("\u5ba4\u53cb", "\u5bbf\u820d", "\u7535\u8bdd", "\u7761\u89c9", "\u9634\u9633\u602a\u6c14")):
        return True
    if thesis_template and not any(term in combined for term in ("\u8bba\u6587", "\u67e5\u91cd", "\u91cd\u590d\u7387", "\u5f15\u7528")):
        return True
    if family_template and not any(term in combined for term in ("\u7236\u6bcd", "\u6211\u5988", "\u6211\u7238", "\u5bb6\u91cc", "\u4e0d\u5b5d")):
        return True
    return False


def _repair_irrelevant_template(user_text: str, history_text: str = "") -> str:
    combined = f"{history_text} {user_text}"
    if any(term in combined for term in ("\u5ba0\u7269", "\u5b83\u4f1a\u4e0d\u4f1a\u8fd8\u5728", "\u6ca1\u7167\u987e\u597d")):
        return (
            "\u5931\u53bb\u4e4b\u540e\uff0c\u4eba\u5f88\u5bb9\u6613\u53cd\u590d\u60f3\u201c\u5982\u679c\u6211\u65e9\u70b9\u53d1\u73b0\u201d\uff0c\u597d\u50cf\u627e\u5230\u4e00\u4e2a\u539f\u56e0\u5c31\u80fd\u628a\u7ed3\u679c\u6539\u56de\u6765\u3002"
            "\u4f46\u5f88\u591a\u4e8b\u60c5\u5e76\u4e0d\u5b8c\u5168\u7531\u4f60\u63a7\u5236\uff0c\u4f60\u66fe\u7ecf\u7167\u987e\u5b83\u3001\u7231\u5b83\uff0c\u8fd9\u4e5f\u662f\u771f\u5b9e\u7684\u3002"
            "\u4eca\u665a\u53ef\u4ee5\u5148\u505a\u4e00\u4ef6\u5f88\u5c0f\u7684\u544a\u522b\u52a8\u4f5c\uff1a\u5199\u4e0b\u4e09\u4e2a\u4f60\u548c\u5b83\u5728\u4e00\u8d77\u7684\u7247\u6bb5\uff0c\u4e0d\u7528\u6025\u7740\u8bc1\u660e\u81ea\u5df1\u6ca1\u9519\u3002"
        )
    if any(term in combined for term in ("\u73ed\u7ea7\u6d3b\u52a8", "\u6ca1\u4eba\u642d\u7406", "\u6ca1\u6709\u5b58\u5728\u611f", "\u62cd\u7167")):
        return (
            "\u4f60\u53bb\u4e86\u6d3b\u52a8\uff0c\u5374\u6ca1\u6709\u611f\u5230\u88ab\u63a5\u7eb3\uff0c\u8fd9\u79cd\u843d\u5dee\u5f88\u4f24\u4eba\u3002"
            "\u5148\u4e0d\u628a\u5b83\u4e0a\u5347\u6210\u201c\u6211\u6c38\u8fdc\u6ca1\u4f4d\u7f6e\u201d\uff0c\u4eca\u5929\u7684\u4f53\u9a8c\u786e\u5b9e\u4e0d\u597d\uff0c\u4f46\u5b83\u8fd8\u4e0d\u80fd\u5b9a\u4e49\u4f60\u5728\u73ed\u91cc\u7684\u5168\u90e8\u4eba\u9645\u5173\u7cfb\u3002"
            "\u4e0b\u6b21\u53ef\u4ee5\u5148\u9009\u4e00\u4e2a\u4f4e\u538b\u529b\u5165\u53e3\uff1a\u63d0\u524d\u7ea6\u4e00\u4e2a\u540c\u5b66\u4e00\u8d77\u5230\u573a\uff0c\u6216\u53ea\u4e3b\u52a8\u53c2\u4e0e\u4e00\u4e2a\u5c0f\u4efb\u52a1\uff0c\u522b\u8981\u6c42\u81ea\u5df1\u4e00\u6b21\u5c31\u878d\u5165\u6240\u6709\u4eba\u3002"
        )
    if any(term in combined for term in ("\u8ddf\u8e2a", "\u5c3e\u968f", "\u6ca1\u6709\u8bc1\u636e", "\u964c\u751f\u4eba")):
        return (
            "\u62a5\u544a\u5b89\u5168\u9690\u60a3\u4e0d\u7b49\u4e8e\u4f60\u5728\u6307\u63a7\u67d0\u4e2a\u4eba\uff0c\u4f60\u53ef\u4ee5\u53ea\u63cf\u8ff0\u4e8b\u5b9e\u548c\u611f\u53d7\u5230\u7684\u98ce\u9669\u3002"
            "\u6bd4\u5982\u8bf4\uff1a\u201c\u6211\u5728\u67d0\u65f6\u95f4\u67d0\u8def\u6bb5\u611f\u5230\u88ab\u5c3e\u968f\uff0c\u73b0\u5728\u4e0d\u6562\u72ec\u81ea\u7ecf\u8fc7\uff0c\u5e0c\u671b\u5b66\u6821\u5173\u6ce8\u7167\u660e\u3001\u5de1\u903b\u6216\u966a\u540c\u8fd4\u56de\u3002\u201d"
            "\u4eca\u665a\u5c3d\u91cf\u4e0d\u72ec\u81ea\u8d70\u90a3\u6bb5\u8def\uff0c\u53ef\u4ee5\u8ba9\u540c\u5b66\u966a\u4f60\u6216\u9009\u4eba\u591a\u7684\u8def\u7ebf\u3002"
        )
    if any(term in combined for term in ("\u6253\u6e38\u620f", "\u6e38\u620f\u5230\u51cc\u6668", "\u63a7\u5236\u4e0d\u4e86", "\u4e0d\u60f3\u53bb\u4e0a\u8bfe")):
        return (
            "\u6e38\u620f\u73b0\u5728\u50cf\u662f\u4e00\u4e2a\u907f\u96be\u6240\uff0c\u80fd\u8ba9\u4f60\u6682\u65f6\u4e0d\u7528\u9762\u5bf9\u4f5c\u4e1a\u3001\u8bba\u6587\u548c\u672a\u6765\uff0c\u6240\u4ee5\u4e0d\u662f\u7b80\u5355\u7684\u201c\u4f60\u5e9f\u201d\u3002"
            "\u4eca\u665a\u5148\u4e0d\u8981\u8bbe\u76ee\u6807\u201c\u5f7b\u5e95\u6212\u6389\u201d\uff0c\u53ea\u505a\u4e00\u4e2a\u4e2d\u65ad\u70b9\uff1a\u8bbe\u4e00\u4e2a\u56fa\u5b9a\u4e0b\u7ebf\u65f6\u95f4\uff0c\u5230\u70b9\u628a\u8d26\u53f7\u9000\u51fa\u3001\u624b\u673a\u653e\u8fdc\uff0c\u7136\u540e\u53ea\u5904\u7406\u4e00\u4ef6\u73b0\u5b9e\u5c0f\u4e8b\u3002"
            "\u5982\u679c\u5df2\u7ecf\u6301\u7eed\u5f71\u54cd\u4e0a\u8bfe\u3001\u7761\u7720\u548c\u57fa\u672c\u751f\u6d3b\uff0c\u5efa\u8bae\u627e\u8f85\u5bfc\u5458\u6216\u5fc3\u7406\u8001\u5e08\u4e00\u8d77\u505a\u884c\u4e3a\u8ba1\u5212\u3002"
        )
    return _contextual_safe_reply(user_text, history_text)


def _has_unsupported_personal_inference(reply_text: str, user_text: str) -> bool:
    compact_reply = reply_text.replace(" ", "")
    compact_user = user_text.replace(" ", "")
    inferred_terms = ("\u5c0f\u65f6\u5019", "\u7ae5\u5e74", "\u539f\u751f\u5bb6\u5ead")
    user_terms = ("\u5c0f\u65f6\u5019", "\u7ae5\u5e74", "\u7236\u6bcd", "\u5bb6\u91cc", "\u5bb6\u5ead")
    return any(term in compact_reply for term in inferred_terms) and not any(term in compact_user for term in user_terms)


def _has_avoidant_or_unhelpful_action(reply_text: str) -> bool:
    compact = reply_text.replace(" ", "")
    terms = (
        "\u56de\u4e00\u4e2a\u8868\u60c5",
        "\u53ea\u56de\u4e00\u4e2a\u8868\u60c5",
        "\u4e0d\u56de\u4fe1\u606f",
        "\u4e0d\u56de\u6d88\u606f",
        "\u660e\u5929\u4e5f\u53ef\u4ee5\u53ea\u53d1\u4e00\u4e2a\u7b80\u5355\u7684\u8868\u60c5",
    )
    return any(term in compact for term in terms)


def _unsupported_inference_repair_reply(user_text: str, history_text: str = "") -> str:
    combined = f"{history_text} {user_text}"
    if any(term in combined for term in ("\u6bd4\u8d5b", "\u9879\u76ee", "\u8001\u5e08", "\u521b\u65b0", "\u5b9e\u73b0\u601d\u8def", "\u6d45")):
        return (
            "\u201c\u6d45\u201d\u8fd9\u4e2a\u8bcd\u786e\u5b9e\u5f88\u523a\u8033\uff0c\u5c24\u5176\u662f\u4f60\u5df2\u7ecf\u6295\u5165\u4e86\u65f6\u95f4\u548c\u7cbe\u529b\u3002"
            "\u4f46\u5b83\u66f4\u9002\u5408\u88ab\u62c6\u6210\u53ef\u4fee\u6539\u7684\u95ee\u9898\uff1a\u662f\u6570\u636e\u652f\u6491\u4e0d\u591f\u3001\u573a\u666f\u4e0d\u591f\u7ec6\uff0c\u8fd8\u662f\u65b9\u6cd5\u53ea\u505c\u5728\u8868\u5c42\uff1f"
            "\u4eca\u665a\u5148\u4e0d\u628a\u5b83\u7ffb\u8bd1\u6210\u201c\u6211\u4e0d\u884c\u201d\uff0c\u53ef\u4ee5\u5199\u4e09\u5217\uff1a\u8001\u5e08\u539f\u8bdd\u3001\u53ef\u80fd\u6307\u5411\u7684\u6280\u672f\u95ee\u9898\u3001\u4e0b\u4e00\u7248\u80fd\u8865\u7684\u4e00\u4e2a\u52a8\u4f5c\u3002"
        )
    return _contextual_safe_reply(user_text, history_text)


def _specific_action_repair_reply(user_text: str, history_text: str = "") -> str:
    combined = f"{history_text} {user_text}"
    if any(term in combined for term in ("\u5c0f\u7ec4", "\u7ec4\u5458", "\u56e2\u961f", "\u4f5c\u4e1a", "\u5206\u5de5", "\u8d21\u732e")):
        return (
            "\u4f60\u73b0\u5728\u8981\u4fdd\u62a4\u7684\u4e0d\u662f\u201c\u5fcd\u4e00\u5fcd\u5c31\u7b97\u4e86\u201d\uff0c\u800c\u662f\u8ba9\u81ea\u5df1\u7684\u53c2\u4e0e\u548c\u8d21\u732e\u53d8\u5f97\u53ef\u89c1\u3002"
            "\u53ef\u4ee5\u5148\u53d1\u4e00\u6761\u77ed\u800c\u5177\u4f53\u7684\u6d88\u606f\uff1a\u201c\u6211\u60f3\u786e\u8ba4\u4e00\u4e0b\u6211\u8fd9\u8fb9\u8d1f\u8d23\u54ea\u4e00\u90e8\u5206\uff0c\u4eca\u665a\u6211\u53ef\u4ee5\u5148\u628a\u8d44\u6599\u6c47\u603b/\u7b2c\u4e09\u90e8\u5206PPT\u521d\u7a3f\u505a\u51fa\u6765\u3002\u201d"
            "\u5982\u679c\u5bf9\u65b9\u4ecd\u7136\u4e0d\u56de\uff0c\u4f60\u53ef\u4ee5\u4fdd\u7559\u804a\u5929\u8bb0\u5f55\uff0c\u5148\u505a\u4e00\u4efd\u80fd\u5c55\u793a\u7684\u6210\u679c\uff0c\u5fc5\u8981\u65f6\u518d\u7528\u4e8b\u5b9e\u5411\u7ec4\u957f\u6216\u8001\u5e08\u8bf4\u660e\u5206\u5de5\u60c5\u51b5\u3002"
        )
    return _contextual_safe_reply(user_text, history_text)


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
        term in text for term in ("压力", "焦虑", "心慌", "烦", "咖啡", "硬撑", "熬")
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
    compact = text.replace(" ", "")
    negated_self_harm = (
        "不想伤害自己",
        "不会伤害自己",
        "没有想伤害自己",
        "没想伤害自己",
    )
    direct_crisis_terms = (
        "不想活",
        "想死",
        "自杀",
        "活不下去",
        "撑不住",
        "结束生命",
    )
    if any(phrase in compact for phrase in negated_self_harm) and not any(term in compact for term in direct_crisis_terms):
        return False
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
    if any(term in combined for term in ("睡不好", "失眠", "睡不着", "晚上", "压力", "咖啡", "硬撑")):
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
