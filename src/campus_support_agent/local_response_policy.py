from __future__ import annotations

from typing import Any

from .schemas import LocalPolicyInfo, LocalPolicyResult, PsychologicalEntropy, SupportAssessment, SupportPlan


def maybe_build_local_support_plan(
    text: str,
    *,
    entropy: PsychologicalEntropy,
    conversation_history: list[dict[str, Any]] | None = None,
) -> LocalPolicyResult | None:
    normalized = _normalize(text)
    history = conversation_history or []
    recent_text = " ".join(str(item.get("content", "")) for item in history[-4:])
    recent_normalized = _normalize(recent_text)

    if _matches_privacy_concern(normalized):
        return _wrap_local_policy(
            "privacy_concern",
            "rapport_boundary",
            "trust_and_confidentiality",
            _build_privacy_plan(entropy),
        )
    if _matches_dorm_conflict(normalized):
        return _wrap_local_policy(
            "dorm_conflict",
            "situational_support",
            "watch_interpersonal_escalation",
            _build_dorm_conflict_plan(entropy),
        )
    if _matches_exam_anxiety(normalized):
        return _wrap_local_policy(
            "exam_anxiety",
            "situational_support",
            "watch_sleep_and_academic_functioning",
            _build_exam_anxiety_plan(entropy),
        )
    if _matches_social_isolation(normalized):
        return _wrap_local_policy(
            "social_isolation",
            "situational_support",
            "watch_withdrawal_trend",
            _build_social_isolation_plan(entropy),
        )
    if _matches_exhaustion_withdrawal(normalized):
        return _wrap_local_policy(
            "exhaustion_withdrawal",
            "situational_support",
            "watch_daily_functioning",
            _build_exhaustion_withdrawal_plan(entropy),
        )
    if _matches_authority_pressure(normalized):
        return _wrap_local_policy(
            "authority_pressure",
            "situational_support",
            "watch_external_pressure_load",
            _build_authority_pressure_plan(entropy),
        )
    if _matches_future_panic(normalized):
        return _wrap_local_policy(
            "future_panic",
            "situational_support",
            "watch_future_uncertainty_spiral",
            _build_future_panic_plan(entropy),
        )
    if _matches_self_blame(normalized):
        return _wrap_local_policy(
            "self_blame",
            "situational_support",
            "watch_negative_self_evaluation",
            _build_self_blame_plan(entropy),
        )
    if _matches_sleep_appetite_drift(normalized, recent_normalized):
        return _wrap_local_policy(
            "sleep_appetite_drift",
            "escalation_watch",
            "consider_sleep_and_health_followup",
            _build_sleep_appetite_drift_plan(entropy),
        )
    if _matches_helplessness_escalation(normalized):
        return _wrap_local_policy(
            "helplessness_escalation",
            "escalation_watch",
            "consider_human_followup_if_persistent",
            _build_helplessness_escalation_plan(entropy),
        )
    if _matches_rising_emotional_spiral(normalized, recent_normalized):
        return _wrap_local_policy(
            "rising_emotional_spiral",
            "escalation_watch",
            "watch_multi_turn_worsening",
            _build_rising_emotional_spiral_plan(entropy),
        )
    if _matches_repetitive_help_seeking(normalized, recent_normalized):
        return _wrap_local_policy(
            "repetitive_help_seeking",
            "rapport_boundary",
            "watch_repetition_and_support_burnout",
            _build_repetitive_help_seeking_plan(entropy),
        )
    if _matches_late_night_distress(normalized, recent_normalized):
        return _wrap_local_policy(
            "late_night_distress",
            "escalation_watch",
            "watch_sleep_disruption",
            _build_late_night_plan(entropy),
        )
    if _matches_boundary_concern(normalized):
        return _wrap_local_policy(
            "boundary_concern",
            "rapport_boundary",
            "respect_disclosure_boundary",
            _build_boundary_plan(entropy, recent_normalized),
        )
    if _matches_weak_input(normalized):
        return _wrap_local_policy(
            "weak_input",
            "rapport_boundary",
            "stabilize_conversation_entry",
            _build_weak_input_plan(entropy, recent_normalized),
        )
    return None


def _wrap_local_policy(
    policy_name: str,
    policy_stage: str,
    escalation_hint: str | None,
    result: tuple[SupportAssessment, SupportPlan],
) -> LocalPolicyResult:
    assessment, plan = result
    plan = _soften_user_facing_plan(policy_name, plan)
    return LocalPolicyResult(
        assessment=assessment,
        plan=plan,
        info=LocalPolicyInfo(
            policy_name=policy_name,
            policy_stage=policy_stage,
            escalation_hint=escalation_hint,
        ),
    )


def _soften_user_facing_plan(policy_name: str, plan: SupportPlan) -> SupportPlan:
    """Keep clinical judgment in metadata while making the visible reply feel human."""
    if policy_name == "privacy_concern":
        plan.summary = (
            "你放心，我们可以先停在你觉得有安全感的范围里。你不用一次说很多，"
            "也不用说具体的人名、地点或细节。"
        )
        plan.immediate_support = [
            "我不会把你的话主动告诉别人；如果真的出现安全风险，我会优先建议你联系现实中可信任的人来保护你。",
            "你可以先只说一点点，比如你最担心的是被谁知道，还是担心说出来以后被误解。",
        ]
        plan.self_regulation = [
            "先不用逼自己讲完整经过。你可以把信息模糊一点，只说感受，不说具体人物和细节。"
        ]
        plan.follow_up = [
            "如果你愿意，下一句只告诉我：你现在更担心“别人知道”，还是更担心“说出来以后没人理解”。"
        ]
    elif policy_name == "boundary_concern":
        plan.summary = (
            "可以的，我们先不追问细节。你不想细说并不代表你在逃避，"
            "也可能只是现在还需要一点安全感和空间。"
        )
        plan.immediate_support = [
            "你不用为了让我理解就勉强自己多说，我们先不追问，可以停在你觉得舒服的位置。",
            "如果现在说具体事情太难，我们就用轻的方式来：你只说一个感觉也可以，比如烦、怕、委屈、累，或者先什么都不选。",
        ]
        plan.self_regulation = [
            "先把注意力放回当下，喝口水、慢一点呼吸，提醒自己：现在不需要一次讲清楚全部。"
        ]
        plan.follow_up = [
            "你可以只回一个词：现在更像烦、怕、委屈，还是累？如果不想选，也可以直接说“不想选”。"
        ]
    elif policy_name == "sleep_appetite_drift":
        plan.summary = (
            "这听起来已经影响到你的睡眠和吃饭了，确实会让人很难受。"
            "我们先不急着找出所有原因，先把身体状态往回稳一点。"
        )
        plan.immediate_support = [
            "今晚先别给自己加很多任务，先照顾最基础的两件事：能不能稍微吃一点、能不能让身体慢下来一点。",
            "你不用马上变好，我们先找一个最小动作，比如喝点温水、吃两口容易入口的东西，或者先躺下安静十分钟。",
        ]
        plan.campus_actions = [
            "如果睡不好、吃不下已经连续好几天，建议尽快找校医院或学校心理中心做一次基础评估，不要一个人硬扛。"
        ]
        plan.self_regulation = [
            "先把目标降到很小：今天只做一件照顾身体的小事，不要求自己立刻恢复状态。"
        ]
        plan.follow_up = [
            "你下一句只说一个就好：现在更明显的是睡不好，还是吃不下？我先陪你从这一点开始。"
        ]
    elif policy_name == "dorm_conflict":
        plan.summary = (
            "一回宿舍就烦，说明那个环境可能已经让你有点绷紧了。"
            "我们先不用急着判断谁对谁错，先把你现在这股烦接住。"
        )
        plan.immediate_support = [
            "你可以先不用讲完整经过，只说最刺你的一个点：是气氛压抑、被针对，还是一想到回宿舍就烦。",
            "如果现在很堵，可以先让自己离开那个刺激源一小会儿，比如去走廊、楼下或洗手间缓一下。"
        ]
        plan.follow_up = [
            "你愿意的话，先告诉我：这个烦更像生气、委屈，还是害怕之后继续相处？"
        ]
    return plan


def _normalize(text: str) -> str:
    normalized = text.strip().lower()
    for token in ["。", "，", "、", "？", "?", "！", "!", "；", ";", "：", ":", "“", "”", "\"", "'", " ", "\n", "\t"]:
        normalized = normalized.replace(token, "")
    return normalized


def _matches_privacy_concern(text: str) -> bool:
    triggers = [
        "怕别人知道",
        "害怕别人知道",
        "害怕别人会知道",
        "害怕被别人知道",
        "怕被别人知道",
        "不想让别人知道",
        "不想被别人知道",
        "你会告诉别人",
        "怕你告诉别人",
        "怕你会告诉别人",
        "怕你说出去",
        "会不会告诉别人",
        "会不会被别人看到",
        "会不会传出去",
        "保密",
        "泄露",
        "隐私",
        "privacy",
        "tellothers",
        "tellsomeone",
    ]
    return any(term in text for term in triggers)


def _matches_boundary_concern(text: str) -> bool:
    triggers = [
        "不太想细说",
        "不想细说",
        "不太想说",
        "不想说",
        "不想展开",
        "不想被追问",
        "不想继续说",
        "先不说了",
        "先不想说",
        "不想聊这个",
        "不想讲",
        "dontwanttotalk",
        "dontwanttoexplain",
    ]
    return any(term in text for term in triggers)


def _matches_dorm_conflict(text: str) -> bool:
    triggers = [
        "宿舍就烦",
        "一回宿舍就烦",
        "回宿舍就烦",
        "舍友针对我",
        "室友针对我",
        "舍友",
        "室友",
        "宿舍关系",
        "人际关系很紧张",
        "不想回宿舍",
    ]
    return any(term in text for term in triggers)


def _matches_exam_anxiety(text: str) -> bool:
    triggers = [
        "考试",
        "挂科",
        "成绩很差",
        "成绩不好",
        "考砸",
        "学业压力",
        "论文",
        "答辩",
        "复习不完",
        "不敢查成绩",
    ]
    return any(term in text for term in triggers)


def _matches_late_night_distress(text: str, recent_text: str) -> bool:
    direct_markers = [
        "半夜",
        "凌晨",
        "深夜",
        "睡不着",
        "失眠",
        "现在很晚",
        "晚上睡不着",
        "夜里睡不着",
    ]
    weak_followups = {"", "嗯", "?", "？", "不知道", "...", "还是"}
    if any(term in text for term in direct_markers):
        return True
    if "睡不着" in recent_text and text in weak_followups:
        return True
    return False


def _matches_weak_input(text: str) -> bool:
    return text in {"", "?", "？", "嗯", "哦", "啊", "...", "1", "2", "3", "不知道", "ok", "okay", "hmm"}


def _matches_social_isolation(text: str) -> bool:
    triggers = [
        "没人理我",
        "被孤立",
        "被排挤",
        "没有人找我",
        "不想见人",
        "不想社交",
        "不想跟人说话",
        "不想跟别人说话",
        "没有朋友",
        "融不进去",
    ]
    return any(term in text for term in triggers)


def _matches_exhaustion_withdrawal(text: str) -> bool:
    triggers = [
        "这几天都很累",
        "连续几天都很累",
        "什么都不想做",
        "不想动",
        "不想起床",
        "什么都提不起劲",
        "什么都没力气",
        "只想躺着",
        "不想见人也不想说话",
    ]
    return any(term in text for term in triggers)


def _matches_authority_pressure(text: str) -> bool:
    triggers = [
        "老师一直催",
        "老师给我压力",
        "家里一直催",
        "爸妈一直催",
        "父母一直催",
        "家长一直催",
        "他们一直逼我",
        "老师说我不够努力",
        "爸妈说我不够好",
        "总觉得达不到他们要求",
    ]
    return any(term in text for term in triggers)


def _matches_future_panic(text: str) -> bool:
    triggers = [
        "对未来很慌",
        "想到未来就慌",
        "不知道以后怎么办",
        "不知道未来怎么办",
        "好怕以后找不到工作",
        "怕毕业以后完蛋",
        "想到毕业就慌",
        "前途一片空白",
        "未来让我很害怕",
        "一想到以后就发慌",
    ]
    return any(term in text for term in triggers)


def _matches_self_blame(text: str) -> bool:
    triggers = [
        "都是我的错",
        "我是不是很差劲",
        "我觉得自己很失败",
        "我总觉得是我不好",
        "是不是我有问题",
        "我怎么什么都做不好",
        "我就是不行",
        "我总是在怪自己",
        "我一直在怪自己",
        "我真的很没用",
    ]
    return any(term in text for term in triggers)


def _matches_sleep_appetite_drift(text: str, recent_text: str) -> bool:
    sleep_markers = ["睡不着", "失眠", "睡不好", "醒得很早", "整晚没睡"]
    appetite_markers = ["吃不下", "没胃口", "不想吃饭", "吃不进去", "食欲很差"]
    has_sleep = any(term in text for term in sleep_markers)
    has_appetite = any(term in text for term in appetite_markers)
    if has_sleep and has_appetite:
        return True
    if has_sleep and any(term in recent_text for term in appetite_markers):
        return True
    if has_appetite and any(term in recent_text for term in sleep_markers):
        return True
    return False


def _matches_helplessness_escalation(text: str) -> bool:
    triggers = [
        "一点办法都没有",
        "我不知道还能怎么办",
        "怎么办都没用",
        "感觉什么都没用了",
        "什么都没用了",
        "我真的撑不住了",
        "我快撑不下去了",
        "我感觉没办法了",
        "我好像没救了",
        "怎么做都没有用",
        "我已经不知道该怎么撑了",
    ]
    return any(term in text for term in triggers)


def _matches_rising_emotional_spiral(text: str, recent_text: str) -> bool:
    escalation_markers = [
        "越来越糟",
        "越来越严重",
        "越来越慌",
        "越来越慢",
        "一天比一天糟",
        "比昨天更难受",
        "这几天更严重",
        "最近越来越",
    ]
    prior_distress_markers = [
        "难受",
        "慌",
        "烦",
        "压",
        "睡不着",
        "害怕",
        "不想见人",
    ]
    if any(term in text for term in escalation_markers):
        return True
    if recent_text and any(term in recent_text for term in prior_distress_markers) and any(term in text for term in ["越来越", "更严重", "更难受", "更慌", "更糟", "更慢"]):
        return True
    return False


def _matches_repetitive_help_seeking(text: str, recent_text: str) -> bool:
    if any(term in text for term in ["半夜", "凌晨", "深夜", "睡不着", "失眠"]):
        return False

    repeat_markers = [
        "我又来了",
        "还是很难受",
        "还是不行",
        "还是一样",
        "又开始了",
        "反复想",
        "一直在想",
        "还是很怕",
    ]
    if any(term in text for term in repeat_markers):
        return True
    if recent_text and any(term in recent_text for term in ["挂科", "宿舍", "害怕", "难受", "不想细说"]) and text in {"", "?", "？", "嗯", "还是", "又来了"}:
        return True
    return False


def _build_privacy_plan(entropy: PsychologicalEntropy) -> tuple[SupportAssessment, SupportPlan]:
    assessment = SupportAssessment(
        primary_emotions=["担心", "不安"],
        stressors=["害怕隐私被泄露", "担心说出来会被别人知道"],
        protective_factors=["还愿意表达顾虑", "在主动确认这里是否安全"],
        entropy_level=entropy.level,
        balance_state=entropy.balance_state,
    )
    plan = SupportPlan(
        summary=(
            "你现在卡住，更多是因为你担心说出来之后会不会被别人知道。"
            "先把这份安全感稳住，比马上讲清楚来龙去脉更重要。"
        ),
        immediate_support=[
            "你放心，我们先按你觉得安全的范围来，不用一下子把所有细节都说出来。",
            "如果你愿意，我们可以先只说一点点，比如这件事最让你紧张的，到底是被议论、被误解，还是被追着问。",
        ],
        campus_actions=[
            "如果后面你想找线下支持，也可以先只问咨询流程和保密范围，不一定一开始就把经过讲完整。",
        ],
        self_regulation=[
            "先做一件让自己更有安全感的小事，比如先不提名字、不提具体人，只用一句话说说你最担心什么。",
        ],
        follow_up=[
            "如果你愿意，下一句你只要告诉我：你现在最怕的是被看到、被传开，还是被追着解释。",
        ],
    )
    return assessment, plan


def _build_dorm_conflict_plan(entropy: PsychologicalEntropy) -> tuple[SupportAssessment, SupportPlan]:
    assessment = SupportAssessment(
        primary_emotions=["烦躁", "压抑"],
        stressors=["宿舍氛围让人绷着", "人际关系持续消耗精力"],
        protective_factors=["还能察觉自己被什么触发", "还愿意把难受说出来"],
        entropy_level=entropy.level,
        balance_state=entropy.balance_state,
    )
    plan = SupportPlan(
        summary="你不是单纯心情差，更像是一回到宿舍就会被那种压着的感觉重新顶上来。一直这样绷着，确实会很累。",
        immediate_support=[
            "我们先不急着判断谁对谁错，先把你现在这股烦和堵接住。",
            "如果你愿意，你可以先只说一句最扎你的点，比如是气氛压抑、被针对，还是一想到回宿舍就很烦。",
        ],
        campus_actions=[
            "如果这种状态已经持续一段时间，也可以考虑找辅导员、宿舍管理老师或学校心理中心先聊聊怎么把环境压力降下来。",
        ],
        self_regulation=[
            "今晚如果还得回宿舍，先给自己留一个缓冲动作，比如在进门前站一分钟、喝口水，别让自己一下子硬顶进去。",
        ],
        follow_up=[
            "如果你愿意，我们下一句就只说一件事：宿舍里到底是哪一下最容易让你瞬间烦起来。",
        ],
    )
    return assessment, plan


def _build_exam_anxiety_plan(entropy: PsychologicalEntropy) -> tuple[SupportAssessment, SupportPlan]:
    assessment = SupportAssessment(
        primary_emotions=["焦虑", "紧绷"],
        stressors=["担心考试或成绩失控", "学业任务堆积感太重"],
        protective_factors=["还能说出压力来源", "还在寻找能让自己稳一点的方法"],
        entropy_level=entropy.level,
        balance_state=entropy.balance_state,
    )
    plan = SupportPlan(
        summary="你现在最难受的，不只是任务多，而是脑子已经开始往“会不会全盘搞砸”那边冲了。一直这样绷着，睡眠也很容易被带坏。",
        immediate_support=[
            "我们先不急着谈整门课或整场考试，先只盯住眼前最卡的一件事。",
            "如果你愿意，你先告诉我现在最压你的，是怕挂科、怕来不及复习，还是一躺下脑子就停不下来。",
        ],
        campus_actions=[
            "如果这周已经明显影响睡眠和白天状态，也可以考虑找学院老师或学校心理中心聊聊，先把学业压力拆开。",
        ],
        self_regulation=[
            "今晚先别要求自己把整周都安排完，只列明天最必要的一小步，剩下的先放着。",
        ],
        follow_up=[
            "你下一句只说一个最重的点就行，我先陪你把那一个点拆开。",
        ],
    )
    return assessment, plan


def _build_late_night_plan(entropy: PsychologicalEntropy) -> tuple[SupportAssessment, SupportPlan]:
    assessment = SupportAssessment(
        primary_emotions=["疲惫", "失控感"],
        stressors=["深夜越想越停不下来", "睡不着让情绪更容易放大"],
        protective_factors=["还愿意求助", "还没完全把自己封住"],
        entropy_level=entropy.level,
        balance_state=entropy.balance_state,
    )
    plan = SupportPlan(
        summary="已经到这种时候还睡不着，人的脑子很容易把压力放大。所以我们现在先不追求想通，先让你从继续耗着的状态里退半步。",
        immediate_support=[
            "你先别逼自己立刻把问题讲清楚，我们先把节奏放慢，让身体先松一点。",
            "如果你愿意，先只告诉我：你现在更像是慌、累，还是胸口一直绷着。",
        ],
        campus_actions=[
            "如果你最近连续几晚都这样，白天状态也被拖下来了，后面可以考虑去校医院或学校心理中心评估一下睡眠和压力。",
        ],
        self_regulation=[
            "先把屏幕亮度降下来，放下“今晚必须想明白”的要求，只做一件最小的安静动作，比如慢呼吸或喝几口温水。",
        ],
        follow_up=[
            "我们先只处理今晚这一刻，不处理整周。你下一句只要告诉我现在最明显的不舒服是什么。",
        ],
    )
    return assessment, plan


def _build_social_isolation_plan(entropy: PsychologicalEntropy) -> tuple[SupportAssessment, SupportPlan]:
    assessment = SupportAssessment(
        primary_emotions=["委屈", "退缩"],
        stressors=["觉得自己被放在外面", "和人接触会消耗很多力气"],
        protective_factors=["还能把孤立感说出来", "还愿意维持一点联系"],
        entropy_level=entropy.level,
        balance_state=entropy.balance_state,
    )
    plan = SupportPlan(
        summary="你现在难受的，不只是一个人待着，而是那种“好像自己被放在外面”的感觉。这个感觉会很伤人，也会让人越来越不想靠近别人。",
        immediate_support=[
            "我们先不逼自己马上回到热闹里，先把这股被隔开的委屈接住。",
            "如果你愿意，你先只告诉我最近最扎你的那一下，是被忽略、被排挤，还是明明在人群里也还是很孤单。",
        ],
        campus_actions=[
            "如果这种状态已经持续一段时间，也可以考虑先找一个你相对没那么有压力的人聊一句，不用一开始就讲很多。",
        ],
        self_regulation=[
            "今天先别给自己定“必须重新融入”的目标，只做一个最低门槛的联系动作，比如回一条消息，或者和一个人打个招呼。",
        ],
        follow_up=[
            "你下一句只说最近最让你觉得“被放在外面”的一个场景就行，我先陪你把那个场景说清楚。",
        ],
    )
    return assessment, plan


def _build_exhaustion_withdrawal_plan(entropy: PsychologicalEntropy) -> tuple[SupportAssessment, SupportPlan]:
    assessment = SupportAssessment(
        primary_emotions=["疲惫", "迟钝"],
        stressors=["连续消耗后整个人发空", "连基本动作都变得费力"],
        protective_factors=["还能察觉自己状态在掉", "还愿意发出一点求助信号"],
        entropy_level=entropy.level,
        balance_state=entropy.balance_state,
    )
    plan = SupportPlan(
        summary="你现在更像是被连续消耗到没电了，不是简单的懒，也不是一句“打起精神”就能过去。人在这种时候，连见人和说话都会变得很费劲。",
        immediate_support=[
            "那我们先不要求自己振作，也不安排一堆事，先只做最小的一步。",
            "如果你愿意，你先告诉我这几天最明显的是哪一种：身体特别累、脑子发木，还是一想到要面对人就更想躲开。",
        ],
        campus_actions=[
            "如果这种状态已经连续几天影响到上课、吃饭或睡眠，后面最好还是找学校心理中心或校医院做一次基本评估。",
        ],
        self_regulation=[
            "今天先把目标降到最低，只保留一件最必要的事，其他的先不和自己较劲。",
        ],
        follow_up=[
            "我们先别谈整周安排，你下一句只告诉我：最近最拖垮你的到底是累、空，还是不想面对人。",
        ],
    )
    return assessment, plan


def _build_authority_pressure_plan(entropy: PsychologicalEntropy) -> tuple[SupportAssessment, SupportPlan]:
    assessment = SupportAssessment(
        primary_emotions=["压迫感", "委屈"],
        stressors=["外部期待一直压下来", "怎么做都像是不够"],
        protective_factors=["还能分辨出压力来源", "还愿意把这种被逼着走的感觉说出来"],
        entropy_level=entropy.level,
        balance_state=entropy.balance_state,
    )
    plan = SupportPlan(
        summary="你现在难受的，不只是任务多，而是那种一直被催、一直怕达不到要求的压迫感。人在这种状态里，很容易慢慢把别人的标准全压成对自己的否定。",
        immediate_support=[
            "我们先不急着证明你够不够好，先把这股一直顶着你的压力认出来。",
            "如果你愿意，你先只告诉我，现在压你最紧的是老师、家里，还是两边一起上。",
        ],
        campus_actions=[
            "如果这种压力已经明显影响到睡眠、吃饭或者上课状态，后面可以考虑找辅导员、心理中心或信任的老师做一次具体梳理，不用自己一个人扛到底。",
        ],
        self_regulation=[
            "先把“我要同时让所有人满意”这个任务拆掉，只盯住今天最现实的一件事。",
        ],
        follow_up=[
            "你下一句只说现在最让你喘不过气的一句话是谁说的，或者是哪一种要求最压你，我陪你先把那个点拎出来。",
        ],
    )
    return assessment, plan


def _build_future_panic_plan(entropy: PsychologicalEntropy) -> tuple[SupportAssessment, SupportPlan]:
    assessment = SupportAssessment(
        primary_emotions=["发慌", "失控感"],
        stressors=["未来太远太大", "一想到后面就脑子发空"],
        protective_factors=["还能意识到自己是在被未来感拖住", "还愿意停下来求助"],
        entropy_level=entropy.level,
        balance_state=entropy.balance_state,
    )
    plan = SupportPlan(
        summary="你现在卡住的，不一定是眼前这一步做不了，而是一想到后面那一长串未知，整个人就先慌起来了。未来一旦被脑子放大，人就很容易觉得哪一步都迈不动。",
        immediate_support=[
            "我们先不把未来一口气想完，先把它从一整团模糊的慌里拆出来。",
            "如果你愿意，你先只告诉我：你现在最怕的，是毕业去向、工作、成绩，还是怕自己以后会一直这样。",
        ],
        campus_actions=[
            "后面如果你愿意，也可以把这个问题拆成具体的信息缺口，再去找就业中心、导师或者心理中心各补一块，不用一个人硬想完整答案。",
        ],
        self_regulation=[
            "今天先不追求想通未来，只处理最靠近眼前的一步，比如这周最需要确认的一件事。",
        ],
        follow_up=[
            "你下一句只说“未来”里最吓你的那个词就行，我先陪你缩小到一个点。",
        ],
    )
    return assessment, plan


def _build_self_blame_plan(entropy: PsychologicalEntropy) -> tuple[SupportAssessment, SupportPlan]:
    assessment = SupportAssessment(
        primary_emotions=["自责", "挫败"],
        stressors=["把问题全压到自己身上", "越想越像是在给自己定罪"],
        protective_factors=["还能觉察到自己在反复责怪自己", "愿意把这种内耗说出来"],
        entropy_level=entropy.level,
        balance_state=entropy.balance_state,
    )
    plan = SupportPlan(
        summary="你现在像是在把很多事都往自己身上揽，好像只要结果不好，就会自动落成“是我不行”。这种自责会很伤人，也会把人越困越窄。",
        immediate_support=[
            "我们先不急着判断你到底有没有做错，先把那股一直往自己身上压的劲松一点。",
            "如果你愿意，你先告诉我，现在你最常拿来怪自己的那一句话是什么。",
        ],
        campus_actions=[
            "如果这种自责已经连续很多天都停不下来，也可以考虑找学校心理中心做一次更系统的梳理，别让它一直在脑子里自己打转。",
        ],
        self_regulation=[
            "先把“都是我的错”换成更具体的一句话，比如“这件事里我最遗憾的是哪一部分”，先别一次判整个人。",
        ],
        follow_up=[
            "你下一句只说你现在最想责怪自己的那个点，我先陪你把那句话拆开，不急着下结论。",
        ],
    )
    return assessment, plan


def _build_sleep_appetite_drift_plan(entropy: PsychologicalEntropy) -> tuple[SupportAssessment, SupportPlan]:
    assessment = SupportAssessment(
        primary_emotions=["被拖垮感", "虚耗"],
        stressors=["睡眠和食欲一起往下掉", "身体状态在慢慢撑不住"],
        protective_factors=["已经注意到身体在报警", "愿意把这种变化说出来"],
        entropy_level=entropy.level,
        balance_state=entropy.balance_state,
    )
    plan = SupportPlan(
        summary="这已经不只是心里烦一点了，而是睡眠和吃饭都开始被拖下来了。人一旦连身体节律都被打乱，很多情绪会被一起放大，所以我们要把它当成一个更需要认真看待的信号。",
        immediate_support=[
            "我们先不急着把所有原因都想清楚，先把这两个最基础的状态稳一点：睡和吃。",
            "如果你愿意，你先只告诉我现在更明显的是睡不着、醒得很早，还是一整天都没胃口。",
        ],
        campus_actions=[
            "如果这种情况已经连续几天，后面最好尽快去校医院或学校心理中心做一次基础评估，不要一直靠硬扛。",
        ],
        self_regulation=[
            "今晚先别给自己加任务，只保留最基本的照顾动作，比如喝点温的、吃两口容易入口的东西，或者先躺下让身体安静十分钟。",
        ],
        follow_up=[
            "你下一句只说现在最明显掉下来的那一项是什么，我先陪你从最基础的地方稳住一点。",
        ],
    )
    return assessment, plan


def _build_helplessness_escalation_plan(entropy: PsychologicalEntropy) -> tuple[SupportAssessment, SupportPlan]:
    assessment = SupportAssessment(
        primary_emotions=["无助", "塌陷感"],
        stressors=["开始觉得怎么做都没有用", "能撑住的感觉在明显变弱"],
        protective_factors=["还愿意把这种撑不住的感觉说出来", "还在尝试抓住外部支持"],
        entropy_level=entropy.level,
        balance_state=entropy.balance_state,
    )
    plan = SupportPlan(
        summary="你现在不是单纯在烦，而是已经开始有那种“怎么做都没有用”的无助感了。这个阶段最难受的地方，是人会慢慢失去抓手，觉得自己往哪边都够不到。",
        immediate_support=[
            "我们先不要求你立刻振作，也不急着给大答案，先把这个“撑不住”的感觉接住。",
            "如果你愿意，你先只告诉我，现在最让你觉得无力的，是事情本身太多，还是你已经不知道下一步抓什么。",
        ],
        campus_actions=[
            "如果这种无助感已经明显加重，后面最好尽快把它带去做一次线下支持，不要等到自己彻底耗光再处理。",
        ],
        self_regulation=[
            "先把目标缩到只剩下一步，不处理整个局面，只找眼前一个还勉强能抓住的小动作。",
        ],
        follow_up=[
            "你下一句只说现在最让你觉得“没办法了”的那个点，我先陪你把那个点单独拎出来。",
        ],
    )
    return assessment, plan


def _build_rising_emotional_spiral_plan(entropy: PsychologicalEntropy) -> tuple[SupportAssessment, SupportPlan]:
    assessment = SupportAssessment(
        primary_emotions=["持续走高的慌乱", "失稳感"],
        stressors=["情绪不是原地不动，而是在往上卷", "最近几轮状态在明显变重"],
        protective_factors=["已经能察觉到变化趋势", "还愿意停下来求助而不是继续硬撑"],
        entropy_level=entropy.level,
        balance_state=entropy.balance_state,
    )
    plan = SupportPlan(
        summary="你现在最值得注意的，不只是这件事本身难受，而是你的状态好像还在继续往上走。人一旦觉得自己一天比一天更糟，就很容易开始慌着对付整个局面。",
        immediate_support=[
            "我们先别急着解决全部问题，先确认这一轮加重最明显的是哪一块。",
            "如果你愿意，你先只告诉我：最近这几天更明显的是更慌了、更烦了，还是更累、更撑不住了。",
        ],
        campus_actions=[
            "如果你感觉这种上升趋势已经连续几天，后面最好尽快把这条变化带去做一次正式支持，别只靠自己盯着它继续变重。",
        ],
        self_regulation=[
            "先别把注意力放在“怎么一下子扭转回来”，先只做一件事：把最近加重最明显的那一块说清楚。",
        ],
        follow_up=[
            "你下一句只说最近和前两天比，最明显变重的是哪一种感觉，我先陪你抓那个变化。",
        ],
    )
    return assessment, plan


def _build_repetitive_help_seeking_plan(entropy: PsychologicalEntropy) -> tuple[SupportAssessment, SupportPlan]:
    assessment = SupportAssessment(
        primary_emotions=["反复焦灼", "耗竭"],
        stressors=["同一个困扰在来回拉扯", "越想解决越被困住"],
        protective_factors=["还愿意回来求助", "还没有完全放弃自己"],
        entropy_level=entropy.level,
        balance_state=entropy.balance_state,
    )
    plan = SupportPlan(
        summary="我能感觉到，这件事不是刚刚冒出来的，而是在反复拉扯你。一直这样来回顶着，人会越来越累，也会更容易觉得自己怎么都出不去。",
        immediate_support=[
            "这次我们先不求一下子解决它，先只找出“现在和上一次比，最难受的是不是同一个点”。",
            "如果你愿意，你先告诉我这次回来最明显的变化：是更慌了、更累了，还是还是卡在同一个地方。",
        ],
        campus_actions=[
            "如果你发现自己已经反复被同一个点拖住很多次，后面可以考虑把这件事带去做一次线下支持，别一直一个人循环消耗。",
        ],
        self_regulation=[
            "先不要逼自己立刻给出结论，只区分一件事：现在是在重复同一个痛点，还是又多了一个新的压力源。",
        ],
        follow_up=[
            "你下一句只说这次和上一次相比，最不一样的一点是什么，我陪你先抓那个变化。",
        ],
    )
    return assessment, plan


def _build_boundary_plan(
    entropy: PsychologicalEntropy,
    recent_text: str,
) -> tuple[SupportAssessment, SupportPlan]:
    if any(term in recent_text for term in ["压力", "睡不好", "失眠", "挂科", "考试", "宿舍", "舍友", "烦"]):
        summary = "你不是不想聊，你只是现在没有力气一下子讲很深。那我们就先把节奏放慢一点。"
        follow_up = "如果你愿意，我们先只停在一句话：最近最压着你的，是考试、睡眠，还是宿舍这边。"
    else:
        summary = "我知道你现在不想细说，这个我会尊重。先不追细节，也不代表你只能一个人硬扛。"
        follow_up = "如果你愿意，我们先不讲经过，只讲你现在心里最沉的那一下是什么感觉。"

    assessment = SupportAssessment(
        primary_emotions=["疲惫", "防备"],
        stressors=["不想被追问", "暂时没有力气展开细节"],
        protective_factors=["还能明确表达边界"],
        entropy_level=entropy.level,
        balance_state=entropy.balance_state,
    )
    plan = SupportPlan(
        summary=summary,
        immediate_support=[
            "好，那我们先不往下追问细节，你不用为了让我理解就勉强自己多说。",
            "我们可以先用最轻的方式聊，比如只说感受、只说一个片段，或者先什么都不展开。",
        ],
        campus_actions=[
            "如果你后面想找线下支持，也可以先约一个简短咨询，把目标放在“先试试看”，不用一开始就讲很深。",
        ],
        self_regulation=[
            "先把注意力放回当下，比如看一眼周围、喝口水，告诉自己：现在不用立刻把一切都说清楚。",
        ],
        follow_up=[follow_up],
    )
    return assessment, plan


def _build_weak_input_plan(
    entropy: PsychologicalEntropy,
    recent_text: str,
) -> tuple[SupportAssessment, SupportPlan]:
    if any(term in recent_text for term in ["压力", "睡不好", "失眠", "考试", "挂科", "宿舍", "舍友", "烦", "害怕"]):
        summary = "我感觉你不是没事，只是现在一下子不想说太多。那我们就慢一点，不急着把话一下子说满。"
        follow_up = "如果你愿意，你下一句只说一个词也可以，比如“慌”“累”“烦”里哪一个更像你现在。"
    else:
        summary = "我在，你不用急着组织得很完整。我们可以慢慢来。"
        follow_up = "如果你愿意，我们就接着刚才那件事；如果你想换个轻一点的话题也可以。"

    assessment = SupportAssessment(
        primary_emotions=["卡住"],
        stressors=["一时说不出来", "表达负荷偏高"],
        protective_factors=["还愿意保持对话"],
        entropy_level=entropy.level,
        balance_state=entropy.balance_state,
    )
    plan = SupportPlan(
        summary=summary,
        immediate_support=[
            "没关系，你现在不用马上把话说完整，我会等你。",
            "如果打字很累，我们就先用最短的方式聊，一两个字也可以。",
        ],
        campus_actions=[],
        self_regulation=[
            "先让自己慢半拍，呼一口气，告诉自己现在不用着急证明或解释什么。",
        ],
        follow_up=[follow_up],
    )
    return assessment, plan
