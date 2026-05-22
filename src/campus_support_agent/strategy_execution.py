from __future__ import annotations

from typing import Any

from .schemas import (
    DynamicAdjustment,
    EntropyAdjustmentLoop,
    FeedbackAdaptation,
    InterventionStrategy,
    StateProfile,
    SupportPlan,
)


def apply_intervention_strategy_to_plan(
    plan: SupportPlan,
    *,
    strategy: InterventionStrategy,
    state_profile: StateProfile,
) -> SupportPlan:
    """Shape the user-visible plan so it actually follows the chosen strategy."""

    if strategy.strategy_id == "privacy_reassurance":
        return _privacy_reassurance(plan)
    if strategy.strategy_id == "low_pressure_presence":
        return _low_pressure_presence(plan)
    if strategy.strategy_id == "sleep_stabilization":
        return _sleep_stabilization(plan)
    if strategy.strategy_id == "grounding_small_step":
        return _grounding_small_step(plan, state_profile)
    if strategy.strategy_id == "future_uncertainty_grounding":
        return _future_uncertainty_grounding(plan)
    if strategy.strategy_id == "dorm_boundary_support":
        return _dorm_boundary_support(plan)
    if strategy.strategy_id == "group_work_visibility":
        return _group_work_visibility(plan)
    if strategy.strategy_id == "performance_grounding":
        return _performance_grounding(plan)
    if strategy.strategy_id == "escape_loop_interruption":
        return _escape_loop_interruption(plan)
    if strategy.strategy_id == "family_boundary_sustainability":
        return _family_boundary_sustainability(plan)
    if strategy.strategy_id == "grief_without_self_blame":
        return _grief_without_self_blame(plan)
    if strategy.strategy_id == "safety_reporting_without_blame":
        return _safety_reporting_without_blame(plan)
    return _supportive_listening(plan, state_profile)


def apply_dynamic_adjustment_to_plan(
    plan: SupportPlan,
    *,
    dynamic_adjustment: DynamicAdjustment,
) -> SupportPlan:
    """Adjust the reply intensity using session-level entropy movement."""

    action = dynamic_adjustment.action
    if action == "soften_and_stabilize":
        plan.summary = "我先不急着分析原因，也不催你把事情讲完整。现在更重要的是让你这一刻先缓下来。"
        plan.immediate_support = _prepend(
            plan.immediate_support,
            "你可以先只做一件很小的事：离开正在刺激你的场景一分钟，喝两口水，或者把手机放下闭眼十秒。",
        )
        plan.follow_up = _prepend(
            plan.follow_up,
            "如果你愿意，我们下一句只需要确认一件事：现在最让你难受的是人、任务，还是身体状态？",
        )
        return plan

    if action == "escalate_support":
        plan.summary = "我注意到你的压力比前面更往上走了。我们先不把问题扩大，先把眼前这一段稳住。"
        plan.immediate_support = _prepend(
            plan.immediate_support,
            "先把今晚的目标降到最低：不要求马上变好，只先保证自己不继续被压力推着走。",
        )
        plan.follow_up = _prepend(
            plan.follow_up,
            "如果这种状态继续升高，建议你尽快找一个现实里可信的人说一声，比如同学、室友、辅导员或心理中心。",
        )
        return plan

    if action == "human_followup_watch":
        plan.summary = "这个状态已经不太适合只靠自己硬撑了。你不用马上做很多决定，但需要让现实里的支持靠近一点。"
        plan.campus_actions = _prepend(
            plan.campus_actions,
            "可以优先联系学校心理中心、辅导员，或者先告诉一个你信得过的人：我最近状态不太稳，想有人陪我一下。",
        )
        plan.follow_up = _prepend(
            plan.follow_up,
            "接下来我们可以继续聊，但系统也会把重点放在持续观察和必要时转介支持上。",
        )
        return plan

    if action == "maintain_and_consolidate":
        plan.follow_up = _prepend(
            plan.follow_up,
            "刚才的状态有一点往下稳的趋势，我们先不加新任务，只保留那个对你有帮助的小动作。",
        )
        return plan

    return plan


def apply_feedback_adaptation_to_plan(
    plan: SupportPlan,
    *,
    feedback_adaptation: FeedbackAdaptation,
) -> SupportPlan:
    """Use prior user feedback to repair the next reply style."""

    if feedback_adaptation.mode == "standard":
        return plan

    if feedback_adaptation.mode == "repair_next_turn":
        plan.summary = "我先调整一下刚才的方式：不急着追问，也不套模板，我们只围绕你现在真正卡住的地方来。"
        plan.immediate_support = _prepend(
            plan.immediate_support,
            "你可以不用解释完整。我会先根据你已经说出的部分回应，如果我理解偏了，你只要指出来就行。",
        )

    if feedback_adaptation.question_pressure == "low":
        plan.follow_up = _prepend(
            plan.follow_up,
            "下一步不用回答很多问题；如果愿意，只选一个方向说：想被陪着、想要建议，还是想先安静一下。",
        )

    if feedback_adaptation.detail_level == "more_concrete":
        plan.self_regulation = _prepend(
            plan.self_regulation,
            "先做一个很具体的小动作：把接下来 10 分钟要做的事写成一句话，只处理这一句，不处理整件事。",
        )

    if feedback_adaptation.should_avoid_repetition:
        plan.follow_up = _prepend(
            plan.follow_up,
            "我会尽量避免重复刚才的说法，换一种更贴近你当前处境的方式陪你梳理。",
        )

    return plan


def apply_adjustment_loop_to_plan(
    plan: SupportPlan,
    *,
    adjustment_loop: EntropyAdjustmentLoop | dict[str, Any] | None,
) -> SupportPlan:
    """Apply the hidden entropy-adjustment loop to the next visible support move."""

    if not adjustment_loop:
        return plan
    loop = adjustment_loop if isinstance(adjustment_loop, dict) else {
        "loop_action": adjustment_loop.loop_action,
        "question_policy": adjustment_loop.question_policy,
        "preferred_moves": adjustment_loop.preferred_moves,
        "human_followup_policy": adjustment_loop.human_followup_policy,
    }
    action = str(loop.get("loop_action") or "")
    question_policy = str(loop.get("question_policy") or "")

    if action == "safety_first":
        plan.summary = "我先把重点放在你此刻的安全和现实支持上，不急着分析原因。"
        plan.immediate_support = _prepend(
            plan.immediate_support,
            "如果你现在有伤害自己或控制不住冲动的风险，请先离开危险物品，联系身边可信的人，或直接联系当地紧急求助渠道。",
        )
        return plan

    if action == "human_followup_watch":
        plan.campus_actions = _prepend(
            plan.campus_actions,
            "如果这件事已经连续影响睡眠、上课或基本生活，可以考虑找辅导员、心理中心或一个可信同学陪你一起处理，不需要一个人硬扛。",
        )

    if action == "switch_strategy":
        plan.summary = "我换一种方式接住你：先不重复刚才那套建议，我们只看眼前最卡住的一点。"
        plan.immediate_support = _prepend(
            plan.immediate_support,
            "刚才的办法如果不贴合你，我们先停一下。你不需要马上解释完整，我会尽量跟着你现在说到的部分来。",
        )

    if action == "repair_reply_style":
        plan.summary = "我刚才可能没有真正接住你的意思，先把节奏放慢一点。"
        plan.immediate_support = _prepend(
            plan.immediate_support,
            "你可以不用继续解释细节。我先确认一点：你现在更需要被理解和陪着，而不是马上被安排一堆建议。",
        )

    if action == "increase_contextuality":
        plan.summary = "我会尽量贴着你刚刚说的具体处境来回应，不用套话带过去。"
        plan.follow_up = _prepend(
            plan.follow_up,
            "下一步我们只围绕你刚才提到的那个场景看，不跳到别的话题，也不重复泛泛的安慰。",
        )

    if action == "stabilize_before_problem_solving":
        plan.summary = "现在先不急着解决全部问题，先让身体和脑子从高压里降下来一点。"
        plan.self_regulation = _prepend(
            plan.self_regulation,
            "可以先做一个很小的动作：把手机放下半分钟，慢慢吐气三次，然后只决定接下来十分钟做什么。",
        )

    if action == "build_trust":
        plan.summary = "你可以选择说多少，也可以暂时不说细节；我会尊重你的边界。"
        plan.immediate_support = _prepend(
            plan.immediate_support,
            "在这里我们先以你的感受和安全为主，不强迫你解释原因。你愿意说一点点也可以，不想说也没关系。",
        )

    if question_policy in {"low_pressure_or_no_question", "no_forced_disclosure"}:
        plan.follow_up = _prepend(
            plan.follow_up,
            "如果你不想回答问题，也可以只回我一个词，比如“陪着”“建议”或“先停一下”。",
        )
    elif question_policy in {"one_optional_question", "one_contextual_question", "one_low_pressure_question"}:
        plan.follow_up = _prepend(
            plan.follow_up,
            "如果你愿意，我们下一步只看一个小点：此刻最让你难受的是情绪、身体疲惫，还是眼前那件事本身？",
        )

    return plan


def apply_session_continuity_to_plan(
    plan: SupportPlan,
    *,
    continuity_summary: dict[str, Any] | None,
) -> SupportPlan:
    """Turn hidden session continuity into concrete next-turn behavior."""

    if not continuity_summary:
        return plan

    stage = str(continuity_summary.get("dialogue_stage") or "")
    avoid_next_turn = {str(item) for item in continuity_summary.get("avoid_next_turn") or []}
    recommended_moves = [str(item) for item in continuity_summary.get("recommended_next_moves") or []]
    user_needs = [str(item) for item in continuity_summary.get("recent_user_needs") or []]

    if stage == "safety_priority":
        plan.summary = "我会先把安全放在第一位。现在不用解释完整原因，我们先确认你此刻身边是否安全、有没有可以立刻联系的人。"
        plan.immediate_support = _prepend(
            plan.immediate_support,
            "如果你有伤害自己的冲动，先把可能伤到自己的东西放远一点，并尽快联系身边可信任的人、辅导员或当地紧急电话。",
        )
        plan.follow_up = _prepend(plan.follow_up, "你可以只回复我：现在是安全的，还是需要马上找人陪你。")
        return plan

    if stage == "human_followup_watch":
        plan.summary = "这已经不只是普通烦躁了，我会继续陪你，但也建议把现实里的支持拉近一点。"
        plan.campus_actions = _prepend(
            plan.campus_actions,
            "如果可以，今天先联系一个现实里可信的人：同学、辅导员、家人或学校心理中心，让他们知道你最近状态不太稳。",
        )
        return plan

    if stage == "boundary_building":
        plan.summary = "你可以不用把事情讲完整，我会先尊重你的边界。我们可以只围绕你现在的感受来，不聊具体人名、地点或细节。"
        plan.immediate_support = _prepend(
            plan.immediate_support,
            "如果担心被知道，你可以把信息说得很模糊，只说“我现在很怕/很烦/很委屈”也可以。",
        )
        plan.follow_up = _prepend(
            plan.follow_up,
            "下一句不用解释原因；你只要告诉我现在更需要“陪着”，还是“一个小办法”。",
        )
        return plan

    if stage in {"high_entropy_stabilization", "deteriorating_watch"}:
        plan.summary = "我先不加重你的负担，也不急着让你分析原因。现在重点是把这一阵强度先降下来一点。"
        plan.immediate_support = _prepend(
            plan.immediate_support,
            "先只做一个动作：坐稳、喝口水、慢慢呼气十秒。我们先处理这一分钟，不处理全部问题。",
        )
        plan.follow_up = _prepend(
            plan.follow_up,
            "如果你愿意，下一句只回一个数字：现在难受程度 0 到 10，大概是多少？",
        )
        return plan

    if stage == "stabilizing":
        plan.summary = "看起来你不是一下子解决了所有问题，而是已经有一点点往稳的方向走。我们先巩固这点，不急着增加新任务。"
        plan.follow_up = _prepend(
            plan.follow_up,
            "你可以回想一下刚才哪一点稍微有用：被陪着、把事情说出来一点，还是那个小动作。我们保留有用的部分就好。",
        )
        return plan

    if "不要追问隐私细节" in avoid_next_turn:
        plan.follow_up = _prepend(
            plan.follow_up,
            "我不会追问具体细节；如果要继续，我们只聊你现在最需要被怎么支持。",
        )

    if "需要先被接住情绪，而不是立刻被分析" in user_needs:
        plan.immediate_support = _prepend(
            plan.immediate_support,
            "先不急着想办法。你现在这份难受本身就值得被认真接住。",
        )

    if recommended_moves and not plan.follow_up:
        plan.follow_up = [recommended_moves[0]]
    return plan


def apply_strategy_reselection_to_plan(
    plan: SupportPlan,
    *,
    strategy_reselection: dict[str, Any] | None,
) -> SupportPlan:
    """If a goal failed, switch the next visible support move instead of repeating it."""

    if not strategy_reselection or not strategy_reselection.get("should_reselect"):
        return plan
    recommended_strategy = str(strategy_reselection.get("recommended_strategy") or "")
    trigger = str(strategy_reselection.get("trigger") or "")

    if recommended_strategy == "safety_first_human_linkage":
        plan.summary = "我先把重点放到安全和现实支持上。现在不需要继续分析原因，先确认你不是一个人硬撑。"
        plan.immediate_support = _prepend(
            plan.immediate_support,
            "如果你现在有伤害自己的冲动，先把危险物品放远，并马上联系身边可信任的人或紧急支持。",
        )
        return plan

    if recommended_strategy == "repair_response_style_before_advice":
        plan.summary = "我先换一种方式回应你：刚才如果显得像在套话或催你解决问题，那不是我想要的效果。我们先回到你现在的感受。"
        plan.immediate_support = _prepend(
            plan.immediate_support,
            "这一轮我先不继续给建议，只先确认：你现在最需要的是被听见，还是希望我帮你把事情变简单一点？",
        )
        return plan

    if recommended_strategy == "repair_trust_and_stop_detail_questions":
        plan.summary = "我会先尊重你的边界，不追问细节。你不需要证明自己为什么难受，也不需要说出会让你不安全的信息。"
        plan.follow_up = _prepend(
            plan.follow_up,
            "如果继续聊，我们只聊感受，不聊具体人名、地点或经过。",
        )
        return plan

    if recommended_strategy == "reduce_task_scope_and_validate_emotion":
        plan.summary = "上一种直接拆任务的方式可能还不够贴近你现在的难受。我们先承认这确实压得你很紧，再只留一个小动作。"
        plan.immediate_support = _prepend(
            plan.immediate_support,
            "先不做完整计划，只选接下来十分钟里最不痛苦的一步，做不到也没关系。",
        )
        return plan

    if recommended_strategy == "switch_to_body_stabilization_then_human_support":
        plan.summary = "现在先不靠讲道理硬撑，身体状态已经在提醒你需要降速。我们先把睡眠、呼吸或吃一点东西放在前面。"
        plan.campus_actions = _prepend(
            plan.campus_actions,
            "如果睡不好、吃不下持续几天，建议联系校医院或心理中心做一次基础支持，不要一个人拖着。",
        )
        return plan

    if recommended_strategy == "move_from_reflection_to_boundary_action":
        plan.summary = "我们不只停在描述难受上，下一步可以做一个很小的边界动作，让你从触发环境里退出来一点。"
        plan.immediate_support = _prepend(
            plan.immediate_support,
            "先给自己一个短暂缓冲：离开宿舍或冲突现场五分钟，去走廊、楼下或一个安静位置。",
        )
        return plan

    if recommended_strategy == "change_micro_intervention_format":
        plan.follow_up = _prepend(
            plan.follow_up,
            "我们换成更省力的方式：你只需要选一个，A 想被陪着，B 想要一个小办法，C 想先安静。",
        )
        return plan

    if trigger in {"latest_worsened", "repeated_failure"}:
        plan.summary = "我注意到前面的方式可能没有帮你降下来，所以这一轮我会放慢，不继续重复同一种建议。"
    return plan


def _privacy_reassurance(plan: SupportPlan) -> SupportPlan:
    plan.summary = (
        "你担心别人知道，这个顾虑很重要。你不用说姓名、宿舍号、具体对象这些能识别身份的信息；"
        "我们可以只聊你的感受和你现在需要什么支持。"
    )
    plan.immediate_support = _prepend(
        plan.immediate_support,
        "如果你不想展开，也完全可以先停在这里。我会尽量按你的节奏来，不会逼你把细节说出来。",
    )
    plan.follow_up = _prepend(
        plan.follow_up,
        "你可以只回我一个方向：现在更需要我安静陪你一下，还是给你一个很小的缓解办法？",
    )
    return plan


def _low_pressure_presence(plan: SupportPlan) -> SupportPlan:
    plan.summary = "可以，不想说也没关系。我们先不急着分析原因，你现在能留在这里已经算是在照顾自己了。"
    plan.immediate_support = _prepend(
        plan.immediate_support,
        "你可以不用解释完整，只要先把注意力放回此刻：慢慢呼一口气，确认自己现在是安全坐着或站着的。",
    )
    plan.follow_up = _prepend(
        plan.follow_up,
        "如果愿意，你只回“陪着”或“建议”都可以；如果不想回，也可以先停一会儿。",
    )
    return plan


def _sleep_stabilization(plan: SupportPlan) -> SupportPlan:
    plan.summary = (
        "睡不着会把压力放大，所以我们先不急着一次解决所有问题。今晚的重点可以先放在让身体稍微降下来。"
    )
    plan.immediate_support = _prepend(
        plan.immediate_support,
        "先做一个很小的动作：把接下来十分钟只设成“躺下、放低屏幕亮度、慢慢呼气”，不要在床上继续推演明天会怎样。",
    )
    plan.self_regulation = _prepend(
        plan.self_regulation,
        "如果脑子一直转，可以把最担心的一件事写成一句话，告诉自己“明天再处理”，先把今晚和明天分开。",
    )
    plan.follow_up = _prepend(
        plan.follow_up,
        "等你稍微缓下来后，我们再把考试压力拆成一个很小的复习步骤，不用现在就想完整计划。",
    )
    return plan


def _grounding_small_step(plan: SupportPlan, state_profile: StateProfile) -> SupportPlan:
    domain = "考试" if "academic" in state_profile.stress_domains else "这件事"
    plan.summary = f"你现在像是被{domain}和压力一起压住了。我们先把目标缩小，不急着证明你必须马上变好。"
    plan.immediate_support = _prepend(
        plan.immediate_support,
        "先选一个十到十五分钟能完成的小动作，比如只看一页、只整理一道题、或者只列出三件最急的事。",
    )
    plan.follow_up = _prepend(
        plan.follow_up,
        "做完这一小步后，再判断下一步；现在不要提前审判整个结果。",
    )
    return plan


def _dorm_boundary_support(plan: SupportPlan) -> SupportPlan:
    plan.summary = "回到宿舍就烦，说明那个环境现在对你很消耗。我们先不急着判断谁对谁错，先让你别继续被它压着。"
    plan.immediate_support = _prepend(
        plan.immediate_support,
        "如果可以，先给自己一个短暂的缓冲区：去走廊、楼下、洗手间或安静角落待五分钟，让身体先离开触发点。",
    )
    plan.follow_up = _prepend(
        plan.follow_up,
        "等情绪降一点后，再决定是先减少接触、换个空间，还是找辅导员/宿管做一次低冲突沟通。",
    )
    return plan


def _future_uncertainty_grounding(plan: SupportPlan) -> SupportPlan:
    plan.summary = "一想到毕业、找工作和未来会很慌，是因为问题一下子被拉得太远太大了。我们先不假装未来已经有答案，只先把视线拉回一个能动的小点。"
    plan.immediate_support = _prepend(
        plan.immediate_support,
        "先写下你最担心的一个词，比如“工作”“城市”“能力”或“选择”，不要一次处理整个人生。",
    )
    plan.follow_up = _prepend(
        plan.follow_up,
        "下一步只做一个近处动作：查一个岗位、问一个学长学姐、整理一条经历，三选一就够。",
    )
    return plan


def _group_work_visibility(plan: SupportPlan) -> SupportPlan:
    plan.summary = "你现在卡住的不只是小组关系，而是自己的参与和贡献可能被看不见。先不急着争对错，目标是让你的工作变得清楚、可证明、可交付。"
    plan.immediate_support = _prepend(
        plan.immediate_support,
        "先发一条具体消息：我可以负责资料汇总/第几部分PPT/展示稿初版，今晚几点前给一版。这样不是抱怨，而是把自己的位置放回项目里。",
    )
    plan.follow_up = _prepend(
        plan.follow_up,
        "如果仍然没人回应，保留聊天记录，主动完成一份可见成果；必要时再向组长或老师做事实陈述，重点说分工过程和你完成的内容。",
    )
    return plan


def _performance_grounding(plan: SupportPlan) -> SupportPlan:
    plan.summary = "你的身体把汇报当成了危险场景，所以提前心跳、手抖，不等于你没准备。现在先把目标从表现完美降到把内容讲完。"
    plan.immediate_support = _prepend(
        plan.immediate_support,
        "准备三个救场锚点：第一页开头句、每部分过渡句、最后总结句。忘词时看一眼锚点，喝一口水，直接回到结构里。",
    )
    plan.self_regulation = _prepend(
        plan.self_regulation,
        "上台前只做一个身体动作：脚踩实地面，慢慢呼气两次，提醒自己可以停顿、可以看稿，不需要像背诵机器一样顺滑。",
    )
    return plan


def _escape_loop_interruption(plan: SupportPlan) -> SupportPlan:
    plan.summary = "游戏现在像一个避难所，能让你暂时不用面对作业、论文和未来；问题不是你废，而是压力太大后形成了逃避循环。"
    plan.immediate_support = _prepend(
        plan.immediate_support,
        "今晚不要设彻底戒掉，只设一个中断点：到点退出账号、手机放远，然后只处理一件现实小事，比如打开文档、写标题或洗漱。",
    )
    plan.campus_actions = _prepend(
        plan.campus_actions,
        "如果已经连续影响上课、睡眠和基本生活，建议找辅导员或心理老师一起做行为计划；求助不是失败，是在阻断循环。",
    )
    return plan


def _family_boundary_sustainability(plan: SupportPlan) -> SupportPlan:
    plan.summary = "你关心家人，但你不应该长期承担父母关系里的情绪中间人角色。支持家人和保护自己，可以同时存在。"
    plan.immediate_support = _prepend(
        plan.immediate_support,
        "可以温和但明确地说：我关心你，但我不能每天听你骂爸爸，这会影响我的学习和睡眠。我们可以每周固定聊一次。",
    )
    plan.campus_actions = _prepend(
        plan.campus_actions,
        "如果家庭情绪长期压到你，可以找辅导员或心理中心讨论边界方案；这不是告状，而是让支持从无限消耗变成可持续。",
    )
    return plan


def _grief_without_self_blame(plan: SupportPlan) -> SupportPlan:
    plan.summary = "失去之后，人很容易反复想如果当初，好像这样就能把结果改回来。但很多事情并不完全由你控制。"
    plan.immediate_support = _prepend(
        plan.immediate_support,
        "先不要逼自己证明“我没错”。可以写下三个你和它在一起的片段，让难过有一个能放下的地方，也承认你曾经照顾它、爱它。",
    )
    plan.follow_up = _prepend(
        plan.follow_up,
        "如果今晚一直反复自责，我们先把问题从“是不是全怪我”改成“我现在需要怎样度过这阵失去后的难受”。",
    )
    return plan


def _safety_reporting_without_blame(plan: SupportPlan) -> SupportPlan:
    plan.summary = "报告安全隐患不等于指控某个人犯罪。你可以只描述事实、时间、地点和你感到的风险。"
    plan.immediate_support = _prepend(
        plan.immediate_support,
        "可以这样说：我在某时间某路段感到被尾随，现在不敢独自经过，希望学校关注照明、巡逻或陪同返回。",
    )
    plan.campus_actions = _prepend(
        plan.campus_actions,
        "今晚尽量不要独自走那段路，可以让同学陪你，或选择人多、有灯光、有监控的路线。",
    )
    return plan


def _supportive_listening(plan: SupportPlan, state_profile: StateProfile) -> SupportPlan:
    if state_profile.primary_state == "sadness_distress":
        plan.summary = "你现在已经难受到想哭了，这不是小题大做，是情绪真的到了很满的位置。先不用急着把原因讲清楚，我会先陪你把这一阵难受接住。"
        plan.immediate_support = _prepend(
            plan.immediate_support,
            "你可以先做一件很小的事：找个相对安全的位置坐下，喝一口水，或者把手放在桌面上感受一下触感。我们先让身体知道，现在这一分钟是安全的。",
        )
        plan.follow_up = _prepend(
            plan.follow_up,
            "如果你愿意，可以只回我一个词：委屈、害怕、累、烦，或者“先陪我一会儿”。不需要马上解释完整。",
        )
        return plan
    if plan.summary.strip():
        return plan
    if state_profile.emotion_signals:
        plan.summary = "听起来这件事确实让你不太好受。我们可以先从最困扰你的那一小块开始。"
    else:
        plan.summary = "我在听。你不用一次说清楚，我们可以慢一点来。"
    return plan


def _prepend(items: list[str], item: str) -> list[str]:
    clean_items = [existing for existing in items if existing.strip() and existing.strip() != item]
    return [item, *clean_items][:5]
