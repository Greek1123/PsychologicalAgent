from __future__ import annotations

from .schemas import (
    EntropyReductionStrategy,
    InterventionStrategy,
    PsychologicalEntropy,
    RiskAssessment,
    RiskLevel,
    StateProfile,
)


def select_intervention_strategy(
    *,
    state_profile: StateProfile,
    risk: RiskAssessment,
    entropy: PsychologicalEntropy,
    entropy_reduction: EntropyReductionStrategy,
) -> InterventionStrategy:
    """Choose the backend response strategy for the current turn.

    This is the layer between analysis and generation. It keeps professional
    reasoning hidden from the user while giving the model/backend a clear route:
    safety, confidentiality, low-pressure presence, sleep stabilization, etc.
    """

    if risk.level in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
        return InterventionStrategy(
            strategy_id="safety_first",
            priority="urgent",
            response_mode="safety_referral",
            user_visible_goal="Make the user safer in the next few minutes.",
            hidden_clinical_goal="Reduce immediate risk and move support into the real world.",
            should_ask_question=False,
            max_questions=0,
            suggested_opening="我先把安全放在第一位。",
            next_step="Encourage contacting a trusted person or emergency/campus support immediately.",
            avoid=["lengthy analysis", "debating risk", "generic reassurance"],
            tags=["risk", "referral", "safety"],
        )

    focus = state_profile.recommended_focus
    if focus == "contribution_visibility":
        return InterventionStrategy(
            strategy_id="group_work_visibility",
            priority="medium",
            response_mode="visibility_and_boundary",
            user_visible_goal="Help the user make their contribution visible without escalating conflict.",
            hidden_clinical_goal="Reduce helpless waiting and convert social exclusion anxiety into concrete agency.",
            should_ask_question=False,
            max_questions=0,
            suggested_opening="你现在难受的不只是被忽略，还有担心自己的努力最后看不见。",
            next_step="Validate the marginalization, then suggest one visible deliverable, saved chat records, and factual escalation if needed.",
            avoid=["telling the user to simply confront the group", "encouraging passive waiting", "turning it into blame"],
            tags=["group_work", "visibility", "agency", *entropy_reduction.targeted_drivers[:1]],
        )

    if focus == "performance_grounding":
        return InterventionStrategy(
            strategy_id="performance_grounding",
            priority="medium",
            response_mode="body_grounding_and_anchor_plan",
            user_visible_goal="Reduce presentation panic and keep the task executable.",
            hidden_clinical_goal="Shift from threat response to structured performance cues.",
            should_ask_question=False,
            max_questions=0,
            suggested_opening="你的身体把汇报当成了危险场景，所以提前紧张不等于你没准备。",
            next_step="Lower the goal from perfect performance to completion, then give anchor sentences and fallback moves.",
            avoid=["demanding confidence", "long rehearsal plan", "telling the user not to be nervous"],
            tags=["presentation", "panic", "grounding"],
        )

    if focus == "escape_loop_interruption":
        return InterventionStrategy(
            strategy_id="escape_loop_interruption",
            priority="medium",
            response_mode="avoidance_loop_repair",
            user_visible_goal="Interrupt the game-avoidance loop without shaming the user.",
            hidden_clinical_goal="Name the avoidance cycle and create one low-friction offline interruption.",
            should_ask_question=False,
            max_questions=0,
            suggested_opening="游戏现在像一个避难所，问题不是你废，而是现实任务太压人。",
            next_step="Name the pressure-game-guilt loop and set one shutdown point plus one tiny real-world task.",
            avoid=["moralizing", "calling the user lazy", "asking for a full life plan"],
            tags=["avoidance", "daily_functioning", "behavior_loop"],
        )

    if focus == "family_boundary_sustainability":
        return InterventionStrategy(
            strategy_id="family_boundary_sustainability",
            priority="medium",
            response_mode="sustainable_family_boundary",
            user_visible_goal="Help the user care without becoming the family's emotional container.",
            hidden_clinical_goal="Convert unlimited emotional labor into bounded, sustainable support.",
            should_ask_question=False,
            max_questions=0,
            suggested_opening="你关心家人，但你不应该长期承担父母关系里的情绪中间人角色。",
            next_step="Offer a warm but clear boundary sentence and suggest widening support beyond the student.",
            avoid=["accusing parents", "telling the user to cut contact", "making the user responsible for fixing the family"],
            tags=["family", "boundary", "sustainability"],
        )

    if focus == "grief_without_self_blame":
        return InterventionStrategy(
            strategy_id="grief_without_self_blame",
            priority="medium",
            response_mode="grief_and_self_blame_relief",
            user_visible_goal="Hold grief while reducing excessive self-blame.",
            hidden_clinical_goal="Separate loss pain from total responsibility attribution.",
            should_ask_question=False,
            max_questions=0,
            suggested_opening="失去之后，人很容易反复想如果当初，好像这样能把结果改回来。",
            next_step="Validate grief, soften total responsibility, and suggest a small farewell action.",
            avoid=["rushing acceptance", "saying it was not important", "debating facts the user cannot verify"],
            tags=["grief", "self_blame", "loss"],
        )

    if focus == "safety_reporting_without_blame":
        return InterventionStrategy(
            strategy_id="safety_reporting_without_blame",
            priority="high",
            response_mode="practical_safety_reporting",
            user_visible_goal="Make self-protection feel legitimate even without perfect evidence.",
            hidden_clinical_goal="Move from fear paralysis to factual safety reporting and supported movement.",
            should_ask_question=False,
            max_questions=0,
            suggested_opening="报告安全隐患不等于指控某个人犯罪，你可以只描述事实和风险。",
            next_step="Suggest factual reporting language and avoid walking alone through the same route.",
            avoid=["dismissing fear", "requiring proof before seeking help", "encouraging confrontation"],
            tags=["campus_safety", "reporting", "support"],
        )

    if focus == "confidentiality_and_control":
        return InterventionStrategy(
            strategy_id="privacy_reassurance",
            priority="high",
            response_mode="trust_boundary",
            user_visible_goal="Help the user feel safer to stay in the conversation without pressure.",
            hidden_clinical_goal="Stabilize disclosure boundary and preserve user autonomy.",
            should_ask_question=True,
            max_questions=1,
            suggested_opening="你担心别人知道，这个顾虑很重要。",
            next_step="Clarify privacy limits, avoid asking identifying details, and offer a low-pressure choice.",
            avoid=["pushing for details", "clinical labels", "asking who exactly is involved"],
            tags=["privacy", "boundary", "rapport"],
        )

    if focus == "low_pressure_presence":
        return InterventionStrategy(
            strategy_id="low_pressure_presence",
            priority="medium",
            response_mode="minimal_support",
            user_visible_goal="Keep connection without making the user explain.",
            hidden_clinical_goal="Avoid withdrawal after weak or boundary-setting input.",
            should_ask_question=True,
            max_questions=1,
            suggested_opening="可以，不想说也没关系。",
            next_step="Offer either quiet company or one very small grounding action.",
            avoid=["why questions", "multiple choices that feel like a test", "long advice"],
            tags=["weak_input", "boundary", "rapport"],
        )

    if focus == "sleep_stabilization_first":
        return InterventionStrategy(
            strategy_id="sleep_stabilization",
            priority="medium",
            response_mode="practical_stabilization",
            user_visible_goal="Reduce tonight's physiological load before solving everything else.",
            hidden_clinical_goal="Target sleep disruption as a near-term entropy driver.",
            should_ask_question=True,
            max_questions=1,
            suggested_opening="睡不着会把压力放大，我们先别急着解决所有事。",
            next_step="Give one short sleep-stabilizing action and suggest reviewing academic pressure after rest.",
            avoid=["full study plan immediately", "blaming lifestyle", "overly technical sleep education"],
            tags=["sleep", "stabilization", *entropy_reduction.targeted_drivers[:1]],
        )

    if focus == "grounding_then_small_next_step":
        return InterventionStrategy(
            strategy_id="grounding_small_step",
            priority="medium",
            response_mode="ground_then_plan",
            user_visible_goal="Lower overwhelm and make the next action small enough to start.",
            hidden_clinical_goal="Reduce cognitive load and catastrophic thinking.",
            should_ask_question=True,
            max_questions=1,
            suggested_opening="你现在像是被很多事情一起压住了。",
            next_step="Reflect the pressure, then propose one 10-15 minute action.",
            avoid=["large plans", "lecturing", "telling the user to simply relax"],
            tags=["cognitive_load", "academic", *entropy_reduction.targeted_drivers[:1]],
        )

    if state_profile.primary_state == "future_uncertainty":
        return InterventionStrategy(
            strategy_id="future_uncertainty_grounding",
            priority="medium",
            response_mode="future_to_near_term",
            user_visible_goal="Reduce future panic by bringing attention back to a near-term controllable step.",
            hidden_clinical_goal="Contain uncertainty spiral without pretending the future is solved.",
            should_ask_question=True,
            max_questions=1,
            suggested_opening="一想到毕业和找工作就慌，这种不确定感确实会把人一下子推远。",
            next_step="Name the future worry, then choose one near-term controllable action.",
            avoid=["empty reassurance", "full career planning immediately", "claiming everything will be fine"],
            tags=["future", "uncertainty", "grounding"],
        )

    if focus == "safe_space_and_boundary_options":
        return InterventionStrategy(
            strategy_id="dorm_boundary_support",
            priority="medium",
            response_mode="situational_boundary",
            user_visible_goal="Help the user feel less trapped in the shared-living situation.",
            hidden_clinical_goal="Separate emotional validation from practical boundary planning.",
            should_ask_question=True,
            max_questions=1,
            suggested_opening="回到宿舍就烦，说明那个环境现在很消耗你。",
            next_step="Offer one immediate decompression option and one boundary/space option.",
            avoid=["telling user to directly confront roommates immediately", "taking sides too strongly"],
            tags=["dorm", "interpersonal", "boundary"],
        )

    return InterventionStrategy(
        strategy_id="supportive_listening",
        priority="normal",
        response_mode="reflect_and_invite",
        user_visible_goal="Make the user feel heard and keep the conversation moving gently.",
        hidden_clinical_goal="Collect enough context for later state tracking without pressure.",
        should_ask_question=True,
        max_questions=1,
        suggested_opening="听起来这件事确实让你不太好受。",
        next_step="Reflect emotion, name the likely stressor, and invite one small clarification.",
        avoid=["diagnosis", "professional jargon", "too many questions"],
        tags=["support", entropy.balance_state],
    )
