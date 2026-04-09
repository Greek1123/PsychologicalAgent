from __future__ import annotations

from .logging_utils import get_logger
from .schemas import CampusResource, EntropyReductionStrategy, PsychologicalEntropy, RiskAssessment, RiskLevel


logger = get_logger("reduction")


DRIVER_LABELS = {
    "emotion_intensity": "情绪强度",
    "emotional_volatility": "情绪波动",
    "cognitive_load": "认知负荷",
    "physiological_imbalance": "生理失衡",
    "social_support_tension": "社会支持张力",
    "risk_pressure": "风险压力",
}


DRIVER_PLAYBOOK = {
    "emotion_intensity": {
        "objective": "先把情绪激活水平降下来，避免继续被情绪推着走。",
        "actions": [
            "先暂停 3 分钟，只做缓慢呼吸和放松肩颈，不继续逼自己思考结果。",
            "把当前情绪写成 1 句话，区分事实、感受和最担心的结果。",
        ],
    },
    "emotional_volatility": {
        "objective": "先降低波动和失控感，恢复最基本的稳定感。",
        "actions": [
            "把接下来 2 小时安排得更具体，只保留 1 件必须完成的事。",
            "如果情绪在快速上冲，先离开冲突现场或高刺激场景 10-15 分钟。",
        ],
    },
    "cognitive_load": {
        "objective": "先削减灾难化思考和任务堆积感，降低认知熵。",
        "actions": [
            "把任务拆成 15 分钟一段，先只做第一段，不提前想完整周计划。",
            "列出最坏结果、最可能结果、现在能做的一步，防止脑内循环扩大。",
        ],
    },
    "physiological_imbalance": {
        "objective": "先恢复睡眠和身体节律，减少高熵状态的持续供能。",
        "actions": [
            "今晚优先保睡眠，不再用继续熬夜来换短期进度。",
            "记录最近 3 天的睡眠、饮食和疲惫情况，必要时去校医院评估。",
        ],
    },
    "social_support_tension": {
        "objective": "先补足现实支持，避免独自承受导致熵继续上升。",
        "actions": [
            "把问题同步给 1 个可信任的人，只说现状和你希望得到的帮助。",
            "如果涉及室友或人际冲突，改成预约式沟通，不在情绪高点继续争执。",
        ],
    },
    "risk_pressure": {
        "objective": "先阻止风险继续升级，优先保证现实安全。",
        "actions": [
            "不要独自承受，立刻联系辅导员、家人或值班人员建立陪伴。",
            "如果已经出现明确自伤/伤人念头，立即转入校园或当地紧急求助流程。",
        ],
    },
}


TARGET_STATE_FLOW = {
    "crisis": "fragile",
    "fragile": "strained",
    "strained": "stable",
    "stable": "stable",
}


def build_entropy_reduction_strategy(
    entropy: PsychologicalEntropy,
    risk: RiskAssessment,
    campus_resources: list[CampusResource],
) -> EntropyReductionStrategy:
    tags = entropy.driver_tags or ["cognitive_load"]
    objectives: list[str] = []
    actions: list[str] = []

    # 这里用 driver playbook 把“熵”翻译成真正可执行的干预动作。
    for tag in tags[:2]:
        playbook = DRIVER_PLAYBOOK.get(tag)
        if not playbook:
            continue
        objectives.append(playbook["objective"])
        actions.extend(playbook["actions"])

    # 把校园资源的第一条动作并入减熵策略，让方案更贴近校园落地。
    for resource in campus_resources[:2]:
        if resource.recommended_actions:
            actions.append(resource.recommended_actions[0])

    deduped_actions: list[str] = []
    for action in actions:
        clean = action.strip()
        if clean and clean not in deduped_actions:
            deduped_actions.append(clean)

    if risk.level in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
        expected_delta_score = -5
        review_window_hours = 6
    elif entropy.level >= 4:
        expected_delta_score = -12
        review_window_hours = 24
    elif entropy.level == 3:
        expected_delta_score = -10
        review_window_hours = 48
    else:
        expected_delta_score = -6
        review_window_hours = 72

    rationale = "；".join(objectives[:2]) if objectives else "先稳定情绪和节律，再削减当前最强压力源。"
    targeted_drivers = [DRIVER_LABELS.get(tag, tag) for tag in tags[:2]]
    strategy = EntropyReductionStrategy(
        target_state=TARGET_STATE_FLOW.get(entropy.balance_state, "stable"),
        targeted_drivers=targeted_drivers or ["认知负荷"],
        rationale=rationale,
        core_actions=deduped_actions[:5],
        expected_delta_score=expected_delta_score,
        review_window_hours=review_window_hours,
    )
    logger.info(
        "Built entropy reduction strategy target_state=%s targeted_drivers=%s expected_delta=%s",
        strategy.target_state,
        strategy.targeted_drivers,
        strategy.expected_delta_score,
    )
    return strategy
