from __future__ import annotations

import json
from typing import Any

from .config import Settings
from .schemas import CampusResource, EntropyReductionStrategy, PsychologicalEntropy, RiskAssessment


def build_system_prompt(settings: Settings) -> str:
    return f"""
你是一个面向高校场景的心理支持 Agent，服务对象主要是大学生。

你的任务：
1. 识别学生当前的情绪、压力源和保护因子。
2. 用支持性、非评判、易执行的语言给出建议。
3. 不做心理诊断，不承诺治疗效果，不替代专业心理咨询。
4. 优先给出校园场景可落地方案，如心理中心、辅导员、课程压力拆解、睡眠和作息干预、社交支持。
5. 输出必须是 JSON，对应字段如下：

{{
  "primary_emotions": ["string"],
  "stressors": ["string"],
  "protective_factors": ["string"],
  "entropy_level": 1,
  "balance_state": "stable|strained|fragile",
  "summary": "string",
  "immediate_support": ["string"],
  "campus_actions": ["string"],
  "self_regulation": ["string"],
  "follow_up": ["string"]
}}

额外要求：
- 熵水平 1-5：数值越高表示状态越混乱、越需要外部支持。
- 优先围绕“熵减”给出建议，也就是先稳定情绪和节律，再降低关键压力源。
- 如果给定了 entropy_reduction_strategy，优先围绕 targeted_drivers 和 core_actions 组织方案。
- 建议要短、清晰、可以马上执行。
- 如果给定了 campus_resources，优先从中选择最相关的校园资源来组织 campus_actions。
- 用简体中文输出。
- 不要输出 JSON 以外的文字。

当前学校信息：
- 学校：{settings.campus_name}
- 校园支持资源：{settings.campus_counseling_center}
- 心理热线：{settings.campus_counseling_hotline}
- 联系邮箱：{settings.campus_counseling_email}
""".strip()


def build_user_prompt(
    text: str,
    student_context: dict[str, Any],
    conversation_history: list[dict[str, Any]],
    risk: RiskAssessment,
    entropy: PsychologicalEntropy,
    entropy_reduction: EntropyReductionStrategy,
    campus_resources: list[CampusResource],
) -> str:
    history = conversation_history[-6:] if conversation_history else []
    payload = {
        "student_text": text,
        "student_context": student_context or {},
        "conversation_history": history,
        "risk_assessment": {
            "level": risk.level,
            "score": risk.score,
            "reason": risk.reason,
            "trigger_terms": risk.trigger_terms,
        },
        "psychological_entropy": {
            "score": entropy.score,
            "level": entropy.level,
            "balance_state": entropy.balance_state,
            "driver_tags": entropy.driver_tags,
            "dominant_drivers": entropy.dominant_drivers,
            "dimensions": {
                "emotion_intensity": entropy.dimensions.emotion_intensity,
                "emotional_volatility": entropy.dimensions.emotional_volatility,
                "cognitive_load": entropy.dimensions.cognitive_load,
                "physiological_imbalance": entropy.dimensions.physiological_imbalance,
                "social_support_tension": entropy.dimensions.social_support_tension,
                "risk_pressure": entropy.dimensions.risk_pressure,
            },
        },
        "entropy_reduction_strategy": {
            "target_state": entropy_reduction.target_state,
            "targeted_drivers": entropy_reduction.targeted_drivers,
            "rationale": entropy_reduction.rationale,
            "core_actions": entropy_reduction.core_actions,
            "expected_delta_score": entropy_reduction.expected_delta_score,
            "review_window_hours": entropy_reduction.review_window_hours,
        },
        "campus_resources": [
            {
                "title": item.title,
                "category": item.category,
                "summary": item.summary,
                "recommended_actions": item.recommended_actions,
                "relevance_reason": item.relevance_reason,
            }
            for item in campus_resources
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
