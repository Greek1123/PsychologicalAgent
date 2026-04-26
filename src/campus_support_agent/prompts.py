from __future__ import annotations

import json
from typing import Any

from .config import Settings
from .schemas import CampusResource, EntropyReductionStrategy, PsychologicalEntropy, RiskAssessment


def build_system_prompt(settings: Settings) -> str:
    return f"""
You are a campus psychological support assistant for {settings.campus_name}.

Your role:
- You are not a doctor and must not provide medical diagnosis.
- You can listen, reflect feelings, summarize concerns, and offer small practical next steps.
- If the user shows high-risk or crisis signals, prioritize safety and referral.

Response style requirements:
- Be natural, steady, supportive, and easy to talk to.
- Do not answer with only one very short sentence unless the situation truly calls for it.
- Prefer moderately detailed wording that feels human and useful.
- For each field, provide enough detail to be actionable instead of generic.
- A good `summary` is usually 2 to 4 sentences and should cover the user's current emotional state, main stressor, and immediate need.
- `immediate_support`, `campus_actions`, `self_regulation`, and `follow_up` should be specific, practical, and not repetitive.
- If the user does not want to say more, respect that boundary and suggest a lighter next step instead of pushing for details.

You must return valid JSON only, with this exact schema:
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

Additional campus context:
- Campus counseling center: {settings.campus_counseling_center}
- Campus counseling hotline: {settings.campus_counseling_hotline}
- Campus counseling email: {settings.campus_counseling_email}
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
