from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(slots=True)
class RiskAssessment:
    level: RiskLevel
    score: int
    reason: str
    trigger_terms: list[str] = field(default_factory=list)
    needs_human_followup: bool = False


@dataclass(slots=True)
class SupportAssessment:
    primary_emotions: list[str]
    stressors: list[str]
    protective_factors: list[str]
    entropy_level: int
    balance_state: str


@dataclass(slots=True)
class EntropyDimensions:
    emotion_intensity: int
    emotional_volatility: int
    cognitive_load: int
    physiological_imbalance: int
    social_support_tension: int
    risk_pressure: int


@dataclass(slots=True)
class EntropyTrend:
    previous_score: int | None
    delta: int | None
    direction: str


@dataclass(slots=True)
class PsychologicalEntropy:
    score: int
    level: int
    balance_state: str
    driver_tags: list[str]
    dominant_drivers: list[str]
    dimensions: EntropyDimensions
    trend: EntropyTrend


@dataclass(slots=True)
class EntropyReductionStrategy:
    target_state: str
    targeted_drivers: list[str]
    rationale: str
    core_actions: list[str]
    expected_delta_score: int
    review_window_hours: int


@dataclass(slots=True)
class SupportPlan:
    summary: str
    immediate_support: list[str]
    campus_actions: list[str]
    self_regulation: list[str]
    follow_up: list[str]


@dataclass(slots=True)
class SafetyNotice:
    disclaimer: str
    emergency_notice: str | None
    human_referral: str | None


@dataclass(slots=True)
class CampusResource:
    resource_id: str
    title: str
    category: str
    summary: str
    recommended_actions: list[str]
    relevance_reason: str


@dataclass(slots=True)
class AgentMetadata:
    model_backend: str
    generated_at: str


@dataclass(slots=True)
class SupportResponse:
    response_id: str
    source: str
    input_text: str
    transcript: str | None
    risk: RiskAssessment
    entropy: PsychologicalEntropy
    entropy_reduction: EntropyReductionStrategy
    assessment: SupportAssessment
    plan: SupportPlan
    campus_resources: list[CampusResource]
    safety: SafetyNotice
    metadata: AgentMetadata

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def new_metadata(model_backend: str) -> AgentMetadata:
    return AgentMetadata(
        model_backend=model_backend,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def new_response_id() -> str:
    return f"support_{uuid4().hex}"
