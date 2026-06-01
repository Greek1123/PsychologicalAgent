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
class LocalPolicyInfo:
    policy_name: str
    policy_stage: str
    escalation_hint: str | None = None


@dataclass(slots=True)
class ReferralDecision:
    should_refer: bool
    urgency: str
    reasons: list[str] = field(default_factory=list)
    recommended_channel: str | None = None


@dataclass(slots=True)
class StateProfile:
    primary_state: str
    intensity: int
    confidence: float
    stress_domains: list[str] = field(default_factory=list)
    emotion_signals: list[str] = field(default_factory=list)
    body_signals: list[str] = field(default_factory=list)
    cognitive_signals: list[str] = field(default_factory=list)
    social_signals: list[str] = field(default_factory=list)
    boundary_flags: list[str] = field(default_factory=list)
    risk_signals: list[str] = field(default_factory=list)
    weak_input_detected: bool = False
    noisy_input_detected: bool = False
    recommended_focus: str = "supportive_listening"
    evidence: list[str] = field(default_factory=list)


@dataclass(slots=True)
class InterventionStrategy:
    strategy_id: str
    priority: str
    response_mode: str
    user_visible_goal: str
    hidden_clinical_goal: str
    should_ask_question: bool
    max_questions: int
    suggested_opening: str
    next_step: str
    avoid: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DynamicAdjustment:
    adjustment_id: str
    stability_state: str
    action: str
    intensity_shift: str
    trend_direction: str
    trend_delta: int | None
    should_modify_strategy: bool
    should_refer: bool
    review_window_hours: int
    next_focus: str
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FeedbackAdaptation:
    adaptation_id: str
    mode: str
    question_pressure: str
    detail_level: str
    should_avoid_repetition: bool
    should_collect_bad_case: bool
    avoid_tags: list[str] = field(default_factory=list)
    preferred_moves: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EntropyOrchestration:
    orchestration_id: str
    route: str
    user_visible_goal: str
    next_focus: str
    response_constraints: list[str] = field(default_factory=list)
    hidden_actions: list[str] = field(default_factory=list)
    memory_topics: list[str] = field(default_factory=list)
    boundary_flags: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EntropyReductionGoal:
    goal_id: str
    active_driver: str
    reduction_goal: str
    target_entropy_delta: int
    micro_intervention: str
    review_condition: str
    priority: str
    success_signal: str
    avoid: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GoalAttainmentEvaluation:
    evaluation_id: str
    status: str
    completion_score: int
    entropy_delta: int | None
    risk_shift: str
    driver: str | None
    goal_text: str | None
    interpretation: str
    next_adjustment: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StrategyReselectionDecision:
    decision_id: str
    should_reselect: bool
    trigger: str
    from_driver: str | None
    recommended_strategy: str
    recommended_goal_driver: str
    priority: str
    rationale: str
    constraints: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReferralExplanation:
    explanation_id: str
    should_escalate: bool
    referral_level: str
    recommended_channel: str
    user_visible_reason: str
    backend_reason: str
    urgency: str
    trigger_reasons: list[str] = field(default_factory=list)
    protective_notes: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InterventionAuditLog:
    audit_id: str
    response_id: str | None
    session_id: str | None
    created_at: str | None
    decision_route: str
    risk_snapshot: dict[str, Any]
    entropy_snapshot: dict[str, Any]
    selected_goal: dict[str, Any]
    strategy_snapshot: dict[str, Any]
    referral_snapshot: dict[str, Any]
    explanation: str
    backend_trace: list[str] = field(default_factory=list)
    user_visible_summary: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EntropyAdjustmentLoop:
    loop_id: str
    loop_action: str
    priority: str
    next_reply_mode: str
    question_policy: str
    memory_policy: str
    human_followup_policy: str
    rationale: str
    constraints: list[str] = field(default_factory=list)
    preferred_moves: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LongitudinalStateProfile:
    profile_id: str
    session_id: str
    observation_count: int
    dominant_states: list[str]
    dominant_stress_domains: list[str]
    entropy_course: str
    risk_course: str
    average_entropy: float | None
    latest_entropy_score: int | None
    peak_entropy_score: int | None
    volatility_score: int
    engagement_signal: str
    recommended_care_level: str
    next_review_hours: int
    priority_actions: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EntropyTrendWarning:
    warning_id: str
    session_id: str
    level: str
    trend_state: str
    should_alert: bool
    review_window_hours: int
    recommended_action: str
    user_visible_mode: str
    trigger_reasons: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CarePathwayDecision:
    pathway_id: str
    session_id: str
    route: str
    priority: str
    user_visible_mode: str
    review_window_hours: int
    should_notify_human: bool
    should_pause_ai_only_reply: bool
    backend_actions: list[str] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EntropyReductionOutcome:
    outcome_id: str
    session_id: str
    status: str
    effectiveness_score: int
    entropy_delta: int | None
    risk_shift: str
    feedback_signal: str
    pathway_signal: str
    summary: str
    next_action: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CareQueueItem:
    session_id: str
    priority: str
    route: str
    outcome_status: str
    recommended_action: str
    latest_entropy_score: int | None
    risk_level: str | None
    created_at: str | None
    response_id: str | None
    trend_warning_level: str | None = None
    trend_state: str | None = None
    review_window_hours: int | None = None
    queue_score: int = 0
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SessionCarePlan:
    plan_id: str
    session_id: str
    care_phase: str
    priority: str
    review_window_hours: int
    primary_goal: str
    user_visible_focus: str
    backend_focus: str
    next_actions: list[str] = field(default_factory=list)
    avoid_actions: list[str] = field(default_factory=list)
    success_indicators: list[str] = field(default_factory=list)
    escalation_conditions: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MultimodalSignal:
    modality: str
    filename: str
    content_type: str | None
    byte_size: int
    analysis_available: bool
    format: str | None = None
    duration_seconds: float | None = None
    sample_rate_hz: int | None = None
    channels: int | None = None
    sample_width_bits: int | None = None
    rms_energy: float | None = None
    peak_amplitude: float | None = None
    silence_ratio: float | None = None
    analysis_notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ProcessingSummary:
    route: str
    input_mode: str
    reply_source: str
    safety_priority: str
    risk_level: str
    entropy_score: int
    balance_state: str
    primary_state: str | None
    strategy_id: str | None
    dynamic_action: str | None
    orchestration_route: str | None
    referral_urgency: str | None
    should_refer: bool
    local_policy_name: str | None
    completed_stages: list[str] = field(default_factory=list)
    decision_reasons: list[str] = field(default_factory=list)
    next_backend_action: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LocalPolicyResult:
    assessment: SupportAssessment
    plan: SupportPlan
    info: LocalPolicyInfo

    def __iter__(self):
        yield self.assessment
        yield self.plan


@dataclass(slots=True)
class SupportResponse:
    response_id: str
    source: str
    input_text: str
    transcript: str | None
    reply_text: str
    risk: RiskAssessment
    entropy: PsychologicalEntropy
    entropy_reduction: EntropyReductionStrategy
    assessment: SupportAssessment
    plan: SupportPlan
    campus_resources: list[CampusResource]
    safety: SafetyNotice
    metadata: AgentMetadata
    local_policy: LocalPolicyInfo | None = None
    referral_decision: ReferralDecision | None = None
    state_profile: StateProfile | None = None
    intervention_strategy: InterventionStrategy | None = None
    dynamic_adjustment: DynamicAdjustment | None = None
    feedback_adaptation: FeedbackAdaptation | None = None
    entropy_orchestration: EntropyOrchestration | None = None
    reduction_goal: EntropyReductionGoal | None = None
    referral_explanation: ReferralExplanation | None = None
    adjustment_loop: EntropyAdjustmentLoop | None = None
    multimodal_signal: MultimodalSignal | None = None
    processing_summary: ProcessingSummary | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def new_metadata(model_backend: str) -> AgentMetadata:
    return AgentMetadata(
        model_backend=model_backend,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def new_response_id() -> str:
    return f"support_{uuid4().hex}"
