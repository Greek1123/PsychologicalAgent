from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from .config import Settings
from .care_plan_execution import apply_session_care_plan_to_plan
from .dynamic_adjustment import build_dynamic_adjustment
from .entropy_orchestration import build_entropy_orchestration
from .entropy import evaluate_psychological_entropy
from .feedback_adaptation import build_feedback_adaptation
from .final_reply_guardrails import finalize_user_visible_reply
from .intervention_strategy import select_intervention_strategy
from .intervention_next_step import apply_intervention_next_step_to_plan
from .logging_utils import get_logger
from .local_response_policy import maybe_build_local_support_plan
from .multimodal_signal import analyze_audio_signal
from .noisy_input import analyze_noisy_distress_text
from .prompts import build_system_prompt, build_user_prompt
from .providers import LLMProvider, STTProvider
from .reduction import build_entropy_reduction_strategy
from .reduction_goal import build_entropy_reduction_goal
from .referral_explanation import build_referral_explanation
from .response_guardrails import sanitize_user_visible_reply
from .retrieval import CampusKnowledgeRetriever
from .safety import evaluate_text_risk
from .session_tracking import apply_session_tracking_to_plan
from .strategy_versioning import apply_strategy_version_to_plan
from .schemas import (
    CampusResource,
    EntropyReductionStrategy,
    EntropyTrend,
    FeedbackAdaptation,
    EntropyOrchestration,
    EntropyReductionGoal,
    PsychologicalEntropy,
    ReferralDecision,
    ReferralExplanation,
    RiskAssessment,
    RiskLevel,
    SafetyNotice,
    StateProfile,
    SupportAssessment,
    SupportPlan,
    SupportResponse,
    new_metadata,
    new_response_id,
)
from .state_profile import build_state_profile
from .strategy_execution import (
    apply_adjustment_loop_to_plan,
    apply_dynamic_adjustment_to_plan,
    apply_feedback_adaptation_to_plan,
    apply_intervention_strategy_to_plan,
    apply_session_continuity_to_plan,
    apply_strategy_reselection_to_plan,
)


logger = get_logger("agent")


def _extract_session_continuity(student_context: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(student_context, dict):
        return None
    continuity = student_context.get("session_continuity")
    return continuity if isinstance(continuity, dict) else None


def _extract_strategy_reselection(student_context: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(student_context, dict):
        return None
    decision = student_context.get("strategy_reselection")
    return decision if isinstance(decision, dict) else None


def _extract_adjustment_loop(student_context: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(student_context, dict):
        return None
    loop = student_context.get("adjustment_loop")
    return loop if isinstance(loop, dict) else None


def _extract_session_care_plan(student_context: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(student_context, dict):
        return None
    care_plan = student_context.get("session_care_plan")
    return care_plan if isinstance(care_plan, dict) else None


def _extract_intervention_next_step(student_context: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(student_context, dict):
        return None
    next_step = student_context.get("intervention_next_step")
    return next_step if isinstance(next_step, dict) else None


def _extract_session_tracking(student_context: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(student_context, dict):
        return None
    tracking = student_context.get("session_tracking")
    return tracking if isinstance(tracking, dict) else None


def _extract_strategy_version(student_context: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(student_context, dict):
        return None
    version = student_context.get("strategy_version")
    return version if isinstance(version, dict) else None


def _should_carry_forward_dangerous_place_risk(
    text: str,
    conversation_history: list[dict[str, Any]] | None,
    risk: RiskAssessment,
) -> bool:
    if risk.level in {RiskLevel.HIGH, RiskLevel.CRITICAL} or not conversation_history:
        return False
    compact_text = text.replace(" ", "")
    is_followup = any(term in compact_text for term in ("怎么办", "烦躁", "脑子很乱", "？", "?"))
    if not is_followup:
        return False
    recent = "".join(str(item.get("content", "")) for item in conversation_history[-6:]).replace(" ", "")
    has_dangerous_place = any(term in recent for term in ("天台", "楼顶", "高处", "桥上", "河边"))
    has_unresolved_safety = any(term in recent for term in ("不要一个人", "状态不太安全", "危险", "需要人陪", "紧急"))
    return has_dangerous_place and has_unresolved_safety


class CampusSupportAgent:
    def __init__(
        self,
        settings: Settings,
        llm_provider: LLMProvider,
        stt_provider: STTProvider,
        retriever: CampusKnowledgeRetriever | None = None,
    ) -> None:
        self.settings = settings
        self.llm_provider = llm_provider
        self.stt_provider = stt_provider
        self.retriever = retriever

    def handle_text(
        self,
        *,
        text: str,
        student_context: dict[str, Any] | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
        previous_entropy_score: int | None = None,
        entropy_trace: list[dict[str, Any]] | None = None,
        feedback_adaptation: FeedbackAdaptation | None = None,
        source: str = "text",
        transcript: str | None = None,
    ) -> SupportResponse:
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("text 不能为空。")
        continuity_summary = _extract_session_continuity(student_context)
        strategy_reselection = _extract_strategy_reselection(student_context)
        adjustment_loop = _extract_adjustment_loop(student_context)
        session_care_plan = _extract_session_care_plan(student_context)
        intervention_next_step = _extract_intervention_next_step(student_context)
        session_tracking = _extract_session_tracking(student_context)
        strategy_version = _extract_strategy_version(student_context)

        # Analyze a cautious copy of the text so typo-heavy distress like
        # "我不想或了" is still routed as a possible crisis signal.
        noisy_analysis = analyze_noisy_distress_text(clean_text)
        analysis_text = noisy_analysis.analysis_text

        # 先做安全风控，再做心理熵评估，保证危机信号优先被处理。
        risk = evaluate_text_risk(analysis_text)
        if _should_carry_forward_dangerous_place_risk(clean_text, conversation_history, risk):
            risk = RiskAssessment(
                level=RiskLevel.CRITICAL,
                score=90,
                reason="Recent unresolved dangerous-place crisis context is carried forward for a follow-up message.",
                trigger_terms=["dangerous_place_followup"],
                needs_human_followup=True,
            )
        entropy = evaluate_psychological_entropy(
            analysis_text,
            risk,
            student_context=student_context,
            conversation_history=conversation_history,
        )
        if previous_entropy_score is not None:
            delta = entropy.score - previous_entropy_score
            entropy.trend = EntropyTrend(
                previous_score=previous_entropy_score,
                delta=delta,
                direction="up" if delta > 0 else "down" if delta < 0 else "flat",
            )
        state_profile = build_state_profile(
            clean_text,
            risk=risk,
            entropy=entropy,
            conversation_history=conversation_history,
            noisy_input_detected=noisy_analysis.has_inference,
        )
        logger.info(
            "Text request analyzed with risk=%s score=%s entropy=%s state=%s source=%s",
            risk.level,
            risk.score,
            entropy.score,
            state_profile.primary_state,
            source,
        )

        campus_resources = self._retrieve_campus_resources(analysis_text, risk)
        entropy_reduction = build_entropy_reduction_strategy(entropy, risk, campus_resources)
        intervention_strategy = select_intervention_strategy(
            state_profile=state_profile,
            risk=risk,
            entropy=entropy,
            entropy_reduction=entropy_reduction,
        )
        dynamic_adjustment = build_dynamic_adjustment(
            entropy=entropy,
            risk=risk,
            state_profile=state_profile,
            entropy_trace=entropy_trace,
        )
        feedback_adaptation = feedback_adaptation or build_feedback_adaptation()
        local_result = maybe_build_local_support_plan(
            analysis_text,
            entropy=entropy,
            conversation_history=conversation_history,
        )
        referral_decision = self._build_referral_decision(
            risk=risk,
            entropy=entropy,
            local_policy=getattr(local_result, "info", None),
        )
        entropy_orchestration = build_entropy_orchestration(
            text=analysis_text,
            conversation_history=conversation_history,
            risk=risk,
            entropy=entropy,
            state_profile=state_profile,
            intervention_strategy=intervention_strategy,
            dynamic_adjustment=dynamic_adjustment,
            feedback_adaptation=feedback_adaptation,
            referral_decision=referral_decision,
        )
        reduction_goal = build_entropy_reduction_goal(
            text=analysis_text,
            risk=risk,
            entropy=entropy,
            state_profile=state_profile,
            dynamic_adjustment=dynamic_adjustment,
            entropy_orchestration=entropy_orchestration,
            referral_decision=referral_decision,
            session_continuity=continuity_summary,
        )
        referral_explanation = build_referral_explanation(
            risk=risk,
            entropy=entropy,
            state_profile=state_profile,
            referral_decision=referral_decision,
            dynamic_adjustment=dynamic_adjustment,
            reduction_goal=reduction_goal,
            session_continuity=continuity_summary,
        )
        logger.info(
            "Entropy reduction strategy prepared target=%s drivers=%s intervention=%s",
            entropy_reduction.target_state,
            entropy_reduction.targeted_drivers,
            intervention_strategy.strategy_id,
        )
        if risk.level in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
            logger.warning("Routing request to crisis flow due to risk=%s", risk.level)
            crisis_response = self._build_crisis_response(
                text=clean_text,
                risk=risk,
                entropy=entropy,
                entropy_reduction=entropy_reduction,
                source=source,
                transcript=transcript,
                campus_resources=campus_resources,
                state_profile=state_profile,
                intervention_strategy=intervention_strategy,
                dynamic_adjustment=dynamic_adjustment,
                feedback_adaptation=feedback_adaptation,
                entropy_orchestration=entropy_orchestration,
                reduction_goal=reduction_goal,
                referral_explanation=referral_explanation,
                adjustment_loop=adjustment_loop,
            )
            crisis_response.reply_text = finalize_user_visible_reply(
                clean_text,
                crisis_response.reply_text,
                conversation_history=conversation_history,
                student_context=student_context,
            )
            return crisis_response

        if local_result is not None:
            assessment, plan = local_result
            plan = apply_intervention_strategy_to_plan(
                plan,
                strategy=intervention_strategy,
                state_profile=state_profile,
            )
            plan = apply_dynamic_adjustment_to_plan(
                plan,
                dynamic_adjustment=dynamic_adjustment,
            )
            plan = apply_feedback_adaptation_to_plan(
                plan,
                feedback_adaptation=feedback_adaptation,
            )
            plan = apply_session_continuity_to_plan(
                plan,
                continuity_summary=continuity_summary,
            )
            plan = apply_strategy_reselection_to_plan(
                plan,
                strategy_reselection=strategy_reselection,
            )
            plan = apply_adjustment_loop_to_plan(
                plan,
                adjustment_loop=adjustment_loop,
            )
            plan = apply_intervention_next_step_to_plan(
                plan,
                next_step=intervention_next_step,
            )
            plan = apply_session_tracking_to_plan(
                plan,
                tracking_snapshot=session_tracking,
            )
            plan = apply_strategy_version_to_plan(
                plan,
                strategy_version=strategy_version,
            )
            plan = apply_session_care_plan_to_plan(
                plan,
                session_care_plan=session_care_plan,
            )
            logger.info("Local dialogue policy handled text request.")
            safety = SafetyNotice(
                disclaimer="当前回复由本地规则层和支持策略共同生成，用于稳定边界和基础支持，不替代专业诊断。",
                emergency_notice=None,
                human_referral=(
                    f"如需进一步帮助，可联系 {self.settings.campus_counseling_center} "
                    f"（{self.settings.campus_counseling_hotline} / {self.settings.campus_counseling_email}）。"
                ),
            )
            return SupportResponse(
                response_id=new_response_id(),
                source=source,
                input_text=clean_text,
                transcript=transcript,
                reply_text=finalize_user_visible_reply(
                    clean_text,
                    sanitize_user_visible_reply(
                        clean_text,
                        self._render_reply_text(plan),
                        conversation_history=conversation_history,
                    ),
                    conversation_history=conversation_history,
                    student_context=student_context,
                ),
                risk=risk,
                entropy=entropy,
                entropy_reduction=entropy_reduction,
                assessment=assessment,
                plan=plan,
                campus_resources=campus_resources,
                safety=safety,
                metadata=new_metadata(f"llm:{self.llm_provider.name},stt:{self.stt_provider.name},policy:local"),
                local_policy=local_result.info,
                referral_decision=referral_decision,
                state_profile=state_profile,
                intervention_strategy=intervention_strategy,
                dynamic_adjustment=dynamic_adjustment,
                feedback_adaptation=feedback_adaptation,
                entropy_orchestration=entropy_orchestration,
                reduction_goal=reduction_goal,
                referral_explanation=referral_explanation,
                adjustment_loop=adjustment_loop,
            )

        system_prompt = build_system_prompt(self.settings)
        user_prompt = build_user_prompt(
            analysis_text,
            student_context or {},
            conversation_history or [],
            risk,
            entropy,
            entropy_reduction,
            campus_resources,
            state_profile=state_profile,
            intervention_strategy=intervention_strategy,
        )

        try:
            raw_output = self.llm_provider.complete(system_prompt=system_prompt, user_prompt=user_prompt)
            parsed = self._extract_json(raw_output)
            assessment = self._build_assessment(parsed, entropy)
            plan = self._build_plan(parsed)
        except Exception as exc:
            logger.exception("LLM pipeline failed, using fallback support plan: %s", exc)
            assessment, plan = self._build_fallback_plan(analysis_text, risk, entropy)

        plan = self._align_plan_with_entropy_strategy(plan, entropy_reduction)
        plan = self._enrich_plan_with_resources(plan, campus_resources)
        plan = apply_intervention_strategy_to_plan(
            plan,
            strategy=intervention_strategy,
            state_profile=state_profile,
        )
        plan = apply_dynamic_adjustment_to_plan(
            plan,
            dynamic_adjustment=dynamic_adjustment,
        )
        plan = apply_feedback_adaptation_to_plan(
            plan,
            feedback_adaptation=feedback_adaptation,
        )
        plan = apply_session_continuity_to_plan(
            plan,
            continuity_summary=continuity_summary,
        )
        plan = apply_strategy_reselection_to_plan(
            plan,
            strategy_reselection=strategy_reselection,
        )
        plan = apply_adjustment_loop_to_plan(
            plan,
            adjustment_loop=adjustment_loop,
        )
        plan = apply_intervention_next_step_to_plan(
            plan,
            next_step=intervention_next_step,
        )
        plan = apply_session_tracking_to_plan(
            plan,
            tracking_snapshot=session_tracking,
        )
        plan = apply_strategy_version_to_plan(
            plan,
            strategy_version=strategy_version,
        )
        plan = apply_session_care_plan_to_plan(
            plan,
            session_care_plan=session_care_plan,
        )
        logger.info(
            "Support response built with entropy_score=%s and %s campus resources",
            entropy.score,
            len(campus_resources),
        )

        safety = SafetyNotice(
            disclaimer="本系统提供校园心理支持建议，不能替代专业心理咨询、诊断或治疗。",
            emergency_notice=None,
            human_referral=(
                f"如果状态持续恶化，请联系 {self.settings.campus_counseling_center}"
                f"（{self.settings.campus_counseling_hotline} / {self.settings.campus_counseling_email}）。"
            ),
        )

        return SupportResponse(
            response_id=new_response_id(),
            source=source,
            input_text=clean_text,
            transcript=transcript,
            reply_text=finalize_user_visible_reply(
                analysis_text,
                sanitize_user_visible_reply(
                    analysis_text,
                    self._render_reply_text(plan),
                    conversation_history=conversation_history,
                ),
                conversation_history=conversation_history,
                student_context=student_context,
            ),
            risk=risk,
            entropy=entropy,
            entropy_reduction=entropy_reduction,
            assessment=assessment,
            plan=plan,
            campus_resources=campus_resources,
            safety=safety,
            metadata=new_metadata(f"llm:{self.llm_provider.name},stt:{self.stt_provider.name}"),
            local_policy=None,
            referral_decision=referral_decision,
            state_profile=state_profile,
            intervention_strategy=intervention_strategy,
            dynamic_adjustment=dynamic_adjustment,
            feedback_adaptation=feedback_adaptation,
            entropy_orchestration=entropy_orchestration,
            reduction_goal=reduction_goal,
            referral_explanation=referral_explanation,
            adjustment_loop=adjustment_loop,
        )

    def handle_audio(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        content_type: str | None,
        student_context: dict[str, Any] | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
        previous_entropy_score: int | None = None,
        entropy_trace: list[dict[str, Any]] | None = None,
        feedback_adaptation: FeedbackAdaptation | None = None,
    ) -> SupportResponse:
        multimodal_signal = analyze_audio_signal(
            file_bytes=file_bytes,
            filename=filename,
            content_type=content_type,
        )
        enriched_context = dict(student_context or {})
        enriched_context["multimodal_signal"] = asdict(multimodal_signal)
        if multimodal_signal.analysis_available:
            logger.info(
                "Audio signal analyzed file=%s duration=%s rms=%s silence=%s",
                filename,
                multimodal_signal.duration_seconds,
                multimodal_signal.rms_energy,
                multimodal_signal.silence_ratio,
            )
        else:
            logger.info(
                "Audio signal metadata captured file=%s size=%s notes=%s",
                filename,
                multimodal_signal.byte_size,
                multimodal_signal.analysis_notes,
            )
        transcript = self.stt_provider.transcribe(
            file_bytes=file_bytes,
            filename=filename,
            content_type=content_type,
        )
        logger.info("Audio request transcribed successfully for file=%s", filename)
        response = self.handle_text(
            text=transcript,
            student_context=enriched_context,
            conversation_history=conversation_history,
            previous_entropy_score=previous_entropy_score,
            entropy_trace=entropy_trace,
            feedback_adaptation=feedback_adaptation,
            source="audio",
            transcript=transcript,
        )
        response.multimodal_signal = multimodal_signal
        return response

    def _extract_json(self, raw_output: str) -> dict[str, Any]:
        candidate = raw_output.strip()
        if candidate.startswith("```"):
            # 兼容模型偶尔返回 markdown code fence 的情况。
            candidate = candidate.strip("`")
            if "\n" in candidate:
                candidate = candidate.split("\n", 1)[1]
            candidate = candidate.rsplit("```", 1)[0].strip()

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start >= 0 and end > start:
                return json.loads(candidate[start : end + 1])
            raise

    def _build_assessment(self, parsed: dict[str, Any], entropy: PsychologicalEntropy) -> SupportAssessment:
        return SupportAssessment(
            primary_emotions=self._ensure_list(parsed.get("primary_emotions"), ["焦虑"]),
            stressors=self._ensure_list(parsed.get("stressors"), ["综合压力"]),
            protective_factors=self._ensure_list(parsed.get("protective_factors"), ["愿意主动表达"]),
            entropy_level=entropy.level,
            balance_state=entropy.balance_state,
        )

    def _build_plan(self, parsed: dict[str, Any]) -> SupportPlan:
        return SupportPlan(
            summary=str(parsed.get("summary", "当前需要先稳定节律，再逐步处理压力源。")),
            immediate_support=self._ensure_list(
                parsed.get("immediate_support"),
                ["先停下来做 3 轮缓慢呼吸，再整理眼前最紧急的一件事。"],
            ),
            campus_actions=self._ensure_list(
                parsed.get("campus_actions"),
                [f"如果状态持续，请联系 {self.settings.campus_counseling_center}。"],
            ),
            self_regulation=self._ensure_list(
                parsed.get("self_regulation"),
                ["把任务拆成 15 分钟一个小段，只开始第一段。"],
            ),
            follow_up=self._ensure_list(
                parsed.get("follow_up"),
                ["24-48 小时后回顾情绪、睡眠和压力是否出现改善。"],
            ),
        )

    def _build_fallback_plan(
        self,
        text: str,
        risk: RiskAssessment,
        entropy: PsychologicalEntropy,
    ) -> tuple[SupportAssessment, SupportPlan]:
        emotions = ["焦虑", "疲惫"] if any(term in text for term in ["考试", "睡", "论文"]) else ["低落", "紧张"]
        assessment = SupportAssessment(
            primary_emotions=emotions,
            stressors=["近期综合压力", "作息或情绪负荷"],
            protective_factors=["愿意求助", "能够表达困扰"],
            entropy_level=entropy.level,
            balance_state=entropy.balance_state,
        )
        plan = SupportPlan(
            summary="你目前像是在高负荷状态下持续运转，先把节律稳住，比一次解决所有问题更重要。",
            immediate_support=[
                "先决定今天只完成一件最小任务，避免继续把自己推到过载。",
                "暂停 3 分钟，做缓慢呼吸并放松肩颈。",
            ],
            campus_actions=[
                f"如果困扰持续超过 1-2 周，预约 {self.settings.campus_counseling_center}。",
                "把本周最担心的问题同步给可信任的同学、辅导员或家人。",
            ],
            self_regulation=[
                "今晚优先争取更规律的入睡时间。",
                "把明天要做的事控制在 3 项以内。",
            ],
            follow_up=[
                "明天记录睡眠时长、精力和情绪变化。",
                "48 小时后评估压力是否下降。",
            ],
        )
        return assessment, plan

    def _build_crisis_response(
        self,
        *,
        text: str,
        risk: RiskAssessment,
        entropy: PsychologicalEntropy,
        entropy_reduction: EntropyReductionStrategy,
        source: str,
        transcript: str | None,
        campus_resources: list[CampusResource],
        state_profile: StateProfile,
        intervention_strategy: Any,
        dynamic_adjustment: Any,
        feedback_adaptation: Any,
        entropy_orchestration: EntropyOrchestration,
        reduction_goal: EntropyReductionGoal,
        referral_explanation: ReferralExplanation,
        adjustment_loop: Any = None,
    ) -> SupportResponse:
        emergency_notice = (
            "检测到高风险内容。请不要让当事人独处，并立即联系当地紧急服务、校园值班人员"
            f"或 {self.settings.campus_counseling_center}。"
        )
        contact_line = "；".join(self.settings.crisis_contacts)
        return SupportResponse(
            response_id=new_response_id(),
            source=source,
            input_text=text,
            transcript=transcript,
            reply_text=(
                "当前最重要的不是继续分析问题，而是先保证你的安全。请你现在不要一个人待着，"
                "马上联系身边可信任的人，比如室友、同学、辅导员或家人，并尽快寻求紧急帮助。"
            ),
            risk=risk,
            entropy=entropy,
            entropy_reduction=entropy_reduction,
            assessment=SupportAssessment(
                primary_emotions=["极度痛苦", "失控感"],
                stressors=["强烈危机信号"],
                protective_factors=["仍然有机会通过立刻求助获得支持"],
                entropy_level=entropy.level,
                balance_state=entropy.balance_state,
            ),
            plan=SupportPlan(
                summary="当前最重要的目标不是继续分析问题，而是立刻转入现实世界的安全支持。",
                immediate_support=[
                    "立刻联系身边可信任的人，确保当事人不是一个人。",
                    "移开危险物品，尽量待在有人陪伴、可被及时帮助的地方。",
                    "马上拨打当地紧急电话或校园危机干预电话。",
                ],
                campus_actions=[
                    f"尽快联系 {self.settings.campus_counseling_center}：{self.settings.campus_counseling_hotline}",
                    "联系辅导员、班主任、宿舍管理员或家属进行现场支持。",
                    f"可优先使用这些联络路径：{contact_line}",
                ],
                self_regulation=[
                    "现在先不要单独承受，也不要要求自己立刻想清楚所有问题。",
                    "只做一件事：把求助信息发出去，并让他人来到你身边。",
                ],
                follow_up=[
                    "危机解除后，安排专业心理老师或医生继续评估。",
                    "后续建立睡眠、陪伴和学业减压的短期支持计划。",
                ],
            ),
            campus_resources=campus_resources,
            safety=SafetyNotice(
                disclaimer="本系统不能处理危机干预，当前结果仅用于触发紧急转介。",
                emergency_notice=emergency_notice,
                human_referral=(
                    f"请立即联系 {self.settings.campus_counseling_center}"
                    f"（{self.settings.campus_counseling_hotline} / {self.settings.campus_counseling_email}）。"
                ),
            ),
            metadata=new_metadata(f"llm:{self.llm_provider.name},stt:{self.stt_provider.name}"),
            local_policy=None,
            referral_decision=self._build_referral_decision(risk=risk, entropy=entropy, local_policy=None),
            state_profile=state_profile,
            intervention_strategy=intervention_strategy,
            dynamic_adjustment=dynamic_adjustment,
            feedback_adaptation=feedback_adaptation,
            entropy_orchestration=entropy_orchestration,
            reduction_goal=reduction_goal,
            referral_explanation=referral_explanation,
            adjustment_loop=adjustment_loop,
        )

    @staticmethod
    def _ensure_list(value: Any, fallback: list[str]) -> list[str]:
        if isinstance(value, list):
            normalized = [str(item).strip() for item in value if str(item).strip()]
            if normalized:
                return normalized
        return fallback

    def _retrieve_campus_resources(self, text: str, risk: RiskAssessment) -> list[CampusResource]:
        if not self.retriever:
            return []
        return self.retriever.retrieve(text, risk)

    @staticmethod
    def _enrich_plan_with_resources(plan: SupportPlan, campus_resources: list[CampusResource]) -> SupportPlan:
        extra_actions = []
        for resource in campus_resources:
            if resource.recommended_actions:
                extra_actions.append(resource.recommended_actions[0])

        # 去重后把校园资源动作并入主计划，保证输出既像心理支持也像校园 agent。
        merged_actions: list[str] = []
        for action in [*plan.campus_actions, *extra_actions]:
            clean = action.strip()
            if clean and clean not in merged_actions:
                merged_actions.append(clean)

        plan.campus_actions = merged_actions[:5]
        return plan

    @staticmethod
    def _render_reply_text(plan: SupportPlan) -> str:
        parts: list[str] = []
        if plan.summary.strip():
            parts.append(plan.summary.strip())
        if plan.immediate_support:
            first_support = plan.immediate_support[0].strip()
            if first_support and first_support not in parts:
                parts.append(first_support)
        if plan.follow_up:
            first_follow_up = plan.follow_up[0].strip()
            if first_follow_up and first_follow_up not in parts:
                parts.append(first_follow_up)
        return " ".join(parts)

    def _build_referral_decision(
        self,
        *,
        risk: RiskAssessment,
        entropy: PsychologicalEntropy,
        local_policy: Any | None,
    ) -> ReferralDecision:
        reasons: list[str] = []
        urgency = "none"
        should_refer = False
        recommended_channel: str | None = None

        if risk.level in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
            should_refer = True
            urgency = "urgent"
            reasons.append(f"risk_level:{risk.level}")

        if risk.needs_human_followup:
            should_refer = True
            if urgency == "none":
                urgency = "recommended"
            reasons.append("human_followup_requested")

        if entropy.level >= 4 or entropy.score >= 70:
            should_refer = True
            if urgency == "none":
                urgency = "recommended"
            reasons.append("elevated_entropy")

        if local_policy is not None:
            policy_name = getattr(local_policy, "policy_name", "")
            policy_stage = getattr(local_policy, "policy_stage", "")
            escalation_hint = getattr(local_policy, "escalation_hint", None)

            if policy_stage == "escalation_watch":
                should_refer = True
                if urgency == "none":
                    urgency = "watch"
                reasons.append(f"policy_stage:{policy_stage}")

            if policy_name in {
                "sleep_appetite_drift",
                "helplessness_escalation",
                "rising_emotional_spiral",
            }:
                should_refer = True
                if urgency in {"none", "watch"}:
                    urgency = "recommended"
                reasons.append(f"policy_name:{policy_name}")

            if escalation_hint:
                reasons.append(f"hint:{escalation_hint}")

        if should_refer:
            recommended_channel = self.settings.campus_counseling_center

        deduped_reasons: list[str] = []
        for reason in reasons:
            if reason not in deduped_reasons:
                deduped_reasons.append(reason)

        return ReferralDecision(
            should_refer=should_refer,
            urgency=urgency,
            reasons=deduped_reasons,
            recommended_channel=recommended_channel,
        )

    @staticmethod
    def _align_plan_with_entropy_strategy(
        plan: SupportPlan,
        entropy_reduction: EntropyReductionStrategy,
    ) -> SupportPlan:
        # 减熵策略保留在独立字段里给系统层和前端分析区展示，
        # 不再直接混入用户可见的主回复，避免把“认知熵/72小时复盘”这类系统腔推给用户。
        return plan
