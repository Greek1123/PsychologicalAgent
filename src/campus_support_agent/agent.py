from __future__ import annotations

import json
from typing import Any

from .config import Settings
from .entropy import evaluate_psychological_entropy
from .logging_utils import get_logger
from .local_response_policy import maybe_build_local_support_plan
from .prompts import build_system_prompt, build_user_prompt
from .providers import LLMProvider, STTProvider
from .reduction import build_entropy_reduction_strategy
from .response_guardrails import sanitize_user_visible_reply
from .retrieval import CampusKnowledgeRetriever
from .safety import evaluate_text_risk
from .schemas import (
    CampusResource,
    EntropyReductionStrategy,
    PsychologicalEntropy,
    ReferralDecision,
    RiskAssessment,
    RiskLevel,
    SafetyNotice,
    SupportAssessment,
    SupportPlan,
    SupportResponse,
    new_metadata,
    new_response_id,
)


logger = get_logger("agent")


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
        source: str = "text",
        transcript: str | None = None,
    ) -> SupportResponse:
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("text 不能为空。")

        # 先做安全风控，再做心理熵评估，保证危机信号优先被处理。
        risk = evaluate_text_risk(clean_text)
        entropy = evaluate_psychological_entropy(
            clean_text,
            risk,
            student_context=student_context,
            conversation_history=conversation_history,
        )
        logger.info(
            "Text request analyzed with risk=%s score=%s entropy=%s source=%s",
            risk.level,
            risk.score,
            entropy.score,
            source,
        )

        campus_resources = self._retrieve_campus_resources(clean_text, risk)
        entropy_reduction = build_entropy_reduction_strategy(entropy, risk, campus_resources)
        local_result = maybe_build_local_support_plan(
            clean_text,
            entropy=entropy,
            conversation_history=conversation_history,
        )
        referral_decision = self._build_referral_decision(
            risk=risk,
            entropy=entropy,
            local_policy=getattr(local_result, "info", None),
        )
        logger.info(
            "Entropy reduction strategy prepared target=%s drivers=%s",
            entropy_reduction.target_state,
            entropy_reduction.targeted_drivers,
        )
        if risk.level in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
            logger.warning("Routing request to crisis flow due to risk=%s", risk.level)
            return self._build_crisis_response(
                text=clean_text,
                risk=risk,
                entropy=entropy,
                entropy_reduction=entropy_reduction,
                source=source,
                transcript=transcript,
                campus_resources=campus_resources,
            )

        if local_result is not None:
            assessment, plan = local_result
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
                reply_text=sanitize_user_visible_reply(
                    clean_text,
                    self._render_reply_text(plan),
                    conversation_history=conversation_history,
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
            )

        system_prompt = build_system_prompt(self.settings)
        user_prompt = build_user_prompt(
            clean_text,
            student_context or {},
            conversation_history or [],
            risk,
            entropy,
            entropy_reduction,
            campus_resources,
        )

        try:
            raw_output = self.llm_provider.complete(system_prompt=system_prompt, user_prompt=user_prompt)
            parsed = self._extract_json(raw_output)
            assessment = self._build_assessment(parsed, entropy)
            plan = self._build_plan(parsed)
        except Exception as exc:
            logger.exception("LLM pipeline failed, using fallback support plan: %s", exc)
            assessment, plan = self._build_fallback_plan(clean_text, risk, entropy)

        plan = self._align_plan_with_entropy_strategy(plan, entropy_reduction)
        plan = self._enrich_plan_with_resources(plan, campus_resources)
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
            reply_text=sanitize_user_visible_reply(
                clean_text,
                self._render_reply_text(plan),
                conversation_history=conversation_history,
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
        )

    def handle_audio(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        content_type: str | None,
        student_context: dict[str, Any] | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> SupportResponse:
        transcript = self.stt_provider.transcribe(
            file_bytes=file_bytes,
            filename=filename,
            content_type=content_type,
        )
        logger.info("Audio request transcribed successfully for file=%s", filename)
        return self.handle_text(
            text=transcript,
            student_context=student_context,
            conversation_history=conversation_history,
            source="audio",
            transcript=transcript,
        )

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
            reply_text="当前最重要的不是继续分析问题，而是立刻转入现实世界的安全支持。请马上联系身边可信任的人，并尽快寻求紧急帮助。",
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
