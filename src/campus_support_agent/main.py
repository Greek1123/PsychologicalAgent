from __future__ import annotations

from dataclasses import asdict
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .agent import CampusSupportAgent
from .adjustment_loop import build_entropy_adjustment_loop, enrich_student_context_with_adjustment_loop
from .care_plan import enrich_student_context_with_care_plan
from .config import Settings
from .deployment_readiness import build_deployment_readiness
from .dialogue_memory import enrich_student_context_with_memory
from .dynamic_adjustment import build_dynamic_adjustment
from .entropy import evaluate_psychological_entropy
from .feedback_adaptation import build_feedback_adaptation
from .intervention_next_step import enrich_student_context_with_intervention_next_step
from .logging_utils import configure_logging, get_logger
from .privacy_views import (
    SUPPORTED_VIEW_ROLES,
    normalize_view_role,
    project_session_analysis_for_role,
    project_support_response_for_role,
)
from .providers import build_llm_provider, build_stt_provider
from .reduction import build_entropy_reduction_strategy
from .retrieval import CampusKnowledgeRetriever
from .safety import evaluate_text_risk
from .schemas import EntropyTrend
from .session_continuity import build_session_continuity_summary, enrich_student_context_with_continuity
from .session_tracking import enrich_student_context_with_session_tracking
from .goal_attainment import build_goal_attainment_timeline
from .strategy_reselection import (
    build_strategy_reselection_decision,
    enrich_student_context_with_strategy_reselection,
)
from .strategy_versioning import enrich_student_context_with_strategy_version
from .storage import SQLiteSessionStore


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    configure_logging(settings)
    return settings


logger = get_logger("main")
STATIC_DIR = Path(__file__).resolve().parent / "static"
APP_HTML = STATIC_DIR / "app.html"
HUMAN_INTERVENTION_STATUSES = {"acknowledged", "in_progress", "escalated", "resolved", "closed"}


@lru_cache(maxsize=1)
def get_agent() -> CampusSupportAgent:
    settings = get_settings()
    return CampusSupportAgent(
        settings=settings,
        llm_provider=build_llm_provider(settings),
        stt_provider=build_stt_provider(settings),
        retriever=CampusKnowledgeRetriever(settings),
    )


@lru_cache(maxsize=1)
def get_session_store() -> SQLiteSessionStore:
    settings = get_settings()
    # 会话与熵轨迹默认落到 SQLite，保证重启后仍然可以继续做动态平衡分析。
    return SQLiteSessionStore(
        db_path=settings.database_path,
        max_messages=settings.max_history_turns * 2,
    )


app = FastAPI(
    title="Campus Psychological Support Agent",
    version="0.1.0",
    description="多模态校园心理支持 Agent MVP",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_startup_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_startup_settings.frontend_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _parse_optional_json(value: str | None, field_name: str) -> dict[str, Any] | list[dict[str, Any]]:
    if not value:
        return {} if field_name == "student_context" else []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"{field_name} 不是合法 JSON。") from exc

    if field_name == "student_context" and not isinstance(parsed, dict):
        raise HTTPException(status_code=422, detail="student_context 必须是 JSON 对象。")
    if field_name == "conversation_history" and not isinstance(parsed, list):
        raise HTTPException(status_code=422, detail="conversation_history 必须是 JSON 数组。")
    return parsed


def _merge_conversation_history(session_id: str | None, conversation_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not session_id:
        return list(conversation_history)
    session_store = get_session_store()
    session_history = session_store.get_history(session_id)
    return [*session_history, *conversation_history]


def _update_session_state(
    *,
    session_id: str | None,
    user_text: str,
    assistant_text: str,
    result: Any,
    response: dict[str, Any],
) -> None:
    if not session_id:
        return

    session_store = get_session_store()
    previous_entropy = session_store.get_last_entropy(session_id)
    history_size = session_store.append_exchange(
        session_id,
        user_text=user_text,
        assistant_text=assistant_text,
    )
    entropy_trace_size = session_store.append_entropy_snapshot(
        session_id,
        response_id=result.response_id,
        score=result.entropy.score,
        level=result.entropy.level,
        balance_state=result.entropy.balance_state,
        dominant_drivers=result.entropy.dominant_drivers,
    )

    if previous_entropy:
        delta = result.entropy.score - int(previous_entropy["score"])
        direction = "up" if delta > 0 else "down" if delta < 0 else "flat"
        trend_override = EntropyTrend(
            previous_score=int(previous_entropy["score"]),
            delta=delta,
            direction=direction,
        )
        response["entropy"]["trend"] = {
            "previous_score": trend_override.previous_score,
            "delta": trend_override.delta,
            "direction": trend_override.direction,
        }
        logger.info(
            "Entropy trend updated session_id=%s previous=%s current=%s delta=%s direction=%s",
            session_id,
            previous_entropy["score"],
            result.entropy.score,
            delta,
            direction,
        )
    else:
        trend_override = result.entropy.trend

    entropy_trace = session_store.get_entropy_trace(session_id)
    dynamic_adjustment = build_dynamic_adjustment(
        entropy=result.entropy,
        risk=result.risk,
        state_profile=result.state_profile,
        trend_override=trend_override,
        entropy_trace=entropy_trace,
    )
    response["dynamic_adjustment"] = asdict(dynamic_adjustment)
    _apply_dynamic_adjustment_to_entropy_reduction(response, dynamic_adjustment.review_window_hours)

    response["session"] = {
        "session_id": session_id,
        "history_messages": history_size,
        "entropy_trace_points": entropy_trace_size,
    }


def _apply_dynamic_adjustment_to_entropy_reduction(response: dict[str, Any], review_window_hours: int) -> None:
    reduction = response.get("entropy_reduction")
    if not isinstance(reduction, dict):
        return
    current_window = reduction.get("review_window_hours")
    if isinstance(current_window, int):
        reduction["review_window_hours"] = min(current_window, review_window_hours)
    else:
        reduction["review_window_hours"] = review_window_hours


def _apply_session_escalation(
    *,
    session_id: str | None,
    response: dict[str, Any],
) -> None:
    response.setdefault("system_flags", {"manual_referral_recommended": False, "reasons": []})
    if not session_id:
        return

    session_store = get_session_store()
    recent_records = session_store.list_support_responses(session_id=session_id, limit=3)
    current_referral = response.get("referral_decision") or {}
    current_entropy = response.get("entropy") or {}
    current_local_policy = response.get("local_policy") or {}
    current_dynamic = response.get("dynamic_adjustment") or {}
    trend = current_entropy.get("trend") or {}

    reasons: list[str] = []
    prior_referred = sum(1 for record in recent_records if record.get("referral_should_refer"))
    current_should_refer = bool(current_referral.get("should_refer"))
    current_urgency = current_referral.get("urgency", "none")

    if trend.get("direction") == "up" and (trend.get("delta") or 0) >= 8:
        reasons.append("entropy_rising")

    if current_local_policy.get("policy_stage") == "escalation_watch":
        reasons.append("policy_escalation_watch")

    if current_should_refer and prior_referred >= 2:
        reasons.append("repeated_referral_pattern")

    if current_dynamic.get("should_refer"):
        current_referral["should_refer"] = True
        if not current_referral.get("recommended_channel"):
            current_referral["recommended_channel"] = get_settings().campus_counseling_center
        action = current_dynamic.get("action")
        if action == "urgent_referral":
            current_referral["urgency"] = "urgent"
        elif current_referral.get("urgency") in {None, "", "none", "watch"}:
            current_referral["urgency"] = "recommended"
        reasons.append(f"dynamic_adjustment:{current_dynamic.get('stability_state')}")

    updated_urgency = current_referral.get("urgency", current_urgency)
    manual_referral_recommended = bool(reasons) or updated_urgency == "urgent"
    if manual_referral_recommended and updated_urgency == "watch":
        current_referral["urgency"] = "recommended"

    response["referral_decision"] = current_referral
    response["system_flags"] = {
        "manual_referral_recommended": manual_referral_recommended,
        "reasons": reasons,
        "recent_referred_count": prior_referred,
    }


def _store_referral_event_if_needed(*, session_id: str | None, response: dict[str, Any]) -> None:
    if not session_id:
        return

    referral = response.get("referral_decision") or {}
    flags = response.get("system_flags") or {}
    if not referral.get("should_refer") and not flags.get("manual_referral_recommended"):
        return

    local_policy = response.get("local_policy") or {}
    risk = response.get("risk") or {}
    entropy = response.get("entropy") or {}
    get_session_store().append_referral_event(
        session_id=session_id,
        response_id=response["response_id"],
        urgency=referral.get("urgency") or "watch",
        reasons=[*referral.get("reasons", []), *flags.get("reasons", [])],
        policy_name=local_policy.get("policy_name"),
        risk_level=risk.get("level"),
        entropy_score=entropy.get("score"),
        manual_referral_recommended=bool(flags.get("manual_referral_recommended")),
    )


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/app")


@app.get("/app", include_in_schema=False)
def app_ui() -> FileResponse:
    # 内置一个轻量前端，方便研究阶段直接观察熵值与减熵策略。
    logger.info("Web UI requested.")
    return FileResponse(APP_HTML)


@app.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    logger.info("Health check requested.")
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "stt_provider": settings.stt_provider,
    }


@app.get("/api/v1/model/status")
def get_model_status() -> dict[str, Any]:
    settings = get_settings()
    checkpoint_path = Path(settings.local_checkpoint_path) if settings.local_checkpoint_path else None
    base_model_path = Path(settings.local_base_model_path) if settings.local_base_model_path else None
    cache_root = Path(settings.local_model_cache_root) if settings.local_model_cache_root else None
    is_local = settings.llm_provider.strip().lower() == "local_checkpoint"

    status = {
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "stt_provider": settings.stt_provider,
        "local_checkpoint": {
            "enabled": is_local,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else "",
            "checkpoint_exists": bool(checkpoint_path and checkpoint_path.exists()),
            "base_model_path": str(base_model_path) if base_model_path else "",
            "base_model_exists": bool(base_model_path and base_model_path.exists()),
            "cache_root": str(cache_root) if cache_root else "",
            "cache_root_exists": bool(cache_root and cache_root.exists()),
            "temperature": settings.local_generation_temperature,
            "max_tokens": settings.llm_max_tokens,
        },
    }
    logger.info("Model status requested provider=%s local_enabled=%s", settings.llm_provider, is_local)
    return status


@app.get("/api/v1/ops/readiness")
def get_ops_readiness() -> dict[str, Any]:
    readiness = build_deployment_readiness(get_settings())
    logger.info("Deployment readiness requested status=%s summary=%s", readiness["status"], readiness["summary"])
    return readiness


@app.get("/api/v1/frontend/contract")
def get_frontend_contract() -> dict[str, Any]:
    logger.info("Frontend contract requested.")
    return {
        "version": "2026-05-27",
        "purpose": "Stable handoff contract for the student-facing chat UI and research dashboard.",
        "cors": {
            "allowed_origins": get_settings().frontend_allowed_origins,
            "env": "FRONTEND_ALLOWED_ORIGINS",
            "example": "http://127.0.0.1:5173,http://localhost:5173",
        },
        "endpoints": {
            "text_support": {
                "method": "POST",
                "path": "/api/v1/support/text",
                "content_type": "application/json",
                "required_fields": ["text"],
                "optional_fields": ["session_id", "student_context", "conversation_history"],
            },
            "audio_support": {
                "method": "POST",
                "path": "/api/v1/support/audio",
                "content_type": "multipart/form-data",
                "required_fields": ["file"],
                "optional_fields": ["session_id", "student_context", "conversation_history"],
            },
            "session_history": {
                "method": "GET",
                "path": "/api/v1/sessions/{session_id}",
            },
            "session_analysis": {
                "method": "GET",
                "path": "/api/v1/sessions/{session_id}/analysis",
                "processing_fields": [
                    "latest_processing_summary",
                    "processing_timeline",
                    "processing_summary",
                    "processing_consistency",
                ],
            },
            "care_queue": {
                "method": "GET",
                "path": "/api/v1/analytics/care-queue",
                "query": ["limit", "include_low_priority", "include_resolved"],
            },
            "human_interventions": {
                "method": "POST",
                "path": "/api/v1/sessions/{session_id}/human-interventions",
                "statuses": sorted(HUMAN_INTERVENTION_STATUSES),
            },
            "role_view": {
                "method": "GET",
                "path": "/api/v1/sessions/{session_id}/view",
                "query": ["role=student|counselor|research|admin"],
            },
            "model_status": {
                "method": "GET",
                "path": "/api/v1/model/status",
            },
            "ops_readiness": {
                "method": "GET",
                "path": "/api/v1/ops/readiness",
            },
        },
        "text_request_example": {
            "session_id": "demo-student-001",
            "text": "最近我一到晚上就很焦虑，睡不着，也不知道该怎么和室友说。",
            "student_context": {
                "grade": "undergraduate",
                "campus": "main",
            },
            "conversation_history": [],
        },
        "audio_request_form_example": {
            "file": "audio.wav",
            "session_id": "demo-student-001",
            "student_context": "{\"grade\":\"undergraduate\",\"campus\":\"main\"}",
            "conversation_history": "[]",
        },
        "response_core_fields": {
            "reply_text": "Main assistant reply. Show this as the primary chat bubble.",
            "risk": "Risk level, score, reason, trigger terms, and human follow-up hint.",
            "entropy": "Current entropy score, level, balance state, dimensions, drivers, and trend.",
            "entropy_reduction": "Target state, core actions, expected delta, and review window.",
            "state_profile": "Detected psychological state and evidence signals.",
            "intervention_strategy": "Selected support strategy and next-step mode.",
            "dynamic_adjustment": "Whether strategy or care intensity should change across turns.",
            "referral_decision": "Whether human referral is recommended and at what urgency.",
            "safety": "Disclaimer, emergency notice, and human referral text.",
            "multimodal_signal": "Audio evidence summary when the input is audio.",
            "session": "Session id plus stored history and entropy trace counts.",
            "system_flags": "Backend flags for manual review and repeated referral patterns.",
            "human_interventions": "Manual handling records in session analysis after staff acknowledgement or resolution.",
            "processing_summary": "Backend processing-layer route, completed stages, safety priority, and next backend action. Keep this in debug/research/admin views.",
        },
        "human_intervention_request_example": {
            "response_id": "support_xxx",
            "status": "acknowledged",
            "handler_id": "counselor-001",
            "note": "已查看高优先级队列，准备线下跟进。",
            "next_action": "contact_student_with_low_pressure_checkin",
            "tags": ["manual_followup", "same_day_review"],
        },
        "frontend_display_policy": {
            "student_chat": ["reply_text", "safety.emergency_notice", "safety.human_referral"],
            "student_optional_panel": [
                "entropy.balance_state",
                "entropy_reduction.core_actions",
                "campus_resources",
            ],
            "research_dashboard": [
                "risk",
                "entropy",
                "state_profile",
                "intervention_strategy",
                "dynamic_adjustment",
                "referral_decision",
                "processing_summary",
                "multimodal_signal",
                "system_flags",
            ],
            "hide_from_student_by_default": [
                "hidden_clinical_goal",
                "backend_reason",
                "backend_actions",
                "system_flags.reasons",
            ],
            "backend_role_views": {
                "student": "Shows reply, safety notice, lightweight risk label, balance state, and user-facing care actions.",
                "counselor": "Shows operational care fields while removing hidden clinical goals and backend-only action internals.",
                "research": "Keeps structured metrics but redacts free text and direct intervention notes.",
                "admin": "Full internal payload for local development and audit.",
            },
        },
        "risk_badges": {
            "low": {"label": "Low", "tone": "neutral"},
            "medium": {"label": "Medium", "tone": "watch"},
            "high": {"label": "High", "tone": "alert"},
            "critical": {"label": "Critical", "tone": "urgent"},
        },
        "demo_prompts": [
            "我最近期末复习很崩溃，晚上睡不着。",
            "室友总是在我休息的时候开外放，我又不敢说。",
            "我喜欢一个同学很久了，但怕表白后连朋友都做不成。",
            "有人一直跟着我到宿舍附近，但我又怕是自己想多了。",
        ],
    }


@app.post("/api/v1/support/text")
def support_text(payload: dict[str, Any]) -> dict[str, Any]:
    text = str(payload.get("text", "")).strip()
    if not text:
        raise HTTPException(status_code=422, detail="text 不能为空。")

    student_context = payload.get("student_context") or {}
    conversation_history = payload.get("conversation_history") or []
    session_id = str(payload.get("session_id", "")).strip() or None
    if not isinstance(student_context, dict):
        raise HTTPException(status_code=422, detail="student_context 必须是对象。")
    if not isinstance(conversation_history, list):
        raise HTTPException(status_code=422, detail="conversation_history 必须是数组。")

    logger.info("Received text support request session_id=%s text_length=%s", session_id or "-", len(text))
    merged_history = _merge_conversation_history(session_id, conversation_history)
    student_context = enrich_student_context_with_memory(
        student_context,
        merged_history,
        current_text=text,
    )
    previous_entropy_score = None
    entropy_trace: list[dict[str, Any]] = []
    if session_id:
        session_store = get_session_store()
        previous_entropy = session_store.get_last_entropy(session_id)
        previous_entropy_score = int(previous_entropy["score"]) if previous_entropy else None
        entropy_trace = session_store.get_entropy_trace(session_id)
        previous_records = session_store.list_support_responses(session_id=session_id)
        continuity_summary = build_session_continuity_summary(
            session_id=session_id,
            records=previous_records,
            conversation_history=merged_history,
            entropy_trace=entropy_trace,
        )
        student_context = enrich_student_context_with_continuity(student_context, continuity_summary)
        goal_attainment_timeline = build_goal_attainment_timeline(previous_records)
        strategy_reselection = build_strategy_reselection_decision(
            goal_attainment_timeline=goal_attainment_timeline,
            session_continuity=continuity_summary,
            latest_record=previous_records[-1] if previous_records else None,
        )
        student_context = enrich_student_context_with_strategy_reselection(student_context, strategy_reselection)
        feedback_adaptation = build_feedback_adaptation(
            feedback_summary=session_store.summarize_intervention_feedback(session_id),
            recent_feedback=session_store.get_intervention_feedback(session_id, limit=5),
        )
        adjustment_loop = build_entropy_adjustment_loop(
            session_id=session_id,
            records=previous_records,
            feedback_summary=session_store.summarize_intervention_feedback(session_id),
            recent_feedback=session_store.get_intervention_feedback(session_id, limit=5),
            session_continuity=continuity_summary,
            strategy_reselection=strategy_reselection,
            audit_summary=session_store.get_intervention_audits(session_id, limit=20)["audit_summary"],
        )
        student_context = enrich_student_context_with_adjustment_loop(student_context, adjustment_loop)
        student_context = enrich_student_context_with_care_plan(
            student_context,
            session_store.get_session_care_plan(session_id)["session_care_plan"],
        )
        student_context = enrich_student_context_with_intervention_next_step(
            student_context,
            session_store.get_session_intervention_effectiveness(session_id)["intervention_next_step"],
        )
        student_context = enrich_student_context_with_session_tracking(
            student_context,
            session_store.get_session_tracking(session_id)["session_tracking"],
        )
        student_context = enrich_student_context_with_strategy_version(
            student_context,
            session_store.get_session_strategy_version(session_id)["strategy_version"],
        )
    else:
        feedback_adaptation = build_feedback_adaptation()
        adjustment_loop = build_entropy_adjustment_loop(session_id="anonymous")
        student_context = enrich_student_context_with_adjustment_loop(student_context, adjustment_loop)

    result = get_agent().handle_text(
        text=text,
        student_context=student_context,
        conversation_history=merged_history,
        previous_entropy_score=previous_entropy_score,
        entropy_trace=entropy_trace,
        feedback_adaptation=feedback_adaptation,
    )
    response = result.to_dict()

    _update_session_state(
        session_id=session_id,
        user_text=text,
        assistant_text=result.reply_text,
        result=result,
        response=response,
    )
    _apply_session_escalation(session_id=session_id, response=response)
    _store_referral_event_if_needed(session_id=session_id, response=response)

    session_store = get_session_store()
    session_store.store_support_response(
        session_id=session_id,
        response_id=result.response_id,
        source=result.source,
        input_text=text,
        transcript=result.transcript,
        student_context=student_context,
        conversation_history=merged_history,
        response_payload=response,
    )
    logger.info("Completed text support request response_id=%s", result.response_id)
    return response


@app.post("/api/v1/support/audio")
async def support_audio(
    file: UploadFile = File(...),
    student_context: str | None = Form(default=None),
    conversation_history: str | None = Form(default=None),
    session_id: str | None = Form(default=None),
) -> dict[str, Any]:
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=422, detail="上传的音频文件不能为空。")

    parsed_context = _parse_optional_json(student_context, "student_context")
    parsed_history = _parse_optional_json(conversation_history, "conversation_history")
    clean_session_id = session_id.strip() if session_id else None
    logger.info(
        "Received audio support request session_id=%s filename=%s size=%s",
        clean_session_id or "-",
        file.filename or "audio.wav",
        len(audio_bytes),
    )

    merged_history = _merge_conversation_history(clean_session_id, parsed_history)
    parsed_context = enrich_student_context_with_memory(
        parsed_context,
        merged_history,
    )
    previous_entropy_score = None
    entropy_trace: list[dict[str, Any]] = []
    if clean_session_id:
        session_store = get_session_store()
        previous_entropy = session_store.get_last_entropy(clean_session_id)
        previous_entropy_score = int(previous_entropy["score"]) if previous_entropy else None
        entropy_trace = session_store.get_entropy_trace(clean_session_id)
        previous_records = session_store.list_support_responses(session_id=clean_session_id)
        continuity_summary = build_session_continuity_summary(
            session_id=clean_session_id,
            records=previous_records,
            conversation_history=merged_history,
            entropy_trace=entropy_trace,
        )
        parsed_context = enrich_student_context_with_continuity(parsed_context, continuity_summary)
        goal_attainment_timeline = build_goal_attainment_timeline(previous_records)
        strategy_reselection = build_strategy_reselection_decision(
            goal_attainment_timeline=goal_attainment_timeline,
            session_continuity=continuity_summary,
            latest_record=previous_records[-1] if previous_records else None,
        )
        parsed_context = enrich_student_context_with_strategy_reselection(parsed_context, strategy_reselection)
        feedback_adaptation = build_feedback_adaptation(
            feedback_summary=session_store.summarize_intervention_feedback(clean_session_id),
            recent_feedback=session_store.get_intervention_feedback(clean_session_id, limit=5),
        )
        adjustment_loop = build_entropy_adjustment_loop(
            session_id=clean_session_id,
            records=previous_records,
            feedback_summary=session_store.summarize_intervention_feedback(clean_session_id),
            recent_feedback=session_store.get_intervention_feedback(clean_session_id, limit=5),
            session_continuity=continuity_summary,
            strategy_reselection=strategy_reselection,
            audit_summary=session_store.get_intervention_audits(clean_session_id, limit=20)["audit_summary"],
        )
        parsed_context = enrich_student_context_with_adjustment_loop(parsed_context, adjustment_loop)
        parsed_context = enrich_student_context_with_care_plan(
            parsed_context,
            session_store.get_session_care_plan(clean_session_id)["session_care_plan"],
        )
        parsed_context = enrich_student_context_with_intervention_next_step(
            parsed_context,
            session_store.get_session_intervention_effectiveness(clean_session_id)["intervention_next_step"],
        )
        parsed_context = enrich_student_context_with_session_tracking(
            parsed_context,
            session_store.get_session_tracking(clean_session_id)["session_tracking"],
        )
        parsed_context = enrich_student_context_with_strategy_version(
            parsed_context,
            session_store.get_session_strategy_version(clean_session_id)["strategy_version"],
        )
    else:
        feedback_adaptation = build_feedback_adaptation()
        adjustment_loop = build_entropy_adjustment_loop(session_id="anonymous")
        parsed_context = enrich_student_context_with_adjustment_loop(parsed_context, adjustment_loop)

    result = get_agent().handle_audio(
        file_bytes=audio_bytes,
        filename=file.filename or "audio.wav",
        content_type=file.content_type,
        student_context=parsed_context,
        conversation_history=merged_history,
        previous_entropy_score=previous_entropy_score,
        entropy_trace=entropy_trace,
        feedback_adaptation=feedback_adaptation,
    )
    response = result.to_dict()

    _update_session_state(
        session_id=clean_session_id,
        user_text=result.transcript or "",
        assistant_text=result.reply_text,
        result=result,
        response=response,
    )
    _apply_session_escalation(session_id=clean_session_id, response=response)
    _store_referral_event_if_needed(session_id=clean_session_id, response=response)

    session_store = get_session_store()
    session_store.store_support_response(
        session_id=clean_session_id,
        response_id=result.response_id,
        source=result.source,
        input_text=result.input_text,
        transcript=result.transcript,
        student_context=parsed_context,
        conversation_history=merged_history,
        response_payload=response,
    )
    logger.info("Completed audio support request response_id=%s", result.response_id)
    return response


@app.get("/api/v1/sessions/{session_id}")
def get_session_history(session_id: str) -> dict[str, Any]:
    session_store = get_session_store()
    history = session_store.get_history(session_id)
    entropy_trace = session_store.get_entropy_trace(session_id)
    logger.info("Session history requested session_id=%s history_messages=%s", session_id, len(history))
    return {
        "session_id": session_id,
        "history_messages": len(history),
        "conversation_history": history,
        "entropy_trace": entropy_trace,
    }


@app.get("/api/v1/sessions/{session_id}/analysis")
def get_session_analysis(session_id: str) -> dict[str, Any]:
    session_store = get_session_store()
    analysis = session_store.get_session_analysis(session_id)
    logger.info("Session analysis requested session_id=%s total=%s", session_id, analysis["total_responses"])
    return analysis


@app.get("/api/v1/sessions/{session_id}/view")
def get_session_role_view(session_id: str, role: str = "student") -> dict[str, Any]:
    try:
        normalized_role = normalize_view_role(role)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"role must be one of: {', '.join(sorted(SUPPORTED_VIEW_ROLES))}",
        ) from exc

    session_store = get_session_store()
    records = session_store.list_support_responses(session_id=session_id, limit=None)
    analysis = session_store.get_session_analysis(session_id)
    latest_response = records[-1]["response"] if records else None
    logger.info("Session role view requested session_id=%s role=%s", session_id, normalized_role)
    return {
        "session_id": session_id,
        "role": normalized_role,
        "latest_response": (
            project_support_response_for_role(latest_response, normalized_role)
            if isinstance(latest_response, dict)
            else None
        ),
        "analysis": project_session_analysis_for_role(analysis, normalized_role),
    }


@app.get("/api/v1/sessions/{session_id}/audit")
def get_session_audit(session_id: str, limit: int | None = 50) -> dict[str, Any]:
    session_store = get_session_store()
    audit = session_store.get_intervention_audits(session_id, limit=limit)
    logger.info("Intervention audit requested session_id=%s total=%s", session_id, audit["total_audits"])
    return audit


@app.get("/api/v1/sessions/{session_id}/memory")
def get_session_memory(session_id: str) -> dict[str, Any]:
    session_store = get_session_store()
    memory = session_store.get_session_memory(session_id)
    logger.info("Session memory requested session_id=%s messages=%s", session_id, memory["history_messages"])
    return memory


@app.get("/api/v1/sessions/{session_id}/trend-warning")
def get_session_trend_warning(session_id: str) -> dict[str, Any]:
    session_store = get_session_store()
    warning = session_store.get_session_trend_warning(session_id)
    logger.info(
        "Trend warning requested session_id=%s level=%s",
        session_id,
        warning["trend_warning"]["level"],
    )
    return warning


@app.get("/api/v1/sessions/{session_id}/care-plan")
def get_session_care_plan(session_id: str) -> dict[str, Any]:
    session_store = get_session_store()
    care_plan = session_store.get_session_care_plan(session_id)
    logger.info(
        "Session care plan requested session_id=%s phase=%s priority=%s",
        session_id,
        care_plan["session_care_plan"]["care_phase"],
        care_plan["session_care_plan"]["priority"],
    )
    return care_plan


@app.get("/api/v1/sessions/{session_id}/reply-quality")
def get_session_reply_quality(session_id: str, limit: int | None = None) -> dict[str, Any]:
    session_store = get_session_store()
    quality = session_store.get_session_reply_quality(session_id, limit=limit)
    logger.info(
        "Reply quality requested session_id=%s needs_review=%s",
        session_id,
        quality["summary"]["needs_review"],
    )
    return quality


@app.get("/api/v1/sessions/{session_id}/intervention-effectiveness")
def get_session_intervention_effectiveness(session_id: str) -> dict[str, Any]:
    session_store = get_session_store()
    effectiveness = session_store.get_session_intervention_effectiveness(session_id)
    summary = effectiveness["intervention_effectiveness"]["summary"]
    logger.info(
        "Intervention effectiveness requested session_id=%s status=%s",
        session_id,
        summary["overall_status"],
    )
    return effectiveness


@app.get("/api/v1/sessions/{session_id}/intervention-next-step")
def get_session_intervention_next_step(session_id: str) -> dict[str, Any]:
    session_store = get_session_store()
    next_step = session_store.get_session_intervention_next_step(session_id)
    logger.info(
        "Intervention next step requested session_id=%s mode=%s priority=%s",
        session_id,
        next_step["intervention_next_step"]["next_reply_mode"],
        next_step["intervention_next_step"]["priority"],
    )
    return next_step


@app.get("/api/v1/sessions/{session_id}/tracking")
def get_session_tracking(session_id: str) -> dict[str, Any]:
    session_store = get_session_store()
    tracking = session_store.get_session_tracking(session_id)
    logger.info(
        "Session tracking requested session_id=%s stage=%s",
        session_id,
        tracking["session_tracking"]["stage"],
    )
    return tracking


@app.get("/api/v1/sessions/{session_id}/strategy-version")
def get_session_strategy_version(session_id: str) -> dict[str, Any]:
    session_store = get_session_store()
    version = session_store.get_session_strategy_version(session_id)
    logger.info(
        "Strategy version requested session_id=%s decision=%s target=%s",
        session_id,
        version["strategy_version"]["decision"],
        version["strategy_version"]["target_strategy_family"],
    )
    return version


@app.get("/api/v1/sessions/{session_id}/strategy-layer")
def get_session_strategy_layer(session_id: str) -> dict[str, Any]:
    session_store = get_session_store()
    strategy_layer = session_store.get_session_strategy_layer(session_id)
    latest = strategy_layer["strategy_layer_summary"].get("latest") or {}
    logger.info(
        "Strategy layer requested session_id=%s state=%s strategy=%s",
        session_id,
        latest.get("primary_state"),
        latest.get("strategy_id"),
    )
    return strategy_layer


@app.get("/api/v1/sessions/{session_id}/decision-trace")
def get_session_decision_trace(session_id: str, limit: int | None = 50) -> dict[str, Any]:
    session_store = get_session_store()
    trace = session_store.get_session_decision_trace(session_id, limit=limit)
    logger.info(
        "Decision trace requested session_id=%s returned=%s attention=%s",
        session_id,
        trace["returned_turns"],
        trace["summary"]["needs_attention"],
    )
    return trace


@app.get("/api/v1/sessions/{session_id}/entropy-loop")
def get_session_entropy_reduction_loop(session_id: str) -> dict[str, Any]:
    session_store = get_session_store()
    loop = session_store.get_session_entropy_reduction_loop(session_id)
    summary = loop["entropy_reduction_loop"]["summary"]
    logger.info(
        "Entropy reduction loop requested session_id=%s status=%s score=%s",
        session_id,
        summary["overall_status"],
        summary["latest_loop_score"],
    )
    return loop


@app.get("/api/v1/sessions/{session_id}/referrals")
def get_session_referrals(session_id: str, limit: int | None = None) -> dict[str, Any]:
    session_store = get_session_store()
    events = session_store.get_referral_events(session_id, limit=limit)
    logger.info("Referral events requested session_id=%s total=%s", session_id, len(events))
    return {
        "session_id": session_id,
        "total_events": len(events),
        "referral_events": events,
    }


def _parse_human_intervention_payload(payload: dict[str, Any]) -> dict[str, Any]:
    status = str(payload.get("status", "")).strip()
    if status not in HUMAN_INTERVENTION_STATUSES:
        allowed = ", ".join(sorted(HUMAN_INTERVENTION_STATUSES))
        raise HTTPException(status_code=422, detail=f"status must be one of: {allowed}")

    tags = payload.get("tags") or []
    if not isinstance(tags, list):
        raise HTTPException(status_code=422, detail="tags must be a string array.")

    def optional_text(field_name: str) -> str | None:
        value = payload.get(field_name)
        if value is None:
            return None
        if not isinstance(value, str):
            raise HTTPException(status_code=422, detail=f"{field_name} must be a string.")
        return value.strip() or None

    return {
        "response_id": optional_text("response_id"),
        "status": status,
        "handler_id": optional_text("handler_id"),
        "note": optional_text("note"),
        "next_action": optional_text("next_action"),
        "tags": [str(tag).strip() for tag in tags if str(tag).strip()],
    }


@app.post("/api/v1/sessions/{session_id}/human-interventions")
def append_session_human_intervention(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    parsed = _parse_human_intervention_payload(payload)
    session_store = get_session_store()
    intervention = session_store.append_human_intervention(
        session_id=session_id,
        response_id=parsed["response_id"],
        status=parsed["status"],
        handler_id=parsed["handler_id"],
        note=parsed["note"],
        next_action=parsed["next_action"],
        tags=parsed["tags"],
    )
    logger.info(
        "Human intervention submitted session_id=%s status=%s handler=%s",
        session_id,
        parsed["status"],
        parsed["handler_id"] or "-",
    )
    return {
        "session_id": session_id,
        "human_intervention": intervention,
        "human_interventions": session_store.get_human_interventions(session_id),
    }


@app.get("/api/v1/sessions/{session_id}/human-interventions")
def get_session_human_interventions(session_id: str, limit: int | None = None) -> dict[str, Any]:
    session_store = get_session_store()
    interventions = session_store.get_human_interventions(session_id, limit=limit)
    logger.info("Human interventions requested session_id=%s total=%s", session_id, len(interventions))
    return {
        "session_id": session_id,
        "total_interventions": len(interventions),
        "latest_human_intervention": interventions[-1] if interventions else None,
        "human_interventions": interventions,
    }


def _parse_feedback_payload(payload: dict[str, Any]) -> dict[str, Any]:
    response_id = str(payload.get("response_id", "")).strip()
    if not response_id:
        raise HTTPException(status_code=422, detail="response_id 不能为空。")

    try:
        helpful_score = int(payload.get("helpful_score"))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="helpful_score 必须是 -2 到 2 的整数。") from exc
    if helpful_score < -2 or helpful_score > 2:
        raise HTTPException(status_code=422, detail="helpful_score 必须在 -2 到 2 之间。")

    mood_after = None
    if payload.get("mood_after") is not None:
        try:
            mood_after = int(payload.get("mood_after"))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="mood_after 必须是 0 到 100 的整数。") from exc
        if mood_after < 0 or mood_after > 100:
            raise HTTPException(status_code=422, detail="mood_after 必须在 0 到 100 之间。")

    tags = payload.get("tags") or []
    if not isinstance(tags, list):
        raise HTTPException(status_code=422, detail="tags 必须是字符串数组。")

    user_note = payload.get("user_note")
    if user_note is not None and not isinstance(user_note, str):
        raise HTTPException(status_code=422, detail="user_note 必须是字符串。")

    return {
        "response_id": response_id,
        "helpful_score": helpful_score,
        "mood_after": mood_after,
        "user_note": user_note,
        "tags": [str(tag) for tag in tags],
    }


@app.post("/api/v1/sessions/{session_id}/feedback")
def submit_session_feedback(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    parsed = _parse_feedback_payload(payload)
    session_store = get_session_store()
    feedback = session_store.append_intervention_feedback(
        session_id=session_id,
        response_id=parsed["response_id"],
        helpful_score=parsed["helpful_score"],
        mood_after=parsed["mood_after"],
        user_note=parsed["user_note"],
        tags=parsed["tags"],
    )
    summary = session_store.summarize_intervention_feedback(session_id)
    feedback_adaptation = build_feedback_adaptation(
        feedback_summary=summary,
        recent_feedback=session_store.get_intervention_feedback(session_id, limit=5),
    )
    logger.info(
        "Intervention feedback submitted session_id=%s response_id=%s helpful_score=%s",
        session_id,
        parsed["response_id"],
        parsed["helpful_score"],
    )
    return {
        "session_id": session_id,
        "feedback": feedback,
        "feedback_summary": summary,
        "feedback_adaptation": asdict(feedback_adaptation),
    }


@app.get("/api/v1/sessions/{session_id}/feedback")
def get_session_feedback(session_id: str, limit: int | None = None) -> dict[str, Any]:
    session_store = get_session_store()
    feedback = session_store.get_intervention_feedback(session_id, limit=limit)
    summary = session_store.summarize_intervention_feedback(session_id)
    feedback_adaptation = build_feedback_adaptation(
        feedback_summary=summary,
        recent_feedback=feedback,
    )
    logger.info("Intervention feedback requested session_id=%s total=%s", session_id, len(feedback))
    return {
        "session_id": session_id,
        "total_feedback": len(feedback),
        "feedback": feedback,
        "feedback_summary": summary,
        "feedback_adaptation": asdict(feedback_adaptation),
    }


@app.get("/api/v1/analytics/overview")
def get_overview_analytics(limit: int = 200) -> dict[str, Any]:
    session_store = get_session_store()
    stats = session_store.get_overview_stats(limit=limit)
    logger.info("Overview analytics requested total=%s", stats["total_records"])
    return stats


@app.get("/api/v1/analytics/care-queue")
def get_care_queue(
    limit: int = 100,
    include_low_priority: bool = False,
    include_resolved: bool = False,
) -> dict[str, Any]:
    session_store = get_session_store()
    queue = session_store.get_care_queue(
        limit=limit,
        include_low_priority=include_low_priority,
        include_resolved=include_resolved,
    )
    logger.info(
        "Care queue requested total=%s include_low=%s include_resolved=%s",
        queue["total_items"],
        include_low_priority,
        include_resolved,
    )
    return queue


@app.get("/api/v1/analytics/reply-quality")
def get_reply_quality_overview(limit: int = 200) -> dict[str, Any]:
    session_store = get_session_store()
    quality = session_store.get_reply_quality_overview(limit=limit)
    logger.info(
        "Reply quality overview requested total=%s needs_review=%s",
        quality["total_records"],
        quality["summary"]["needs_review"],
    )
    return quality


@app.get("/api/v1/analytics/intervention-effectiveness")
def get_intervention_effectiveness_overview(limit: int = 200) -> dict[str, Any]:
    session_store = get_session_store()
    overview = session_store.get_intervention_effectiveness_overview(limit=limit)
    logger.info(
        "Intervention effectiveness overview requested sessions=%s",
        overview["total_sessions"],
    )
    return overview


@app.get("/api/v1/analytics/session-tracking")
def get_session_tracking_overview(limit: int = 200) -> dict[str, Any]:
    session_store = get_session_store()
    overview = session_store.get_session_tracking_overview(limit=limit)
    logger.info(
        "Session tracking overview requested sessions=%s",
        overview["total_sessions"],
    )
    return overview


@app.get("/api/v1/analytics/strategy-version")
def get_strategy_version_overview(limit: int = 200) -> dict[str, Any]:
    session_store = get_session_store()
    overview = session_store.get_strategy_version_overview(limit=limit)
    logger.info(
        "Strategy version overview requested sessions=%s switches=%s",
        overview["total_sessions"],
        overview["sessions_needing_strategy_switch"],
    )
    return overview


@app.get("/api/v1/analytics/strategy-layer")
def get_strategy_layer_overview(limit: int = 200) -> dict[str, Any]:
    session_store = get_session_store()
    overview = session_store.get_strategy_layer_overview(limit=limit)
    logger.info(
        "Strategy layer overview requested sessions=%s records=%s",
        overview["total_sessions"],
        overview["source_records_seen"],
    )
    return overview


@app.get("/api/v1/analytics/decision-trace")
def get_decision_trace_overview(limit: int = 200) -> dict[str, Any]:
    session_store = get_session_store()
    overview = session_store.get_decision_trace_overview(limit=limit)
    logger.info(
        "Decision trace overview requested sessions=%s records=%s",
        overview["total_sessions"],
        overview["source_records_seen"],
    )
    return overview


@app.get("/api/v1/analytics/entropy-loop")
def get_entropy_reduction_loop_overview(limit: int = 200) -> dict[str, Any]:
    session_store = get_session_store()
    overview = session_store.get_entropy_reduction_loop_overview(limit=limit)
    logger.info(
        "Entropy reduction loop overview requested sessions=%s",
        overview["total_sessions"],
    )
    return overview


@app.get("/api/v1/analytics/reply-quality/bad-cases")
def get_reply_quality_bad_cases(
    session_id: str | None = None,
    source_limit: int = 200,
    limit: int = 50,
    min_quality_score: int = 80,
) -> dict[str, Any]:
    session_store = get_session_store()
    bad_cases = session_store.get_reply_quality_bad_cases(
        session_id=session_id,
        source_limit=source_limit,
        limit=limit,
        min_quality_score=min_quality_score,
    )
    logger.info(
        "Reply quality bad cases requested records=%s cases=%s session_id=%s",
        bad_cases["records_seen"],
        bad_cases["bad_case_count"],
        session_id or "-",
    )
    return bad_cases


@app.get("/api/v1/analytics/refinement-plan")
def get_quality_refinement_plan(
    session_id: str | None = None,
    source_limit: int = 200,
    bad_case_limit: int = 100,
    min_quality_score: int = 80,
    max_examples_per_bucket: int = 8,
) -> dict[str, Any]:
    session_store = get_session_store()
    plan = session_store.get_quality_refinement_plan(
        session_id=session_id,
        source_limit=source_limit,
        bad_case_limit=bad_case_limit,
        min_quality_score=min_quality_score,
        max_examples_per_bucket=max_examples_per_bucket,
    )
    logger.info(
        "Quality refinement plan requested cases=%s routes=%s session_id=%s",
        plan["total_bad_cases"],
        plan["route_counts"],
        session_id or "-",
    )
    return plan


@app.delete("/api/v1/sessions/{session_id}")
def clear_session_history(session_id: str) -> dict[str, Any]:
    # 研究测试时经常需要从干净状态重新跑同一个案例，这里提供显式清空入口。
    session_store = get_session_store()
    session_store.clear(session_id)
    logger.info("Session cleared session_id=%s", session_id)
    return {
        "session_id": session_id,
        "status": "cleared",
    }


@app.post("/api/v1/entropy/evaluate")
def evaluate_entropy(payload: dict[str, Any]) -> dict[str, Any]:
    text = str(payload.get("text", "")).strip()
    if not text:
        raise HTTPException(status_code=422, detail="text 不能为空。")

    student_context = payload.get("student_context") or {}
    conversation_history = payload.get("conversation_history") or []
    if not isinstance(student_context, dict):
        raise HTTPException(status_code=422, detail="student_context 必须是对象。")
    if not isinstance(conversation_history, list):
        raise HTTPException(status_code=422, detail="conversation_history 必须是数组。")

    # 单独暴露熵评估接口，便于你在研究和测试阶段直接观察熵值变化。
    risk = evaluate_text_risk(text)
    entropy = evaluate_psychological_entropy(
        text,
        risk,
        student_context=student_context,
        conversation_history=conversation_history,
    )
    agent = get_agent()
    campus_resources = agent.retriever.retrieve(text, risk) if agent.retriever else []
    entropy_reduction = build_entropy_reduction_strategy(entropy, risk, campus_resources)
    logger.info("Entropy-only evaluation requested score=%s state=%s", entropy.score, entropy.balance_state)
    return {
        "input_text": text,
        "risk": asdict(risk),
        "entropy": asdict(entropy),
        "campus_resources": [asdict(item) for item in campus_resources],
        "entropy_reduction": asdict(entropy_reduction),
    }


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("campus_support_agent.main:app", host="0.0.0.0", port=settings.app_port, reload=True)
