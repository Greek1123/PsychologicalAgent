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
from .config import Settings
from .entropy import evaluate_psychological_entropy
from .logging_utils import configure_logging, get_logger
from .providers import build_llm_provider, build_stt_provider
from .reduction import build_entropy_reduction_strategy
from .retrieval import CampusKnowledgeRetriever
from .safety import evaluate_text_risk
from .storage import SQLiteSessionStore


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    configure_logging(settings)
    return settings


logger = get_logger("main")
STATIC_DIR = Path(__file__).resolve().parent / "static"
APP_HTML = STATIC_DIR / "app.html"


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
        response["entropy"]["trend"] = {
            "previous_score": int(previous_entropy["score"]),
            "delta": delta,
            "direction": direction,
        }
        logger.info(
            "Entropy trend updated session_id=%s previous=%s current=%s delta=%s direction=%s",
            session_id,
            previous_entropy["score"],
            result.entropy.score,
            delta,
            direction,
        )

    response["session"] = {
        "session_id": session_id,
        "history_messages": history_size,
        "entropy_trace_points": entropy_trace_size,
    }


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

    manual_referral_recommended = bool(reasons) or current_urgency == "urgent"
    if manual_referral_recommended and current_urgency == "watch":
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

    result = get_agent().handle_text(
        text=text,
        student_context=student_context,
        conversation_history=merged_history,
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

    result = get_agent().handle_audio(
        file_bytes=audio_bytes,
        filename=file.filename or "audio.wav",
        content_type=file.content_type,
        student_context=parsed_context,
        conversation_history=merged_history,
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
    }


@app.get("/api/v1/sessions/{session_id}/feedback")
def get_session_feedback(session_id: str, limit: int | None = None) -> dict[str, Any]:
    session_store = get_session_store()
    feedback = session_store.get_intervention_feedback(session_id, limit=limit)
    summary = session_store.summarize_intervention_feedback(session_id)
    logger.info("Intervention feedback requested session_id=%s total=%s", session_id, len(feedback))
    return {
        "session_id": session_id,
        "total_feedback": len(feedback),
        "feedback": feedback,
        "feedback_summary": summary,
    }


@app.get("/api/v1/analytics/overview")
def get_overview_analytics(limit: int = 200) -> dict[str, Any]:
    session_store = get_session_store()
    stats = session_store.get_overview_stats(limit=limit)
    logger.info("Overview analytics requested total=%s", stats["total_records"])
    return stats


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
