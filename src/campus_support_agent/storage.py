from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import Any

from .adjustment_loop import build_entropy_adjustment_loop
from .care_plan import build_session_care_plan
from .dialogue_memory import build_memory_context
from .entropy_reduction_loop import build_entropy_reduction_loop, build_entropy_reduction_loop_overview
from .goal_attainment import build_goal_attainment_timeline, summarize_goal_attainment
from .intervention_audit import build_intervention_audit_timeline, summarize_intervention_audits
from .intervention_effectiveness import (
    build_intervention_effectiveness_overview,
    build_intervention_effectiveness_report,
)
from .intervention_next_step import build_intervention_next_step
from .logging_utils import get_logger
from .longitudinal_profile import build_longitudinal_state_profile
from .care_pathway import build_care_pathway_decision
from .entropy_outcome import build_entropy_reduction_outcome
from .processing_consistency import build_processing_consistency_report
from .quality_refinement import build_quality_refinement_plan
from .reply_quality import build_reply_quality_report
from .reply_quality_export import build_reply_quality_bad_cases
from .schemas import CareQueueItem
from .session_continuity import build_session_continuity_summary
from .session_insights import build_session_insight
from .session_tracking import build_session_tracking_snapshot
from .strategy_reselection import build_strategy_reselection_decision
from .strategy_versioning import build_strategy_version_decision
from .trend_warning import build_entropy_trend_warning, summarize_trend_warnings


logger = get_logger("storage")


def _flatten_response_summary(response: dict[str, Any]) -> dict[str, Any]:
    risk = response.get("risk") or {}
    entropy = response.get("entropy") or {}
    state_profile = response.get("state_profile") or {}
    intervention_strategy = response.get("intervention_strategy") or {}
    dynamic_adjustment = response.get("dynamic_adjustment") or {}
    feedback_adaptation = response.get("feedback_adaptation") or {}
    entropy_orchestration = response.get("entropy_orchestration") or {}
    reduction_goal = response.get("reduction_goal") or {}
    referral_explanation = response.get("referral_explanation") or {}
    adjustment_loop = response.get("adjustment_loop") or {}
    local_policy = response.get("local_policy") or {}
    referral_decision = response.get("referral_decision") or {}
    processing_summary = response.get("processing_summary") or {}
    return {
        "reply_text": response.get("reply_text"),
        "risk_level": risk.get("level"),
        "risk_score": risk.get("score"),
        "entropy_score": entropy.get("score"),
        "entropy_level": entropy.get("level"),
        "balance_state": entropy.get("balance_state"),
        "state_profile": state_profile,
        "primary_state": state_profile.get("primary_state"),
        "state_intensity": state_profile.get("intensity"),
        "recommended_focus": state_profile.get("recommended_focus"),
        "intervention_strategy": intervention_strategy,
        "strategy_id": intervention_strategy.get("strategy_id"),
        "strategy_priority": intervention_strategy.get("priority"),
        "response_mode": intervention_strategy.get("response_mode"),
        "dynamic_adjustment": dynamic_adjustment,
        "dynamic_stability_state": dynamic_adjustment.get("stability_state"),
        "dynamic_action": dynamic_adjustment.get("action"),
        "dynamic_intensity_shift": dynamic_adjustment.get("intensity_shift"),
        "dynamic_should_refer": dynamic_adjustment.get("should_refer"),
        "feedback_adaptation": feedback_adaptation,
        "feedback_adaptation_mode": feedback_adaptation.get("mode"),
        "feedback_question_pressure": feedback_adaptation.get("question_pressure"),
        "entropy_orchestration": entropy_orchestration,
        "orchestration_route": entropy_orchestration.get("route"),
        "orchestration_next_focus": entropy_orchestration.get("next_focus"),
        "orchestration_user_visible_goal": entropy_orchestration.get("user_visible_goal"),
        "reduction_goal": reduction_goal,
        "reduction_goal_active_driver": reduction_goal.get("active_driver"),
        "reduction_goal_text": reduction_goal.get("reduction_goal"),
        "reduction_goal_priority": reduction_goal.get("priority"),
        "reduction_goal_micro_intervention": reduction_goal.get("micro_intervention"),
        "reduction_goal_target_delta": reduction_goal.get("target_entropy_delta"),
        "referral_explanation": referral_explanation,
        "referral_explanation_level": referral_explanation.get("referral_level"),
        "referral_explanation_channel": referral_explanation.get("recommended_channel"),
        "referral_explanation_should_escalate": referral_explanation.get("should_escalate"),
        "referral_explanation_urgency": referral_explanation.get("urgency"),
        "adjustment_loop": adjustment_loop,
        "adjustment_loop_action": adjustment_loop.get("loop_action"),
        "adjustment_loop_priority": adjustment_loop.get("priority"),
        "adjustment_loop_reply_mode": adjustment_loop.get("next_reply_mode"),
        "adjustment_loop_question_policy": adjustment_loop.get("question_policy"),
        "adjustment_loop_human_followup": adjustment_loop.get("human_followup_policy"),
        "local_policy": local_policy,
        "local_policy_name": local_policy.get("policy_name"),
        "local_policy_stage": local_policy.get("policy_stage"),
        "local_policy_escalation_hint": local_policy.get("escalation_hint"),
        "referral_decision": referral_decision,
        "referral_should_refer": referral_decision.get("should_refer"),
        "referral_urgency": referral_decision.get("urgency"),
        "processing_summary": processing_summary,
        "processing_route": processing_summary.get("route"),
        "processing_safety_priority": processing_summary.get("safety_priority"),
        "processing_reply_source": processing_summary.get("reply_source"),
        "processing_next_backend_action": processing_summary.get("next_backend_action"),
    }


class SQLiteSessionStore:
    def __init__(self, db_path: str, max_messages: int = 12) -> None:
        self.db_path = Path(db_path)
        self.max_messages = max_messages
        self._lock = Lock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        # 研究阶段先把三类核心数据落盘：消息、熵轨迹、完整支持回合。
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_conversation_session
                ON conversation_messages (session_id, id);

                CREATE TABLE IF NOT EXISTS entropy_trace (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    response_id TEXT NOT NULL,
                    score INTEGER NOT NULL,
                    level INTEGER NOT NULL,
                    balance_state TEXT NOT NULL,
                    dominant_drivers_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_entropy_session
                ON entropy_trace (session_id, id);

                CREATE TABLE IF NOT EXISTS support_responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    response_id TEXT NOT NULL UNIQUE,
                    source TEXT NOT NULL,
                    input_text TEXT NOT NULL,
                    transcript TEXT,
                    student_context_json TEXT NOT NULL,
                    conversation_history_json TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_support_session
                ON support_responses (session_id, id);

                CREATE TABLE IF NOT EXISTS referral_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    response_id TEXT NOT NULL UNIQUE,
                    urgency TEXT NOT NULL,
                    reasons_json TEXT NOT NULL,
                    policy_name TEXT,
                    risk_level TEXT,
                    entropy_score INTEGER,
                    manual_referral_recommended INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_referral_session
                ON referral_events (session_id, id);

                CREATE TABLE IF NOT EXISTS intervention_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    response_id TEXT NOT NULL,
                    helpful_score INTEGER NOT NULL,
                    mood_after INTEGER,
                    user_note TEXT,
                    tags_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_feedback_session
                ON intervention_feedback (session_id, id);

                CREATE TABLE IF NOT EXISTS human_interventions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    response_id TEXT,
                    status TEXT NOT NULL,
                    handler_id TEXT,
                    note TEXT,
                    next_action TEXT,
                    tags_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_human_intervention_session
                ON human_interventions (session_id, id);
                """
            )
        logger.info("SQLite session store initialized at %s", self.db_path)

    def get_history(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content
                FROM (
                    SELECT id, role, content
                    FROM conversation_messages
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
                """,
                (session_id, self.max_messages),
            ).fetchall()
        history = [{"role": row["role"], "content": row["content"]} for row in rows]
        logger.debug("Loaded %s conversation messages for session_id=%s", len(history), session_id)
        return history

    def get_entropy_trace(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT response_id, score, level, balance_state, dominant_drivers_json
                FROM (
                    SELECT id, response_id, score, level, balance_state, dominant_drivers_json
                    FROM entropy_trace
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
                """,
                (session_id, self.max_messages),
            ).fetchall()
        trace = [
            {
                "response_id": row["response_id"],
                "score": row["score"],
                "level": row["level"],
                "balance_state": row["balance_state"],
                "dominant_drivers": json.loads(row["dominant_drivers_json"]),
            }
            for row in rows
        ]
        logger.debug("Loaded %s entropy points for session_id=%s", len(trace), session_id)
        return trace

    def get_last_entropy(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT response_id, score, level, balance_state, dominant_drivers_json
                FROM entropy_trace
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "response_id": row["response_id"],
            "score": row["score"],
            "level": row["level"],
            "balance_state": row["balance_state"],
            "dominant_drivers": json.loads(row["dominant_drivers_json"]),
        }

    def append_exchange(self, session_id: str, *, user_text: str, assistant_text: str) -> int:
        with self._lock, self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO conversation_messages (session_id, role, content)
                VALUES (?, ?, ?)
                """,
                [
                    (session_id, "user", user_text),
                    (session_id, "assistant", assistant_text),
                ],
            )
            count = connection.execute(
                "SELECT COUNT(*) AS total FROM conversation_messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()["total"]
        visible_count = min(int(count), self.max_messages)
        logger.info("Stored conversation exchange for session_id=%s visible_messages=%s", session_id, visible_count)
        return visible_count

    def append_entropy_snapshot(
        self,
        session_id: str,
        *,
        response_id: str,
        score: int,
        level: int,
        balance_state: str,
        dominant_drivers: list[str],
    ) -> int:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO entropy_trace (
                    session_id,
                    response_id,
                    score,
                    level,
                    balance_state,
                    dominant_drivers_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    response_id,
                    score,
                    level,
                    balance_state,
                    json.dumps(dominant_drivers, ensure_ascii=False),
                ),
            )
            count = connection.execute(
                "SELECT COUNT(*) AS total FROM entropy_trace WHERE session_id = ?",
                (session_id,),
            ).fetchone()["total"]
        visible_count = min(int(count), self.max_messages)
        logger.info(
            "Stored entropy snapshot for session_id=%s score=%s visible_points=%s",
            session_id,
            score,
            visible_count,
        )
        return visible_count

    def store_support_response(
        self,
        *,
        session_id: str | None,
        response_id: str,
        source: str,
        input_text: str,
        transcript: str | None,
        student_context: dict[str, Any],
        conversation_history: list[dict[str, Any]],
        response_payload: dict[str, Any],
    ) -> None:
        # 保存完整支持回合，后续可直接导出成训练样本。
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO support_responses (
                    session_id,
                    response_id,
                    source,
                    input_text,
                    transcript,
                    student_context_json,
                    conversation_history_json,
                    response_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    response_id,
                    source,
                    input_text,
                    transcript,
                    json.dumps(student_context, ensure_ascii=False),
                    json.dumps(conversation_history, ensure_ascii=False),
                    json.dumps(response_payload, ensure_ascii=False),
                ),
            )
        logger.info("Stored support response response_id=%s session_id=%s", response_id, session_id or "-")

    def append_referral_event(
        self,
        *,
        session_id: str,
        response_id: str,
        urgency: str,
        reasons: list[str],
        policy_name: str | None,
        risk_level: str | None,
        entropy_score: int | None,
        manual_referral_recommended: bool,
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO referral_events (
                    session_id,
                    response_id,
                    urgency,
                    reasons_json,
                    policy_name,
                    risk_level,
                    entropy_score,
                    manual_referral_recommended
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    response_id,
                    urgency,
                    json.dumps(reasons, ensure_ascii=False),
                    policy_name,
                    risk_level,
                    entropy_score,
                    1 if manual_referral_recommended else 0,
                ),
            )
        logger.info("Stored referral event response_id=%s session_id=%s urgency=%s", response_id, session_id, urgency)

    def get_referral_events(self, session_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        query = """
            SELECT response_id, urgency, reasons_json, policy_name, risk_level,
                   entropy_score, manual_referral_recommended, created_at
            FROM referral_events
            WHERE session_id = ?
            ORDER BY id ASC
        """
        params: list[Any] = [session_id]
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()

        return [
            {
                "response_id": row["response_id"],
                "urgency": row["urgency"],
                "reasons": json.loads(row["reasons_json"]),
                "policy_name": row["policy_name"],
                "risk_level": row["risk_level"],
                "entropy_score": row["entropy_score"],
                "manual_referral_recommended": bool(row["manual_referral_recommended"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def append_human_intervention(
        self,
        *,
        session_id: str,
        response_id: str | None,
        status: str,
        handler_id: str | None = None,
        note: str | None = None,
        next_action: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        clean_tags = [str(tag).strip() for tag in (tags or []) if str(tag).strip()]
        clean_note = note.strip() if isinstance(note, str) and note.strip() else None
        clean_next_action = next_action.strip() if isinstance(next_action, str) and next_action.strip() else None
        clean_handler_id = handler_id.strip() if isinstance(handler_id, str) and handler_id.strip() else None
        clean_response_id = response_id.strip() if isinstance(response_id, str) and response_id.strip() else None
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO human_interventions (
                    session_id,
                    response_id,
                    status,
                    handler_id,
                    note,
                    next_action,
                    tags_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    clean_response_id,
                    status,
                    clean_handler_id,
                    clean_note,
                    clean_next_action,
                    json.dumps(clean_tags, ensure_ascii=False),
                ),
            )
            row = connection.execute(
                """
                SELECT id, session_id, response_id, status, handler_id, note,
                       next_action, tags_json, created_at
                FROM human_interventions
                WHERE id = ?
                """,
                (cursor.lastrowid,),
            ).fetchone()
        logger.info("Stored human intervention session_id=%s status=%s", session_id, status)
        return _human_intervention_row_to_dict(row)

    def get_human_interventions(self, session_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        query = """
            SELECT id, session_id, response_id, status, handler_id, note,
                   next_action, tags_json, created_at
            FROM human_interventions
            WHERE session_id = ?
            ORDER BY id ASC
        """
        params: list[Any] = [session_id]
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return [_human_intervention_row_to_dict(row) for row in rows]

    def get_latest_human_interventions_by_session(self) -> dict[str, dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT h.id, h.session_id, h.response_id, h.status, h.handler_id,
                       h.note, h.next_action, h.tags_json, h.created_at
                FROM human_interventions AS h
                INNER JOIN (
                    SELECT session_id, MAX(id) AS latest_id
                    FROM human_interventions
                    GROUP BY session_id
                ) AS latest
                ON h.session_id = latest.session_id AND h.id = latest.latest_id
                """
            ).fetchall()
        return {str(row["session_id"]): _human_intervention_row_to_dict(row) for row in rows}

    def append_intervention_feedback(
        self,
        *,
        session_id: str,
        response_id: str,
        helpful_score: int,
        mood_after: int | None = None,
        user_note: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        clean_tags = [str(tag).strip() for tag in (tags or []) if str(tag).strip()]
        clean_note = user_note.strip() if isinstance(user_note, str) and user_note.strip() else None
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO intervention_feedback (
                    session_id,
                    response_id,
                    helpful_score,
                    mood_after,
                    user_note,
                    tags_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    response_id,
                    helpful_score,
                    mood_after,
                    clean_note,
                    json.dumps(clean_tags, ensure_ascii=False),
                ),
            )
            feedback_id = int(cursor.lastrowid)
        logger.info(
            "Stored intervention feedback id=%s session_id=%s response_id=%s helpful_score=%s",
            feedback_id,
            session_id,
            response_id,
            helpful_score,
        )
        return {
            "id": feedback_id,
            "session_id": session_id,
            "response_id": response_id,
            "helpful_score": helpful_score,
            "mood_after": mood_after,
            "user_note": clean_note,
            "tags": clean_tags,
        }

    def get_intervention_feedback(self, session_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        query = """
            SELECT id, session_id, response_id, helpful_score, mood_after, user_note, tags_json, created_at
            FROM intervention_feedback
            WHERE session_id = ?
            ORDER BY id ASC
        """
        params: list[Any] = [session_id]
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()

        return [
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "response_id": row["response_id"],
                "helpful_score": row["helpful_score"],
                "mood_after": row["mood_after"],
                "user_note": row["user_note"],
                "tags": json.loads(row["tags_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def list_feedback_cases(
        self,
        *,
        session_id: str | None = None,
        max_helpful_score: int | None = -1,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT
                feedback.id AS feedback_id,
                feedback.session_id AS feedback_session_id,
                feedback.response_id AS feedback_response_id,
                feedback.helpful_score,
                feedback.mood_after,
                feedback.user_note,
                feedback.tags_json,
                feedback.created_at AS feedback_created_at,
                responses.session_id AS response_session_id,
                responses.source,
                responses.input_text,
                responses.transcript,
                responses.student_context_json,
                responses.conversation_history_json,
                responses.response_json,
                responses.created_at AS response_created_at
            FROM intervention_feedback AS feedback
            INNER JOIN support_responses AS responses
            ON feedback.response_id = responses.response_id
        """
        clauses: list[str] = []
        params: list[Any] = []
        if session_id:
            clauses.append("feedback.session_id = ?")
            params.append(session_id)
        if max_helpful_score is not None:
            clauses.append("feedback.helpful_score <= ?")
            params.append(max_helpful_score)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY feedback.id ASC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()

        cases = []
        for row in rows:
            response = json.loads(row["response_json"])
            record = {
                "session_id": row["response_session_id"] or row["feedback_session_id"],
                "response_id": row["feedback_response_id"],
                "source": row["source"],
                "input_text": row["input_text"],
                "transcript": row["transcript"],
                "student_context": json.loads(row["student_context_json"]),
                "conversation_history": json.loads(row["conversation_history_json"]),
                "response": response,
                "created_at": row["response_created_at"],
                "feedback": {
                    "id": row["feedback_id"],
                    "session_id": row["feedback_session_id"],
                    "response_id": row["feedback_response_id"],
                    "helpful_score": row["helpful_score"],
                    "mood_after": row["mood_after"],
                    "user_note": row["user_note"],
                    "tags": json.loads(row["tags_json"]),
                    "created_at": row["feedback_created_at"],
                },
            }
            record.update(_flatten_response_summary(response))
            cases.append(record)
        logger.info("Loaded %s feedback cases for export", len(cases))
        return cases

    def summarize_intervention_feedback(self, session_id: str | None = None) -> dict[str, Any]:
        params: list[Any] = []
        where = ""
        if session_id:
            where = "WHERE session_id = ?"
            params.append(session_id)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT helpful_score, mood_after, tags_json
                FROM intervention_feedback
                {where}
                """,
                tuple(params),
            ).fetchall()

        if not rows:
            return {
                "total_feedback": 0,
                "average_helpful_score": None,
                "positive_count": 0,
                "negative_count": 0,
                "average_mood_after": None,
                "common_tags": {},
            }

        helpful_scores = [int(row["helpful_score"]) for row in rows]
        mood_values = [int(row["mood_after"]) for row in rows if row["mood_after"] is not None]
        common_tags: dict[str, int] = {}
        for row in rows:
            for tag in json.loads(row["tags_json"]):
                common_tags[tag] = common_tags.get(tag, 0) + 1

        return {
            "total_feedback": len(rows),
            "average_helpful_score": round(sum(helpful_scores) / len(helpful_scores), 2),
            "positive_count": sum(1 for score in helpful_scores if score > 0),
            "negative_count": sum(1 for score in helpful_scores if score < 0),
            "average_mood_after": round(sum(mood_values) / len(mood_values), 2) if mood_values else None,
            "common_tags": dict(sorted(common_tags.items(), key=lambda item: (-item[1], item[0]))),
        }

    def list_support_responses(
        self,
        *,
        session_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT session_id, response_id, source, input_text, transcript,
                   student_context_json, conversation_history_json, response_json, created_at
            FROM support_responses
        """
        params: list[Any] = []
        if session_id:
            query += " WHERE session_id = ?"
            params.append(session_id)
        query += " ORDER BY id ASC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()

        records = [
            {
                "session_id": row["session_id"],
                "response_id": row["response_id"],
                "source": row["source"],
                "input_text": row["input_text"],
                "transcript": row["transcript"],
                "student_context": json.loads(row["student_context_json"]),
                "conversation_history": json.loads(row["conversation_history_json"]),
                "response": json.loads(row["response_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]
        for record in records:
            record.update(_flatten_response_summary(record["response"]))
        logger.info("Loaded %s stored support responses for export", len(records))
        return records

    def clear(self, session_id: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM conversation_messages WHERE session_id = ?", (session_id,))
            connection.execute("DELETE FROM entropy_trace WHERE session_id = ?", (session_id,))
            connection.execute("DELETE FROM support_responses WHERE session_id = ?", (session_id,))
            connection.execute("DELETE FROM referral_events WHERE session_id = ?", (session_id,))
            connection.execute("DELETE FROM intervention_feedback WHERE session_id = ?", (session_id,))
            connection.execute("DELETE FROM human_interventions WHERE session_id = ?", (session_id,))
        logger.info("Cleared persisted session data for session_id=%s", session_id)

    def get_session_analysis(self, session_id: str) -> dict[str, Any]:
        records = self.list_support_responses(session_id=session_id)
        referral_events = self.get_referral_events(session_id)
        entropy_trace = self.get_entropy_trace(session_id)
        intervention_feedback = self.get_intervention_feedback(session_id)
        human_interventions = self.get_human_interventions(session_id)
        feedback_summary = self.summarize_intervention_feedback(session_id)
        conversation_history = self.get_history(session_id)
        conversation_memory = build_memory_context(conversation_history)
        session_insight = build_session_insight(
            session_id=session_id,
            records=records,
            entropy_trace=entropy_trace,
            referral_events=referral_events,
        )
        longitudinal_profile = build_longitudinal_state_profile(
            session_id=session_id,
            records=records,
            entropy_trace=entropy_trace,
            referral_events=referral_events,
            feedback_summary=feedback_summary,
        )
        latest = records[-1] if records else {}
        care_pathway = build_care_pathway_decision(
            session_id=session_id,
            longitudinal_profile=longitudinal_profile,
            latest_dynamic_adjustment=latest.get("dynamic_adjustment"),
            latest_feedback_adaptation=latest.get("feedback_adaptation"),
            latest_referral_decision=latest.get("referral_decision"),
            feedback_summary=feedback_summary,
        )
        entropy_reduction_outcome = build_entropy_reduction_outcome(
            session_id=session_id,
            records=records,
            entropy_trace=entropy_trace,
            feedback_summary=feedback_summary,
            care_pathway=care_pathway,
        )
        session_continuity = build_session_continuity_summary(
            session_id=session_id,
            records=records,
            conversation_history=conversation_history,
            entropy_trace=entropy_trace,
        )
        goal_attainment_timeline = build_goal_attainment_timeline(records)
        goal_attainment_summary = summarize_goal_attainment(goal_attainment_timeline)
        strategy_reselection = build_strategy_reselection_decision(
            goal_attainment_timeline=goal_attainment_timeline,
            session_continuity=session_continuity,
            latest_record=latest,
        )
        intervention_audit_timeline = build_intervention_audit_timeline(records, limit=20)
        intervention_audit_summary = summarize_intervention_audits(intervention_audit_timeline)
        next_adjustment_loop = build_entropy_adjustment_loop(
            session_id=session_id,
            records=records,
            feedback_summary=feedback_summary,
            recent_feedback=intervention_feedback,
            session_continuity=session_continuity,
            strategy_reselection=strategy_reselection,
            audit_summary=intervention_audit_summary,
        )
        trend_warning = build_entropy_trend_warning(
            session_id=session_id,
            records=records,
            entropy_trace=entropy_trace,
            feedback_summary=feedback_summary,
            audit_summary=intervention_audit_summary,
        )
        session_care_plan = build_session_care_plan(
            session_id=session_id,
            conversation_memory=conversation_memory,
            longitudinal_profile=asdict(longitudinal_profile),
            care_pathway=care_pathway,
            trend_warning=trend_warning,
            adjustment_loop=asdict(next_adjustment_loop),
            entropy_outcome=asdict(entropy_reduction_outcome),
            latest_record=latest,
        )
        reply_quality_report = build_reply_quality_report(records)
        intervention_effectiveness = build_intervention_effectiveness_report(
            session_id=session_id,
            records=records,
            feedback_summary=feedback_summary,
            reply_quality_summary=reply_quality_report["summary"],
            care_plan=asdict(session_care_plan),
        )
        intervention_next_step = build_intervention_next_step(
            session_id=session_id,
            intervention_effectiveness=intervention_effectiveness,
            care_plan=asdict(session_care_plan),
            reply_quality_summary=reply_quality_report["summary"],
            trend_warning=asdict(trend_warning),
            latest_record=latest,
        )
        session_tracking = build_session_tracking_snapshot(
            session_id=session_id,
            conversation_memory=conversation_memory,
            longitudinal_profile=asdict(longitudinal_profile),
            intervention_effectiveness=intervention_effectiveness,
            intervention_next_step=intervention_next_step,
            reply_quality_report=reply_quality_report,
            feedback_summary=feedback_summary,
            latest_record=latest,
        )
        strategy_version = build_strategy_version_decision(
            session_id=session_id,
            session_tracking=session_tracking,
            strategy_reselection=strategy_reselection,
            intervention_effectiveness=intervention_effectiveness,
            intervention_next_step=intervention_next_step,
            latest_record=latest,
        )
        entropy_reduction_loop = build_entropy_reduction_loop(
            session_id=session_id,
            records=records,
            feedback_summary=feedback_summary,
            reply_quality_summary=reply_quality_report["summary"],
            strategy_version=strategy_version,
            intervention_effectiveness=intervention_effectiveness,
        )
        processing_consistency = build_processing_consistency_report(records)
        if not records:
            return {
                "session_id": session_id,
                "total_responses": 0,
                "latest_reply_text": None,
                "latest_local_policy": None,
                "latest_referral_decision": None,
                "latest_dynamic_adjustment": None,
                "latest_feedback_adaptation": None,
                "latest_entropy_orchestration": None,
                "latest_reduction_goal": None,
                "latest_referral_explanation": None,
                "latest_adjustment_loop": None,
                "latest_processing_summary": None,
                "strategy_layer_summary": _build_strategy_layer_summary([]),
                "next_adjustment_loop": asdict(next_adjustment_loop),
                "trend_warning": asdict(trend_warning),
                "session_care_plan": asdict(session_care_plan),
                "risk_levels": {},
                "local_policies": {},
                "referral_urgencies": {},
                "dynamic_actions": {},
                "orchestration_routes": {},
                "reduction_goal_drivers": {},
                "reduction_goal_priorities": {},
                "referral_explanation_levels": {},
                "referral_explanation_channels": {},
                "adjustment_loop_actions": {},
                "adjustment_loop_priorities": {},
                "processing_routes": {},
                "processing_safety_priorities": {},
                "processing_next_backend_actions": {},
                "processing_timeline": [],
                "processing_summary": _summarize_processing_timeline([]),
                "processing_consistency": processing_consistency,
                "goal_attainment_timeline": [],
                "goal_attainment_summary": goal_attainment_summary,
                "strategy_reselection": strategy_reselection,
                "latest_intervention_audit": None,
                "intervention_audit_timeline": [],
                "intervention_audit_summary": intervention_audit_summary,
                "orchestration_timeline": [],
                "next_orchestration_recommendation": None,
                "referral_events": [],
                "intervention_feedback": intervention_feedback,
                "human_interventions": human_interventions,
                "latest_human_intervention": human_interventions[-1] if human_interventions else None,
                "feedback_summary": feedback_summary,
                "session_insight": session_insight,
                "session_continuity": session_continuity,
                "conversation_memory": conversation_memory,
                "longitudinal_profile": asdict(longitudinal_profile),
                "care_pathway": asdict(care_pathway),
                "entropy_reduction_outcome": asdict(entropy_reduction_outcome),
                "reply_quality": reply_quality_report,
                "intervention_effectiveness": intervention_effectiveness,
                "intervention_next_step": intervention_next_step,
                "session_tracking": session_tracking,
                "strategy_version": strategy_version,
                "entropy_reduction_loop": entropy_reduction_loop,
            }

        risk_levels: dict[str, int] = {}
        local_policies: dict[str, int] = {}
        referral_urgencies: dict[str, int] = {}
        dynamic_actions: dict[str, int] = {}
        orchestration_routes: dict[str, int] = {}
        reduction_goal_drivers: dict[str, int] = {}
        reduction_goal_priorities: dict[str, int] = {}
        referral_explanation_levels: dict[str, int] = {}
        referral_explanation_channels: dict[str, int] = {}
        adjustment_loop_actions: dict[str, int] = {}
        adjustment_loop_priorities: dict[str, int] = {}
        processing_routes: dict[str, int] = {}
        processing_safety_priorities: dict[str, int] = {}
        processing_next_backend_actions: dict[str, int] = {}
        for record in records:
            if record.get("risk_level"):
                risk_levels[record["risk_level"]] = risk_levels.get(record["risk_level"], 0) + 1
            if record.get("local_policy_name"):
                local_policies[record["local_policy_name"]] = local_policies.get(record["local_policy_name"], 0) + 1
            if record.get("referral_urgency"):
                referral_urgencies[record["referral_urgency"]] = referral_urgencies.get(record["referral_urgency"], 0) + 1
            if record.get("dynamic_action"):
                dynamic_actions[record["dynamic_action"]] = dynamic_actions.get(record["dynamic_action"], 0) + 1
            route = record.get("orchestration_route") or "legacy_or_unclassified"
            orchestration_routes[route] = orchestration_routes.get(route, 0) + 1
            if record.get("reduction_goal_active_driver"):
                driver = str(record["reduction_goal_active_driver"])
                reduction_goal_drivers[driver] = reduction_goal_drivers.get(driver, 0) + 1
            if record.get("reduction_goal_priority"):
                priority = str(record["reduction_goal_priority"])
                reduction_goal_priorities[priority] = reduction_goal_priorities.get(priority, 0) + 1
            if record.get("referral_explanation_level"):
                level = str(record["referral_explanation_level"])
                referral_explanation_levels[level] = referral_explanation_levels.get(level, 0) + 1
            if record.get("referral_explanation_channel"):
                channel = str(record["referral_explanation_channel"])
                referral_explanation_channels[channel] = referral_explanation_channels.get(channel, 0) + 1
            if record.get("adjustment_loop_action"):
                action = str(record["adjustment_loop_action"])
                adjustment_loop_actions[action] = adjustment_loop_actions.get(action, 0) + 1
            if record.get("adjustment_loop_priority"):
                priority = str(record["adjustment_loop_priority"])
                adjustment_loop_priorities[priority] = adjustment_loop_priorities.get(priority, 0) + 1
            if record.get("processing_route"):
                route = str(record["processing_route"])
                processing_routes[route] = processing_routes.get(route, 0) + 1
            if record.get("processing_safety_priority"):
                priority = str(record["processing_safety_priority"])
                processing_safety_priorities[priority] = processing_safety_priorities.get(priority, 0) + 1
            if record.get("processing_next_backend_action"):
                action = str(record["processing_next_backend_action"])
                processing_next_backend_actions[action] = processing_next_backend_actions.get(action, 0) + 1

        processing_timeline = _build_processing_timeline(records)

        return {
            "session_id": session_id,
            "total_responses": len(records),
            "latest_reply_text": latest.get("reply_text"),
            "latest_local_policy": latest.get("local_policy"),
            "latest_referral_decision": latest.get("referral_decision"),
            "latest_dynamic_adjustment": latest.get("dynamic_adjustment"),
            "latest_feedback_adaptation": latest.get("feedback_adaptation"),
            "latest_entropy_orchestration": latest.get("entropy_orchestration"),
            "latest_reduction_goal": latest.get("reduction_goal"),
            "latest_referral_explanation": latest.get("referral_explanation"),
            "latest_adjustment_loop": latest.get("adjustment_loop"),
            "latest_processing_summary": latest.get("processing_summary"),
            "strategy_layer_summary": _build_strategy_layer_summary(records),
            "next_adjustment_loop": asdict(next_adjustment_loop),
            "trend_warning": asdict(trend_warning),
            "session_care_plan": asdict(session_care_plan),
            "risk_levels": risk_levels,
            "local_policies": local_policies,
            "referral_urgencies": referral_urgencies,
            "dynamic_actions": dynamic_actions,
            "orchestration_routes": orchestration_routes,
            "reduction_goal_drivers": reduction_goal_drivers,
            "reduction_goal_priorities": reduction_goal_priorities,
            "referral_explanation_levels": referral_explanation_levels,
            "referral_explanation_channels": referral_explanation_channels,
            "adjustment_loop_actions": adjustment_loop_actions,
            "adjustment_loop_priorities": adjustment_loop_priorities,
            "processing_routes": processing_routes,
            "processing_safety_priorities": processing_safety_priorities,
            "processing_next_backend_actions": processing_next_backend_actions,
            "processing_timeline": processing_timeline,
            "processing_summary": _summarize_processing_timeline(processing_timeline),
            "processing_consistency": processing_consistency,
            "goal_attainment_timeline": goal_attainment_timeline,
            "goal_attainment_summary": goal_attainment_summary,
            "strategy_reselection": strategy_reselection,
            "latest_intervention_audit": intervention_audit_timeline[-1] if intervention_audit_timeline else None,
            "intervention_audit_timeline": intervention_audit_timeline,
            "intervention_audit_summary": intervention_audit_summary,
            "orchestration_timeline": _build_orchestration_timeline(records),
            "next_orchestration_recommendation": _build_next_orchestration_recommendation(
                latest,
                asdict(care_pathway),
                asdict(entropy_reduction_outcome),
            ),
            "referral_events": referral_events,
            "intervention_feedback": intervention_feedback,
            "human_interventions": human_interventions,
            "latest_human_intervention": human_interventions[-1] if human_interventions else None,
            "feedback_summary": feedback_summary,
            "session_insight": session_insight,
            "session_continuity": session_continuity,
            "conversation_memory": conversation_memory,
            "longitudinal_profile": asdict(longitudinal_profile),
            "care_pathway": asdict(care_pathway),
            "entropy_reduction_outcome": asdict(entropy_reduction_outcome),
            "reply_quality": reply_quality_report,
            "intervention_effectiveness": intervention_effectiveness,
            "intervention_next_step": intervention_next_step,
            "session_tracking": session_tracking,
            "strategy_version": strategy_version,
            "entropy_reduction_loop": entropy_reduction_loop,
        }

    def get_overview_stats(self, *, limit: int | None = 200) -> dict[str, Any]:
        records = self.list_support_responses(limit=limit)
        intervention_audit_timeline = build_intervention_audit_timeline(records, limit=len(records))
        intervention_audit_summary = summarize_intervention_audits(intervention_audit_timeline)
        processing_consistency = build_processing_consistency_report(records)
        latest_records = _latest_records_by_session(records)
        current_processing_consistency = build_processing_consistency_report(latest_records)
        risk_levels: dict[str, int] = {}
        local_policies: dict[str, int] = {}
        referral_urgencies: dict[str, int] = {}
        dynamic_actions: dict[str, int] = {}
        risk_routes: dict[str, int] = {}
        care_pathway_routes: dict[str, int] = {}
        care_pathway_priorities: dict[str, int] = {}
        entropy_outcome_statuses: dict[str, int] = {}
        orchestration_routes: dict[str, int] = {}
        reduction_goal_drivers: dict[str, int] = {}
        reduction_goal_priorities: dict[str, int] = {}
        referral_explanation_levels: dict[str, int] = {}
        referral_explanation_channels: dict[str, int] = {}
        adjustment_loop_actions: dict[str, int] = {}
        adjustment_loop_priorities: dict[str, int] = {}
        strategy_ids: dict[str, int] = {}
        primary_states: dict[str, int] = {}
        recommended_focuses: dict[str, int] = {}
        referred_count = 0
        manual_referral_count = 0
        feedback_summary = self.summarize_intervention_feedback()
        for record in records:
            if record.get("risk_level"):
                risk_levels[record["risk_level"]] = risk_levels.get(record["risk_level"], 0) + 1
            if record.get("local_policy_name"):
                local_policies[record["local_policy_name"]] = local_policies.get(record["local_policy_name"], 0) + 1
            if record.get("referral_urgency"):
                referral_urgencies[record["referral_urgency"]] = referral_urgencies.get(record["referral_urgency"], 0) + 1
            if record.get("dynamic_action"):
                dynamic_actions[record["dynamic_action"]] = dynamic_actions.get(record["dynamic_action"], 0) + 1
            if record.get("referral_should_refer"):
                referred_count += 1
            system_flags = (record.get("response") or {}).get("system_flags") or {}
            if system_flags.get("manual_referral_recommended"):
                manual_referral_count += 1
            route = _infer_record_route(record)
            risk_routes[route] = risk_routes.get(route, 0) + 1
            care_pathway = _infer_record_care_pathway(record)
            care_route = care_pathway["route"]
            care_priority = care_pathway["priority"]
            care_pathway_routes[care_route] = care_pathway_routes.get(care_route, 0) + 1
            care_pathway_priorities[care_priority] = care_pathway_priorities.get(care_priority, 0) + 1
            outcome_status = _infer_record_entropy_outcome_status(record)
            entropy_outcome_statuses[outcome_status] = entropy_outcome_statuses.get(outcome_status, 0) + 1
            orchestration_route = record.get("orchestration_route") or "legacy_or_unclassified"
            orchestration_routes[orchestration_route] = orchestration_routes.get(orchestration_route, 0) + 1
            if record.get("reduction_goal_active_driver"):
                driver = str(record["reduction_goal_active_driver"])
                reduction_goal_drivers[driver] = reduction_goal_drivers.get(driver, 0) + 1
            if record.get("reduction_goal_priority"):
                priority = str(record["reduction_goal_priority"])
                reduction_goal_priorities[priority] = reduction_goal_priorities.get(priority, 0) + 1
            if record.get("referral_explanation_level"):
                level = str(record["referral_explanation_level"])
                referral_explanation_levels[level] = referral_explanation_levels.get(level, 0) + 1
            if record.get("referral_explanation_channel"):
                channel = str(record["referral_explanation_channel"])
                referral_explanation_channels[channel] = referral_explanation_channels.get(channel, 0) + 1
            if record.get("adjustment_loop_action"):
                action = str(record["adjustment_loop_action"])
                adjustment_loop_actions[action] = adjustment_loop_actions.get(action, 0) + 1
            if record.get("adjustment_loop_priority"):
                priority = str(record["adjustment_loop_priority"])
                adjustment_loop_priorities[priority] = adjustment_loop_priorities.get(priority, 0) + 1
            if record.get("strategy_id"):
                strategy_id = str(record["strategy_id"])
                strategy_ids[strategy_id] = strategy_ids.get(strategy_id, 0) + 1
            if record.get("primary_state"):
                primary_state = str(record["primary_state"])
                primary_states[primary_state] = primary_states.get(primary_state, 0) + 1
            if record.get("recommended_focus"):
                focus = str(record["recommended_focus"])
                recommended_focuses[focus] = recommended_focuses.get(focus, 0) + 1

        current_care_pathway_routes: dict[str, int] = {}
        current_care_pathway_priorities: dict[str, int] = {}
        current_entropy_outcome_statuses: dict[str, int] = {}
        current_orchestration_routes: dict[str, int] = {}
        current_dialogue_stages: dict[str, int] = {}
        current_reduction_goal_drivers: dict[str, int] = {}
        current_referral_explanation_levels: dict[str, int] = {}
        current_adjustment_loop_actions: dict[str, int] = {}
        current_trend_warnings: list[dict[str, Any]] = []
        goal_attainment_timeline = build_goal_attainment_timeline(records)
        goal_attainment_summary = summarize_goal_attainment(goal_attainment_timeline)
        strategy_reselection_summary: dict[str, int] = {}
        for record in _latest_records_by_session(records):
            care_pathway = _infer_record_care_pathway(record)
            care_route = care_pathway["route"]
            care_priority = care_pathway["priority"]
            current_care_pathway_routes[care_route] = current_care_pathway_routes.get(care_route, 0) + 1
            current_care_pathway_priorities[care_priority] = current_care_pathway_priorities.get(care_priority, 0) + 1
            outcome_status = _infer_record_entropy_outcome_status(record)
            current_entropy_outcome_statuses[outcome_status] = current_entropy_outcome_statuses.get(outcome_status, 0) + 1
            orchestration_route = record.get("orchestration_route") or "legacy_or_unclassified"
            current_orchestration_routes[orchestration_route] = current_orchestration_routes.get(orchestration_route, 0) + 1
            stage = build_session_continuity_summary(
                session_id=str(record.get("session_id") or ""),
                records=[record],
            )["dialogue_stage"]
            current_dialogue_stages[stage] = current_dialogue_stages.get(stage, 0) + 1
            if record.get("reduction_goal_active_driver"):
                driver = str(record["reduction_goal_active_driver"])
                current_reduction_goal_drivers[driver] = current_reduction_goal_drivers.get(driver, 0) + 1
            if record.get("referral_explanation_level"):
                level = str(record["referral_explanation_level"])
                current_referral_explanation_levels[level] = current_referral_explanation_levels.get(level, 0) + 1
            if record.get("adjustment_loop_action"):
                action = str(record["adjustment_loop_action"])
                current_adjustment_loop_actions[action] = current_adjustment_loop_actions.get(action, 0) + 1
            session_records = [item for item in records if item.get("session_id") == record.get("session_id")]
            session_goal_timeline = build_goal_attainment_timeline(session_records)
            session_continuity = build_session_continuity_summary(
                session_id=str(record.get("session_id") or ""),
                records=session_records,
            )
            session_reselection = build_strategy_reselection_decision(
                goal_attainment_timeline=session_goal_timeline,
                session_continuity=session_continuity,
                latest_record=record,
            )
            trigger = str(session_reselection.get("trigger") or "none")
            strategy_reselection_summary[trigger] = strategy_reselection_summary.get(trigger, 0) + 1
            session_id = str(record.get("session_id") or "")
            session_records = [item for item in records if item.get("session_id") == record.get("session_id")]
            session_feedback = self.summarize_intervention_feedback(session_id) if session_id else {}
            session_audits = build_intervention_audit_timeline(session_records, limit=len(session_records))
            current_trend_warnings.append(
                asdict(
                    build_entropy_trend_warning(
                        session_id=session_id,
                        records=session_records,
                        entropy_trace=[],
                        feedback_summary=session_feedback,
                        audit_summary=summarize_intervention_audits(session_audits),
                    )
                )
            )

        return {
            "total_records": len(records),
            "total_sessions": len(latest_records),
            "referred_count": referred_count,
            "manual_referral_count": manual_referral_count,
            "risk_levels": risk_levels,
            "local_policies": local_policies,
            "referral_urgencies": referral_urgencies,
            "dynamic_actions": dynamic_actions,
            "risk_routes": risk_routes,
            "care_pathway_routes": care_pathway_routes,
            "care_pathway_priorities": care_pathway_priorities,
            "current_care_pathway_routes": current_care_pathway_routes,
            "current_care_pathway_priorities": current_care_pathway_priorities,
            "entropy_outcome_statuses": entropy_outcome_statuses,
            "current_entropy_outcome_statuses": current_entropy_outcome_statuses,
            "orchestration_routes": orchestration_routes,
            "current_orchestration_routes": current_orchestration_routes,
            "current_dialogue_stages": current_dialogue_stages,
            "reduction_goal_drivers": reduction_goal_drivers,
            "reduction_goal_priorities": reduction_goal_priorities,
            "current_reduction_goal_drivers": current_reduction_goal_drivers,
            "referral_explanation_levels": referral_explanation_levels,
            "referral_explanation_channels": referral_explanation_channels,
            "current_referral_explanation_levels": current_referral_explanation_levels,
            "adjustment_loop_actions": adjustment_loop_actions,
            "adjustment_loop_priorities": adjustment_loop_priorities,
            "current_adjustment_loop_actions": current_adjustment_loop_actions,
            "strategy_ids": strategy_ids,
            "primary_states": primary_states,
            "recommended_focuses": recommended_focuses,
            "strategy_layer_summary": _build_strategy_layer_overview(records),
            "trend_warning_summary": summarize_trend_warnings(current_trend_warnings),
            "current_trend_warnings": current_trend_warnings,
            "goal_attainment_summary": goal_attainment_summary,
            "strategy_reselection_summary": strategy_reselection_summary,
            "intervention_audit_summary": intervention_audit_summary,
            "processing_consistency_summary": processing_consistency["summary"],
            "current_processing_consistency_summary": current_processing_consistency["summary"],
            "processing_consistency_bad_cases": _build_processing_consistency_bad_cases(
                processing_consistency["timeline"],
                limit=10,
            ),
            "feedback_summary": feedback_summary,
        }

    def get_intervention_audits(self, session_id: str, *, limit: int | None = 50) -> dict[str, Any]:
        records = self.list_support_responses(session_id=session_id, limit=None)
        effective_limit = len(records) if limit is None else max(limit, 0)
        audit_logs = build_intervention_audit_timeline(records, limit=effective_limit)
        return {
            "session_id": session_id,
            "total_audits": len(audit_logs),
            "audit_summary": summarize_intervention_audits(audit_logs),
            "audit_logs": audit_logs,
        }

    def get_session_memory(self, session_id: str) -> dict[str, Any]:
        conversation_history = self.get_history(session_id)
        records = self.list_support_responses(session_id=session_id, limit=None)
        memory = build_memory_context(conversation_history)
        return {
            "session_id": session_id,
            "history_messages": len(conversation_history),
            "response_records": len(records),
            "conversation_memory": memory,
            "latest_topics": memory.get("active_topics") or [],
            "latest_boundaries": memory.get("user_boundaries") or [],
            "preferred_next_move": memory.get("preferred_next_move"),
            "avoid_next_reply": memory.get("avoid_next_reply") or [],
        }

    def get_session_trend_warning(self, session_id: str) -> dict[str, Any]:
        records = self.list_support_responses(session_id=session_id, limit=None)
        entropy_trace = self.get_entropy_trace(session_id)
        feedback_summary = self.summarize_intervention_feedback(session_id)
        audit_logs = build_intervention_audit_timeline(records, limit=len(records))
        audit_summary = summarize_intervention_audits(audit_logs)
        warning = build_entropy_trend_warning(
            session_id=session_id,
            records=records,
            entropy_trace=entropy_trace,
            feedback_summary=feedback_summary,
            audit_summary=audit_summary,
        )
        return {
            "session_id": session_id,
            "trend_warning": asdict(warning),
            "entropy_trace": entropy_trace,
            "audit_summary": audit_summary,
            "feedback_summary": feedback_summary,
        }

    def get_session_care_plan(self, session_id: str) -> dict[str, Any]:
        analysis = self.get_session_analysis(session_id)
        return {
            "session_id": session_id,
            "session_care_plan": analysis["session_care_plan"],
            "trend_warning": analysis["trend_warning"],
            "care_pathway": analysis["care_pathway"],
            "entropy_reduction_outcome": analysis["entropy_reduction_outcome"],
            "next_adjustment_loop": analysis["next_adjustment_loop"],
            "conversation_memory": analysis["conversation_memory"],
        }

    def get_session_intervention_effectiveness(self, session_id: str) -> dict[str, Any]:
        analysis = self.get_session_analysis(session_id)
        return {
            "session_id": session_id,
            "intervention_effectiveness": analysis["intervention_effectiveness"],
            "intervention_next_step": analysis["intervention_next_step"],
            "entropy_reduction_outcome": analysis["entropy_reduction_outcome"],
            "goal_attainment_summary": analysis["goal_attainment_summary"],
            "strategy_reselection": analysis["strategy_reselection"],
            "care_pathway": analysis["care_pathway"],
            "trend_warning": analysis["trend_warning"],
            "reply_quality_summary": analysis["reply_quality"]["summary"],
        }

    def get_session_intervention_next_step(self, session_id: str) -> dict[str, Any]:
        analysis = self.get_session_analysis(session_id)
        return {
            "session_id": session_id,
            "intervention_next_step": analysis["intervention_next_step"],
            "intervention_effectiveness_summary": analysis["intervention_effectiveness"]["summary"],
            "session_care_plan": analysis["session_care_plan"],
            "trend_warning": analysis["trend_warning"],
        }

    def get_session_tracking(self, session_id: str) -> dict[str, Any]:
        analysis = self.get_session_analysis(session_id)
        return {
            "session_id": session_id,
            "session_tracking": analysis["session_tracking"],
            "conversation_memory": analysis["conversation_memory"],
            "longitudinal_profile": analysis["longitudinal_profile"],
            "intervention_next_step": analysis["intervention_next_step"],
        }

    def get_session_strategy_version(self, session_id: str) -> dict[str, Any]:
        analysis = self.get_session_analysis(session_id)
        return {
            "session_id": session_id,
            "strategy_version": analysis["strategy_version"],
            "session_tracking": analysis["session_tracking"],
            "strategy_reselection": analysis["strategy_reselection"],
            "intervention_effectiveness_summary": analysis["intervention_effectiveness"]["summary"],
        }

    def get_session_strategy_layer(self, session_id: str) -> dict[str, Any]:
        analysis = self.get_session_analysis(session_id)
        return {
            "session_id": session_id,
            "total_responses": analysis["total_responses"],
            "strategy_layer_summary": analysis["strategy_layer_summary"],
            "latest_state_profile": (
                analysis["strategy_layer_summary"].get("latest")
                if isinstance(analysis.get("strategy_layer_summary"), dict)
                else None
            ),
            "strategy_reselection": analysis["strategy_reselection"],
            "next_orchestration_recommendation": analysis["next_orchestration_recommendation"],
            "care_pathway": analysis["care_pathway"],
            "trend_warning": analysis["trend_warning"],
        }

    def get_session_decision_trace(self, session_id: str, *, limit: int | None = 50) -> dict[str, Any]:
        records = self.list_support_responses(session_id=session_id, limit=None)
        trace = _build_decision_trace(records, limit=limit)
        latest = trace[-1] if trace else None
        return {
            "session_id": session_id,
            "total_turns": len(records),
            "returned_turns": len(trace),
            "latest_decision": latest,
            "decision_trace": trace,
            "summary": _summarize_decision_trace(trace),
        }

    def get_session_entropy_reduction_loop(self, session_id: str) -> dict[str, Any]:
        analysis = self.get_session_analysis(session_id)
        return {
            "session_id": session_id,
            "entropy_reduction_loop": analysis["entropy_reduction_loop"],
            "strategy_version": analysis["strategy_version"],
            "intervention_effectiveness_summary": analysis["intervention_effectiveness"]["summary"],
            "feedback_summary": analysis["feedback_summary"],
            "reply_quality_summary": analysis["reply_quality"]["summary"],
        }

    def get_session_reply_quality(self, session_id: str, *, limit: int | None = None) -> dict[str, Any]:
        records = self.list_support_responses(session_id=session_id, limit=limit)
        report = build_reply_quality_report(records)
        return {
            "session_id": session_id,
            "total_records": len(records),
            **report,
        }

    def get_reply_quality_overview(self, *, limit: int | None = 200) -> dict[str, Any]:
        records = self.list_support_responses(limit=limit)
        report = build_reply_quality_report(records)
        review_items = [item for item in report["timeline"] if item.get("needs_review")]
        return {
            "total_records": len(records),
            "summary": report["summary"],
            "recent_review_items": review_items[-25:],
        }

    def get_reply_quality_bad_cases(
        self,
        *,
        session_id: str | None = None,
        source_limit: int | None = 200,
        limit: int | None = 50,
        min_quality_score: int = 80,
    ) -> dict[str, Any]:
        records = self.list_support_responses(session_id=session_id, limit=source_limit)
        bad_cases = build_reply_quality_bad_cases(
            records,
            min_quality_score=min_quality_score,
            limit=limit,
        )
        issue_counts: dict[str, int] = {}
        for case in bad_cases:
            for issue in case.get("quality_review", {}).get("issues", []):
                issue = str(issue)
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        return {
            "session_id": session_id,
            "records_seen": len(records),
            "bad_case_count": len(bad_cases),
            "min_quality_score": min_quality_score,
            "issue_counts": dict(sorted(issue_counts.items(), key=lambda item: (-item[1], item[0]))),
            "bad_cases": bad_cases,
        }

    def get_intervention_effectiveness_overview(self, *, limit: int | None = 200) -> dict[str, Any]:
        records = self.list_support_responses(limit=limit)
        reports: list[dict[str, Any]] = []
        for latest in _latest_records_by_session(records):
            session_id = str(latest.get("session_id") or "")
            if not session_id:
                continue
            reports.append(self.get_session_analysis(session_id)["intervention_effectiveness"])
        overview = build_intervention_effectiveness_overview(reports)
        return {
            "source_records_seen": len(records),
            **overview,
        }

    def get_session_tracking_overview(self, *, limit: int | None = 200) -> dict[str, Any]:
        records = self.list_support_responses(limit=limit)
        latest_records = _latest_records_by_session(records)
        snapshots: list[dict[str, Any]] = []
        for latest in latest_records:
            session_id = str(latest.get("session_id") or "")
            if not session_id:
                continue
            snapshots.append(self.get_session_analysis(session_id)["session_tracking"])

        stage_counts: dict[str, int] = {}
        risk_track_counts: dict[str, int] = {}
        entropy_track_counts: dict[str, int] = {}
        privacy_boundary_counts: dict[str, int] = {}
        for snapshot in snapshots:
            stage = str(snapshot.get("stage") or "unknown")
            risk_track = str((snapshot.get("current_risk_track") or {}).get("recommended_care_level") or "unknown")
            entropy_track = str((snapshot.get("entropy_track") or {}).get("entropy_course") or "unknown")
            privacy = str(snapshot.get("privacy_boundary") or "none")
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
            risk_track_counts[risk_track] = risk_track_counts.get(risk_track, 0) + 1
            entropy_track_counts[entropy_track] = entropy_track_counts.get(entropy_track, 0) + 1
            privacy_boundary_counts[privacy] = privacy_boundary_counts.get(privacy, 0) + 1

        return {
            "source_records_seen": len(records),
            "total_sessions": len(snapshots),
            "stage_counts": stage_counts,
            "risk_track_counts": risk_track_counts,
            "entropy_track_counts": entropy_track_counts,
            "privacy_boundary_counts": privacy_boundary_counts,
            "recent_tracking_snapshots": snapshots[-25:],
        }

    def get_strategy_version_overview(self, *, limit: int | None = 200) -> dict[str, Any]:
        records = self.list_support_responses(limit=limit)
        latest_records = _latest_records_by_session(records)
        versions: list[dict[str, Any]] = []
        for latest in latest_records:
            session_id = str(latest.get("session_id") or "")
            if not session_id:
                continue
            versions.append(self.get_session_analysis(session_id)["strategy_version"])

        decision_counts: dict[str, int] = {}
        target_family_counts: dict[str, int] = {}
        switch_count = 0
        for version in versions:
            decision = str(version.get("decision") or "unknown")
            family = str(version.get("target_strategy_family") or "unknown")
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
            target_family_counts[family] = target_family_counts.get(family, 0) + 1
            if version.get("should_switch_strategy"):
                switch_count += 1

        return {
            "source_records_seen": len(records),
            "total_sessions": len(versions),
            "sessions_needing_strategy_switch": switch_count,
            "decision_counts": decision_counts,
            "target_family_counts": target_family_counts,
            "recent_strategy_versions": versions[-25:],
        }

    def get_strategy_layer_overview(self, *, limit: int | None = 200) -> dict[str, Any]:
        records = self.list_support_responses(limit=limit)
        overview = _build_strategy_layer_overview(records)
        return {
            "source_records_seen": len(records),
            "total_sessions": len(_latest_records_by_session(records)),
            "strategy_layer_summary": overview,
        }

    def get_decision_trace_overview(self, *, limit: int | None = 200) -> dict[str, Any]:
        records = self.list_support_responses(limit=limit)
        latest_records = _latest_records_by_session(records)
        trace = _build_decision_trace(latest_records, limit=None)
        return {
            "source_records_seen": len(records),
            "total_sessions": len(latest_records),
            "latest_session_decisions": trace,
            "summary": _summarize_decision_trace(trace),
        }

    def get_entropy_reduction_loop_overview(self, *, limit: int | None = 200) -> dict[str, Any]:
        records = self.list_support_responses(limit=limit)
        loops: list[dict[str, Any]] = []
        for latest in _latest_records_by_session(records):
            session_id = str(latest.get("session_id") or "")
            if not session_id:
                continue
            loops.append(self.get_session_analysis(session_id)["entropy_reduction_loop"])
        overview = build_entropy_reduction_loop_overview(loops)
        return {
            "source_records_seen": len(records),
            **overview,
        }

    def get_quality_refinement_plan(
        self,
        *,
        session_id: str | None = None,
        source_limit: int | None = 200,
        bad_case_limit: int | None = 100,
        min_quality_score: int = 80,
        max_examples_per_bucket: int = 8,
    ) -> dict[str, Any]:
        bad_case_payload = self.get_reply_quality_bad_cases(
            session_id=session_id,
            source_limit=source_limit,
            limit=bad_case_limit,
            min_quality_score=min_quality_score,
        )
        plan = build_quality_refinement_plan(
            bad_case_payload["bad_cases"],
            max_examples_per_bucket=max_examples_per_bucket,
        )
        return {
            "session_id": session_id,
            "source_records_seen": bad_case_payload["records_seen"],
            "min_quality_score": min_quality_score,
            **plan,
        }

    def get_care_queue(
        self,
        *,
        limit: int | None = 100,
        include_low_priority: bool = False,
        include_resolved: bool = False,
    ) -> dict[str, Any]:
        records = self.list_support_responses(limit=None)
        latest_records = _latest_records_by_session(records)
        latest_human_interventions = self.get_latest_human_interventions_by_session()
        items = [
            _build_care_queue_item(
                record,
                session_records=[item for item in records if item.get("session_id") == record.get("session_id")],
                feedback_summary=self.summarize_intervention_feedback(str(record.get("session_id") or "")),
                human_intervention=latest_human_interventions.get(str(record.get("session_id") or "")),
            )
            for record in latest_records
        ]
        if not include_low_priority:
            items = [item for item in items if item.priority != "low"]
        if not include_resolved:
            items = [
                item
                for item in items
                if (item.evidence.get("human_intervention") or {}).get("status") not in {"resolved", "closed"}
            ]
        items.sort(
            key=lambda item: (
                _QUEUE_PRIORITY_RANK.get(item.priority, 0),
                item.queue_score,
                item.created_at or "",
            ),
            reverse=True,
        )
        if limit is not None:
            items = items[:limit]

        priority_counts: dict[str, int] = {}
        route_counts: dict[str, int] = {}
        outcome_counts: dict[str, int] = {}
        trend_warning_counts: dict[str, int] = {}
        for item in items:
            priority_counts[item.priority] = priority_counts.get(item.priority, 0) + 1
            route_counts[item.route] = route_counts.get(item.route, 0) + 1
            outcome_counts[item.outcome_status] = outcome_counts.get(item.outcome_status, 0) + 1
            warning_level = item.trend_warning_level or "none"
            trend_warning_counts[warning_level] = trend_warning_counts.get(warning_level, 0) + 1

        return {
            "total_items": len(items),
            "include_low_priority": include_low_priority,
            "priority_counts": priority_counts,
            "route_counts": route_counts,
            "outcome_counts": outcome_counts,
            "trend_warning_counts": trend_warning_counts,
            "items": [asdict(item) for item in items],
        }


def _infer_record_route(record: dict[str, Any]) -> str:
    risk_level = record.get("risk_level") or "low"
    entropy_score = int(record.get("entropy_score") or 0)
    referral_should_refer = bool(record.get("referral_should_refer"))
    referral_urgency = record.get("referral_urgency") or "none"
    dynamic_action = record.get("dynamic_action") or ""
    system_flags = (record.get("response") or {}).get("system_flags") or {}

    if risk_level == "critical" or referral_urgency == "urgent":
        return "urgent_referral"
    if risk_level == "high" or dynamic_action == "human_followup_watch" or system_flags.get("manual_referral_recommended"):
        return "manual_followup"
    if referral_should_refer or dynamic_action == "escalate_support" or entropy_score >= 65:
        return "watch_closely"
    return "observe"


def _infer_record_care_pathway(record: dict[str, Any]) -> dict[str, str]:
    risk_level = record.get("risk_level") or "low"
    entropy_score = int(record.get("entropy_score") or 0)
    referral_should_refer = bool(record.get("referral_should_refer"))
    referral_urgency = record.get("referral_urgency") or "none"
    dynamic_action = record.get("dynamic_action") or ""
    feedback_mode = record.get("feedback_adaptation_mode") or ""
    system_flags = (record.get("response") or {}).get("system_flags") or {}

    if risk_level == "critical" or referral_urgency == "urgent":
        return {"route": "urgent_safety", "priority": "critical"}
    if (
        risk_level == "high"
        or dynamic_action == "human_followup_watch"
        or referral_should_refer
        or system_flags.get("manual_referral_recommended")
    ):
        return {"route": "human_followup_recommended", "priority": "high"}
    if feedback_mode == "repair_next_turn":
        return {"route": "repair_reply_style", "priority": "medium"}
    if dynamic_action in {"soften_and_stabilize", "escalate_support"} or entropy_score >= 65:
        return {"route": "monitor_next_turn", "priority": "medium"}
    return {"route": "continue_observation", "priority": "low"}


def _infer_record_entropy_outcome_status(record: dict[str, Any]) -> str:
    care_pathway = _infer_record_care_pathway(record)
    route = care_pathway["route"]
    trend = ((record.get("response") or {}).get("entropy") or {}).get("trend") or {}
    feedback_mode = record.get("feedback_adaptation_mode") or ""
    try:
        delta = int(trend.get("delta"))
    except (TypeError, ValueError):
        delta = None

    if route == "urgent_safety":
        return "crisis_priority"
    if feedback_mode == "repair_next_turn" or route == "repair_reply_style":
        return "needs_strategy_repair"
    if route == "human_followup_recommended":
        return "needs_human_followup"
    if delta is not None and delta >= 10:
        return "deteriorating"
    if delta is not None and delta <= -8:
        return "improving"
    if route == "monitor_next_turn":
        return "watching"
    return "stable_observe"


def _build_orchestration_timeline(records: list[dict[str, Any]], *, limit: int = 12) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    for record in records[-limit:]:
        orchestration = record.get("entropy_orchestration") or {}
        timeline.append(
            {
                "response_id": record.get("response_id"),
                "created_at": record.get("created_at"),
                "input_text": record.get("input_text"),
                "risk_level": record.get("risk_level"),
                "entropy_score": record.get("entropy_score"),
                "dynamic_action": record.get("dynamic_action"),
                "feedback_adaptation_mode": record.get("feedback_adaptation_mode"),
                "route": record.get("orchestration_route") or "legacy_or_unclassified",
                "next_focus": record.get("orchestration_next_focus"),
                "user_visible_goal": record.get("orchestration_user_visible_goal"),
                "constraints": orchestration.get("constraints") or [],
                "risk_control": orchestration.get("risk_control"),
            }
        )
    return timeline


def _build_next_orchestration_recommendation(
    latest: dict[str, Any],
    care_pathway: dict[str, Any],
    entropy_reduction_outcome: dict[str, Any],
) -> dict[str, Any]:
    orchestration = latest.get("entropy_orchestration") or {}
    route = orchestration.get("route") or latest.get("orchestration_route") or "legacy_or_unclassified"
    outcome_status = entropy_reduction_outcome.get("status")
    care_route = care_pathway.get("route")

    action_by_route = {
        "safety_first": "activate_safety_protocol",
        "human_support_linkage": "recommend_human_followup",
        "repair_conversation": "repair_next_reply_style",
        "boundary_respecting_support": "respect_boundary_and_offer_low_pressure_support",
        "stabilize_and_reduce_load": "stabilize_then_reduce_one_small_load",
        "explore_and_clarify": "ask_one_contextual_question",
        "maintain_support": "continue_current_support_style",
    }
    rationale_by_route = {
        "safety_first": "最新回合出现高风险或危机线索，下一轮优先安全确认和现实支持。",
        "human_support_linkage": "系统判断需要引入现实中的支持资源，下一轮应温和说明可选求助路径。",
        "repair_conversation": "上一轮可能没有接住用户，下一轮先承认偏差，再降低提问压力。",
        "boundary_respecting_support": "用户表达不想细说或担心隐私，下一轮应先保证边界和保密感。",
        "stabilize_and_reduce_load": "用户负荷较高，下一轮先稳定情绪，再给一个很小的可执行步骤。",
        "explore_and_clarify": "信息不足但风险未升高，下一轮只问一个贴近上下文的问题。",
        "maintain_support": "当前状态相对稳定，下一轮维持自然陪伴，不额外增加专业术语。",
        "legacy_or_unclassified": "历史记录缺少调控中枢字段，下一轮按最新风险和照护路径重新判断。",
    }

    action = action_by_route.get(route)
    rationale = rationale_by_route.get(route, rationale_by_route["legacy_or_unclassified"])
    if not action:
        if care_route == "urgent_safety" or outcome_status == "crisis_priority":
            action = "activate_safety_protocol"
        elif care_route == "human_followup_recommended" or outcome_status == "needs_human_followup":
            action = "recommend_human_followup"
        elif care_route == "repair_reply_style" or outcome_status == "needs_strategy_repair":
            action = "repair_next_reply_style"
        else:
            action = "continue_observation"

    return {
        "route": route,
        "recommended_action": action,
        "next_focus": orchestration.get("next_focus") or latest.get("recommended_focus"),
        "user_visible_goal": orchestration.get("user_visible_goal"),
        "care_pathway_route": care_route,
        "outcome_status": outcome_status,
        "rationale": rationale,
    }


def _build_strategy_layer_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize hidden state/strategy routing for developer-facing analysis."""

    if not records:
        return {
            "latest": None,
            "primary_states": {},
            "recommended_focuses": {},
            "strategy_ids": {},
            "strategy_priorities": {},
            "response_modes": {},
            "timeline": [],
            "next_backend_focus": "collect_more_context",
            "should_consider_human_followup": False,
        }

    latest = records[-1]
    timeline = [
        {
            "response_id": record.get("response_id"),
            "created_at": record.get("created_at"),
            "input_text": record.get("input_text"),
            "entropy_score": record.get("entropy_score"),
            "risk_level": record.get("risk_level"),
            "primary_state": record.get("primary_state"),
            "recommended_focus": record.get("recommended_focus"),
            "strategy_id": record.get("strategy_id"),
            "strategy_priority": record.get("strategy_priority"),
            "response_mode": record.get("response_mode"),
            "dynamic_action": record.get("dynamic_action"),
            "referral_urgency": record.get("referral_urgency"),
        }
        for record in records[-12:]
    ]
    should_followup = any(
        bool(record.get("referral_should_refer"))
        or str(record.get("strategy_priority") or "") in {"high", "urgent"}
        or str(record.get("dynamic_action") or "") in {"escalate_support", "human_followup_watch"}
        for record in records[-3:]
    )
    next_backend_focus = _infer_next_strategy_focus(latest, should_followup=should_followup)
    return {
        "latest": {
            "response_id": latest.get("response_id"),
            "primary_state": latest.get("primary_state"),
            "state_intensity": latest.get("state_intensity"),
            "recommended_focus": latest.get("recommended_focus"),
            "strategy_id": latest.get("strategy_id"),
            "strategy_priority": latest.get("strategy_priority"),
            "response_mode": latest.get("response_mode"),
            "dynamic_action": latest.get("dynamic_action"),
            "referral_urgency": latest.get("referral_urgency"),
        },
        "primary_states": _count_record_values(records, "primary_state"),
        "recommended_focuses": _count_record_values(records, "recommended_focus"),
        "strategy_ids": _count_record_values(records, "strategy_id"),
        "strategy_priorities": _count_record_values(records, "strategy_priority"),
        "response_modes": _count_record_values(records, "response_mode"),
        "timeline": timeline,
        "next_backend_focus": next_backend_focus,
        "should_consider_human_followup": should_followup,
    }


def _build_strategy_layer_overview(records: list[dict[str, Any]]) -> dict[str, Any]:
    latest_records = _latest_records_by_session(records)
    current_summary = _build_strategy_layer_summary(latest_records)
    all_summary = _build_strategy_layer_summary(records)
    return {
        "current_sessions": {
            "primary_states": current_summary["primary_states"],
            "recommended_focuses": current_summary["recommended_focuses"],
            "strategy_ids": current_summary["strategy_ids"],
            "strategy_priorities": current_summary["strategy_priorities"],
            "response_modes": current_summary["response_modes"],
            "human_followup_sessions": sum(
                1 for record in latest_records if _build_strategy_layer_summary([record])["should_consider_human_followup"]
            ),
        },
        "all_records": {
            "primary_states": all_summary["primary_states"],
            "recommended_focuses": all_summary["recommended_focuses"],
            "strategy_ids": all_summary["strategy_ids"],
            "strategy_priorities": all_summary["strategy_priorities"],
            "response_modes": all_summary["response_modes"],
        },
        "recent_strategy_timeline": all_summary["timeline"],
    }


def _build_decision_trace(records: list[dict[str, Any]], *, limit: int | None = 50) -> list[dict[str, Any]]:
    selected_records = records if limit is None else records[-max(limit, 0) :]
    trace: list[dict[str, Any]] = []
    for index, record in enumerate(selected_records, start=max(len(records) - len(selected_records) + 1, 1)):
        response = record.get("response") or {}
        entropy = response.get("entropy") or {}
        entropy_trend = entropy.get("trend") or {}
        care_pathway = _infer_record_care_pathway(record)
        outcome_status = _infer_record_entropy_outcome_status(record)
        system_flags = response.get("system_flags") or {}
        trace.append(
            {
                "turn_index": index,
                "response_id": record.get("response_id"),
                "created_at": record.get("created_at"),
                "input_text": record.get("input_text"),
                "reply_preview": _preview_text(str(record.get("reply_text") or "")),
                "risk": {
                    "level": record.get("risk_level"),
                    "score": record.get("risk_score"),
                    "route": _infer_record_route(record),
                    "referral_urgency": record.get("referral_urgency"),
                    "should_refer": bool(record.get("referral_should_refer")),
                    "manual_referral_recommended": bool(system_flags.get("manual_referral_recommended")),
                },
                "entropy": {
                    "score": record.get("entropy_score"),
                    "level": record.get("entropy_level"),
                    "balance_state": record.get("balance_state"),
                    "trend_delta": entropy_trend.get("delta"),
                    "trend_direction": entropy_trend.get("direction"),
                },
                "state": {
                    "primary_state": record.get("primary_state"),
                    "intensity": record.get("state_intensity"),
                    "recommended_focus": record.get("recommended_focus"),
                },
                "strategy": {
                    "strategy_id": record.get("strategy_id"),
                    "priority": record.get("strategy_priority"),
                    "response_mode": record.get("response_mode"),
                    "next_backend_focus": _infer_next_strategy_focus(
                        record,
                        should_followup=bool(record.get("referral_should_refer"))
                        or str(record.get("strategy_priority") or "") in {"high", "urgent"},
                    ),
                },
                "dynamic_adjustment": {
                    "stability_state": record.get("dynamic_stability_state"),
                    "action": record.get("dynamic_action"),
                    "intensity_shift": record.get("dynamic_intensity_shift"),
                    "should_refer": bool(record.get("dynamic_should_refer")),
                },
                "orchestration": {
                    "route": record.get("orchestration_route") or "legacy_or_unclassified",
                    "next_focus": record.get("orchestration_next_focus"),
                    "user_visible_goal": record.get("orchestration_user_visible_goal"),
                },
                "care": {
                    "pathway_route": care_pathway["route"],
                    "pathway_priority": care_pathway["priority"],
                    "outcome_status": outcome_status,
                },
                "feedback": {
                    "adaptation_mode": record.get("feedback_adaptation_mode"),
                    "question_pressure": record.get("feedback_question_pressure"),
                },
            }
        )
    return trace


def _summarize_decision_trace(trace: list[dict[str, Any]]) -> dict[str, Any]:
    if not trace:
        return {
            "turns": 0,
            "latest_entropy_score": None,
            "latest_risk_level": None,
            "latest_strategy_id": None,
            "risk_levels": {},
            "strategy_ids": {},
            "care_routes": {},
            "needs_attention": False,
            "attention_reasons": [],
        }
    latest = trace[-1]
    risk_levels = _count_trace_values(trace, ("risk", "level"))
    strategy_ids = _count_trace_values(trace, ("strategy", "strategy_id"))
    care_routes = _count_trace_values(trace, ("care", "pathway_route"))
    attention_reasons: list[str] = []
    if str((latest.get("risk") or {}).get("level") or "") in {"high", "critical"}:
        attention_reasons.append("latest_high_risk")
    if (latest.get("risk") or {}).get("should_refer") or (latest.get("risk") or {}).get("manual_referral_recommended"):
        attention_reasons.append("latest_referral_recommended")
    if str((latest.get("care") or {}).get("pathway_priority") or "") in {"high", "critical"}:
        attention_reasons.append("latest_high_care_priority")
    if str((latest.get("dynamic_adjustment") or {}).get("action") or "") in {"escalate_support", "human_followup_watch"}:
        attention_reasons.append("latest_dynamic_escalation")
    return {
        "turns": len(trace),
        "latest_entropy_score": (latest.get("entropy") or {}).get("score"),
        "latest_risk_level": (latest.get("risk") or {}).get("level"),
        "latest_strategy_id": (latest.get("strategy") or {}).get("strategy_id"),
        "latest_primary_state": (latest.get("state") or {}).get("primary_state"),
        "risk_levels": risk_levels,
        "strategy_ids": strategy_ids,
        "care_routes": care_routes,
        "needs_attention": bool(attention_reasons),
        "attention_reasons": attention_reasons,
    }


def _build_processing_timeline(records: list[dict[str, Any]], *, limit: int | None = 20) -> list[dict[str, Any]]:
    selected_records = records if limit is None else records[-max(limit, 0) :]
    timeline: list[dict[str, Any]] = []
    for index, record in enumerate(selected_records, start=max(len(records) - len(selected_records) + 1, 1)):
        summary = record.get("processing_summary") or {}
        if not summary:
            timeline.append(
                {
                    "turn_index": index,
                    "response_id": record.get("response_id"),
                    "created_at": record.get("created_at"),
                    "route": "legacy_or_missing",
                    "safety_priority": None,
                    "next_backend_action": None,
                    "completed_stages": [],
                    "decision_reasons": [],
                }
            )
            continue
        timeline.append(
            {
                "turn_index": index,
                "response_id": record.get("response_id"),
                "created_at": record.get("created_at"),
                "route": summary.get("route"),
                "input_mode": summary.get("input_mode"),
                "reply_source": summary.get("reply_source"),
                "safety_priority": summary.get("safety_priority"),
                "risk_level": summary.get("risk_level"),
                "entropy_score": summary.get("entropy_score"),
                "balance_state": summary.get("balance_state"),
                "primary_state": summary.get("primary_state"),
                "strategy_id": summary.get("strategy_id"),
                "dynamic_action": summary.get("dynamic_action"),
                "orchestration_route": summary.get("orchestration_route"),
                "referral_urgency": summary.get("referral_urgency"),
                "should_refer": bool(summary.get("should_refer")),
                "local_policy_name": summary.get("local_policy_name"),
                "next_backend_action": summary.get("next_backend_action"),
                "completed_stages": summary.get("completed_stages") or [],
                "decision_reasons": summary.get("decision_reasons") or [],
            }
        )
    return timeline


def _summarize_processing_timeline(timeline: list[dict[str, Any]]) -> dict[str, Any]:
    if not timeline:
        return {
            "turns": 0,
            "routes": {},
            "safety_priorities": {},
            "next_backend_actions": {},
            "latest_route": None,
            "latest_safety_priority": None,
            "latest_next_backend_action": None,
            "needs_human_attention": False,
            "attention_reasons": [],
        }
    routes = _count_top_level_values(timeline, "route")
    safety_priorities = _count_top_level_values(timeline, "safety_priority")
    next_actions = _count_top_level_values(timeline, "next_backend_action")
    latest = timeline[-1]
    attention_reasons: list[str] = []
    if latest.get("safety_priority") in {"urgent", "human_followup"}:
        attention_reasons.append(f"safety_priority:{latest.get('safety_priority')}")
    if latest.get("next_backend_action") in {"activate_urgent_handoff", "queue_human_followup"}:
        attention_reasons.append(f"next_backend_action:{latest.get('next_backend_action')}")
    if latest.get("should_refer"):
        attention_reasons.append("latest_should_refer")
    return {
        "turns": len(timeline),
        "routes": routes,
        "safety_priorities": safety_priorities,
        "next_backend_actions": next_actions,
        "latest_route": latest.get("route"),
        "latest_safety_priority": latest.get("safety_priority"),
        "latest_next_backend_action": latest.get("next_backend_action"),
        "needs_human_attention": bool(attention_reasons),
        "attention_reasons": attention_reasons,
    }


def _count_trace_values(trace: list[dict[str, Any]], path: tuple[str, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    outer, inner = path
    for item in trace:
        nested = item.get(outer) or {}
        value = nested.get(inner) if isinstance(nested, dict) else None
        if value is None or value == "":
            continue
        clean = str(value)
        counts[clean] = counts.get(clean, 0) + 1
    return counts


def _count_top_level_values(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = item.get(key)
        if value is None or value == "":
            continue
        clean = str(value)
        counts[clean] = counts.get(clean, 0) + 1
    return counts


def _build_processing_consistency_bad_cases(
    timeline: list[dict[str, Any]],
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    bad_cases = [item for item in timeline if item.get("issues")]
    return [
        {
            "turn_index": item.get("turn_index"),
            "response_id": item.get("response_id"),
            "created_at": item.get("created_at"),
            "risk_level": item.get("risk_level"),
            "route": item.get("route"),
            "safety_priority": item.get("safety_priority"),
            "next_backend_action": item.get("next_backend_action"),
            "issues": item.get("issues") or [],
        }
        for item in bad_cases[: max(limit, 0)]
    ]


def _preview_text(text: str, *, max_chars: int = 80) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean
    return f"{clean[:max_chars].rstrip()}..."


def _infer_next_strategy_focus(latest: dict[str, Any], *, should_followup: bool) -> str:
    if should_followup:
        return "human_followup_or_safety_monitoring"
    dynamic_action = str(latest.get("dynamic_action") or "")
    strategy_id = str(latest.get("strategy_id") or "")
    recommended_focus = str(latest.get("recommended_focus") or "")
    if dynamic_action in {"repair_reply_style", "switch_strategy"}:
        return "repair_next_reply_style"
    if strategy_id in {"privacy_reassurance", "low_pressure_presence"}:
        return "maintain_trust_and_low_pressure"
    if recommended_focus in {"sleep_stabilization_first", "grounding_then_small_next_step"}:
        return "stabilize_then_one_small_action"
    if recommended_focus in {
        "contribution_visibility",
        "performance_grounding",
        "escape_loop_interruption",
        "family_boundary_sustainability",
        "grief_without_self_blame",
        "safety_reporting_without_blame",
    }:
        return "continue_scene_specific_strategy"
    return "continue_supportive_listening"


def _count_record_values(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = record.get(field)
        if value is None or value == "":
            continue
        clean = str(value)
        counts[clean] = counts.get(clean, 0) + 1
    return counts


_QUEUE_PRIORITY_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}


def _build_care_queue_item(
    record: dict[str, Any],
    *,
    session_records: list[dict[str, Any]] | None = None,
    feedback_summary: dict[str, Any] | None = None,
    human_intervention: dict[str, Any] | None = None,
) -> CareQueueItem:
    care_pathway = _infer_record_care_pathway(record)
    outcome_status = _infer_record_entropy_outcome_status(record)
    route = care_pathway["route"]
    priority = care_pathway["priority"]
    session_records = session_records or [record]
    audit_logs = build_intervention_audit_timeline(session_records, limit=len(session_records))
    trend_warning = build_entropy_trend_warning(
        session_id=str(record.get("session_id") or ""),
        records=session_records,
        entropy_trace=[],
        feedback_summary=feedback_summary or {},
        audit_summary=summarize_intervention_audits(audit_logs),
    )
    priority, route = _merge_queue_priority_with_warning(priority, route, trend_warning.level)
    recommended_action = _recommended_queue_action(route, outcome_status, trend_warning.recommended_action)
    queue_score = _queue_score(
        priority=priority,
        route=route,
        outcome_status=outcome_status,
        trend_warning_level=trend_warning.level,
        latest_entropy_score=_safe_int(record.get("entropy_score")),
    )
    return CareQueueItem(
        session_id=str(record.get("session_id") or ""),
        priority=priority,
        route=route,
        outcome_status=outcome_status,
        recommended_action=recommended_action,
        latest_entropy_score=_safe_int(record.get("entropy_score")),
        risk_level=record.get("risk_level"),
        created_at=record.get("created_at"),
        response_id=record.get("response_id"),
        trend_warning_level=trend_warning.level,
        trend_state=trend_warning.trend_state,
        review_window_hours=trend_warning.review_window_hours,
        queue_score=queue_score,
        evidence={
            "risk_score": record.get("risk_score"),
            "dynamic_action": record.get("dynamic_action"),
            "feedback_adaptation_mode": record.get("feedback_adaptation_mode"),
            "referral_urgency": record.get("referral_urgency"),
            "referral_should_refer": bool(record.get("referral_should_refer")),
            "primary_state": record.get("primary_state"),
            "trend_warning_reasons": trend_warning.trigger_reasons,
            "trend_warning_evidence": trend_warning.evidence,
            "human_intervention": human_intervention,
        },
    )


def _merge_queue_priority_with_warning(priority: str, route: str, warning_level: str) -> tuple[str, str]:
    if warning_level == "critical":
        return "critical", "urgent_safety"
    if warning_level == "high" and priority not in {"critical", "high"}:
        return "high", "human_followup_recommended"
    if warning_level == "medium" and priority == "low":
        return "medium", "monitor_next_turn"
    return priority, route


def _recommended_queue_action(route: str, outcome_status: str, warning_action: str | None = None) -> str:
    if route == "urgent_safety" or outcome_status == "crisis_priority":
        return "activate_safety_protocol"
    if route == "human_followup_recommended" or outcome_status == "needs_human_followup":
        return "recommend_human_followup"
    if route == "repair_reply_style" or outcome_status == "needs_strategy_repair":
        return "repair_next_reply_style"
    if warning_action and warning_action not in {"continue_observation"}:
        return warning_action
    if route == "monitor_next_turn" or outcome_status in {"watching", "deteriorating"}:
        return "monitor_next_turn_entropy"
    if outcome_status == "improving":
        return "maintain_working_strategy"
    return "continue_observation"


def _queue_score(
    *,
    priority: str,
    route: str,
    outcome_status: str,
    trend_warning_level: str,
    latest_entropy_score: int | None,
) -> int:
    score = _QUEUE_PRIORITY_RANK.get(priority, 0) * 100
    score += {"critical": 80, "high": 60, "medium": 35, "low": 10, "none": 0}.get(trend_warning_level, 0)
    score += {
        "urgent_safety": 60,
        "human_followup_recommended": 45,
        "monitor_next_turn": 25,
        "repair_reply_style": 20,
        "continue_observation": 0,
    }.get(route, 0)
    score += {
        "crisis_priority": 60,
        "needs_human_followup": 45,
        "deteriorating": 35,
        "watching": 20,
        "needs_strategy_repair": 18,
        "stable_observe": 0,
        "improving": -10,
    }.get(outcome_status, 0)
    if latest_entropy_score is not None:
        score += min(max(latest_entropy_score, 0), 100) // 5
    return score


def _human_intervention_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "session_id": row["session_id"],
        "response_id": row["response_id"],
        "status": row["status"],
        "handler_id": row["handler_id"],
        "note": row["note"],
        "next_action": row["next_action"],
        "tags": json.loads(row["tags_json"]),
        "created_at": row["created_at"],
    }


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _latest_records_by_session(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        session_id = str(record.get("session_id") or f"anonymous:{index}")
        latest[session_id] = record
    return list(latest.values())
