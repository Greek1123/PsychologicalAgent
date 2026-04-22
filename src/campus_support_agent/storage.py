from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Any

from .logging_utils import get_logger


logger = get_logger("storage")


def _flatten_response_summary(response: dict[str, Any]) -> dict[str, Any]:
    risk = response.get("risk") or {}
    entropy = response.get("entropy") or {}
    local_policy = response.get("local_policy") or {}
    referral_decision = response.get("referral_decision") or {}
    return {
        "reply_text": response.get("reply_text"),
        "risk_level": risk.get("level"),
        "risk_score": risk.get("score"),
        "entropy_score": entropy.get("score"),
        "entropy_level": entropy.get("level"),
        "balance_state": entropy.get("balance_state"),
        "local_policy": local_policy,
        "local_policy_name": local_policy.get("policy_name"),
        "local_policy_stage": local_policy.get("policy_stage"),
        "local_policy_escalation_hint": local_policy.get("escalation_hint"),
        "referral_decision": referral_decision,
        "referral_should_refer": referral_decision.get("should_refer"),
        "referral_urgency": referral_decision.get("urgency"),
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
        logger.info("Cleared persisted session data for session_id=%s", session_id)

    def get_session_analysis(self, session_id: str) -> dict[str, Any]:
        records = self.list_support_responses(session_id=session_id)
        if not records:
            return {
                "session_id": session_id,
                "total_responses": 0,
                "latest_reply_text": None,
                "latest_local_policy": None,
                "latest_referral_decision": None,
                "risk_levels": {},
                "local_policies": {},
                "referral_urgencies": {},
            }

        risk_levels: dict[str, int] = {}
        local_policies: dict[str, int] = {}
        referral_urgencies: dict[str, int] = {}
        for record in records:
            if record.get("risk_level"):
                risk_levels[record["risk_level"]] = risk_levels.get(record["risk_level"], 0) + 1
            if record.get("local_policy_name"):
                local_policies[record["local_policy_name"]] = local_policies.get(record["local_policy_name"], 0) + 1
            if record.get("referral_urgency"):
                referral_urgencies[record["referral_urgency"]] = referral_urgencies.get(record["referral_urgency"], 0) + 1

        latest = records[-1]
        return {
            "session_id": session_id,
            "total_responses": len(records),
            "latest_reply_text": latest.get("reply_text"),
            "latest_local_policy": latest.get("local_policy"),
            "latest_referral_decision": latest.get("referral_decision"),
            "risk_levels": risk_levels,
            "local_policies": local_policies,
            "referral_urgencies": referral_urgencies,
        }

    def get_overview_stats(self, *, limit: int | None = 200) -> dict[str, Any]:
        records = self.list_support_responses(limit=limit)
        risk_levels: dict[str, int] = {}
        local_policies: dict[str, int] = {}
        referral_urgencies: dict[str, int] = {}
        referred_count = 0
        for record in records:
            if record.get("risk_level"):
                risk_levels[record["risk_level"]] = risk_levels.get(record["risk_level"], 0) + 1
            if record.get("local_policy_name"):
                local_policies[record["local_policy_name"]] = local_policies.get(record["local_policy_name"], 0) + 1
            if record.get("referral_urgency"):
                referral_urgencies[record["referral_urgency"]] = referral_urgencies.get(record["referral_urgency"], 0) + 1
            if record.get("referral_should_refer"):
                referred_count += 1

        return {
            "total_records": len(records),
            "referred_count": referred_count,
            "risk_levels": risk_levels,
            "local_policies": local_policies,
            "referral_urgencies": referral_urgencies,
        }
