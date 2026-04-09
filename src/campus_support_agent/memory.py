from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import Any


class InMemorySessionStore:
    # 这个类保留给单元测试和临时实验使用；正式运行默认走 SQLite 持久化存储。
    def __init__(self, max_messages: int = 12) -> None:
        self.max_messages = max_messages
        self._sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._entropy_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._lock = Lock()

    def get_history(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def get_entropy_trace(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._entropy_trace.get(session_id, []))

    def get_last_entropy(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            trace = self._entropy_trace.get(session_id, [])
            return dict(trace[-1]) if trace else None

    def append_exchange(self, session_id: str, *, user_text: str, assistant_text: str) -> int:
        with self._lock:
            history = self._sessions[session_id]
            history.extend(
                [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ]
            )
            if len(history) > self.max_messages:
                self._sessions[session_id] = history[-self.max_messages :]
            return len(self._sessions[session_id])

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
        with self._lock:
            trace = self._entropy_trace[session_id]
            trace.append(
                {
                    "response_id": response_id,
                    "score": score,
                    "level": level,
                    "balance_state": balance_state,
                    "dominant_drivers": dominant_drivers,
                }
            )
            if len(trace) > self.max_messages:
                self._entropy_trace[session_id] = trace[-self.max_messages :]
            return len(self._entropy_trace[session_id])

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            self._entropy_trace.pop(session_id, None)
