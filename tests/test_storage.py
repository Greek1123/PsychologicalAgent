from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.storage import SQLiteSessionStore


class SQLiteSessionStoreTests(unittest.TestCase):
    def test_store_persists_history_and_entropy_across_instances(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "agent.db"

            store = SQLiteSessionStore(str(db_path), max_messages=6)
            store.append_exchange("session-a", user_text="我很焦虑", assistant_text="先呼吸")
            store.append_entropy_snapshot(
                "session-a",
                response_id="r1",
                score=58,
                level=3,
                balance_state="strained",
                dominant_drivers=["认知负荷(考试)"],
            )

            reloaded = SQLiteSessionStore(str(db_path), max_messages=6)
            history = reloaded.get_history("session-a")
            trace = reloaded.get_entropy_trace("session-a")

            self.assertEqual(len(history), 2)
            self.assertEqual(history[0]["role"], "user")
            self.assertEqual(history[1]["content"], "先呼吸")
            self.assertEqual(len(trace), 1)
            self.assertEqual(trace[0]["score"], 58)
            self.assertEqual(reloaded.get_last_entropy("session-a")["balance_state"], "strained")

    def test_clear_removes_persisted_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "agent.db"

            store = SQLiteSessionStore(str(db_path), max_messages=6)
            store.append_exchange("session-b", user_text="u1", assistant_text="a1")
            store.append_entropy_snapshot(
                "session-b",
                response_id="r1",
                score=40,
                level=2,
                balance_state="strained",
                dominant_drivers=["情绪强度(焦虑)"],
            )
            store.clear("session-b")

            self.assertEqual(store.get_history("session-b"), [])
            self.assertEqual(store.get_entropy_trace("session-b"), [])
            self.assertIsNone(store.get_last_entropy("session-b"))

    def test_store_support_response_can_be_loaded_for_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "agent.db"

            store = SQLiteSessionStore(str(db_path), max_messages=6)
            store.store_support_response(
                session_id="session-c",
                response_id="resp-c",
                source="text",
                input_text="最近很烦。",
                transcript=None,
                student_context={"grade": "大一"},
                conversation_history=[{"role": "assistant", "content": "之前我们讨论过睡眠。"}],
                response_payload={"risk": {"level": "medium"}, "plan": {"summary": "test"}},
            )

            rows = store.list_support_responses(session_id="session-c")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["response_id"], "resp-c")
            self.assertEqual(rows[0]["student_context"]["grade"], "大一")


if __name__ == "__main__":
    unittest.main()
