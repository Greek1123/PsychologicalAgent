from __future__ import annotations

import json
import shutil
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.general_dialog_builder import build_general_multiturn_dataset


@contextmanager
def _temporary_workspace_dir() -> Path:
    parent = ROOT / "tmp_test_artifacts"
    parent.mkdir(parents=True, exist_ok=True)
    tmpdir = parent / f"general-dialog-{uuid4().hex}"
    tmpdir.mkdir(parents=True, exist_ok=True)
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


class GeneralDialogBuilderTests(unittest.TestCase):
    def test_build_general_multiturn_dataset_alternates_roles(self) -> None:
        with _temporary_workspace_dir() as root:
            input_path = root / "dialog_release.json"
            output_path = root / "general_phase0_train_ms_swift.jsonl"

            payload = [
                {
                    "dialog_id": "1_1",
                    "document_id": 1,
                    "content": [
                        "你好",
                        "你好呀",
                        "最近怎么样",
                        "还行，你呢",
                        "我也还不错",
                        "那挺好",
                        "回头再聊",
                        "好，拜拜",
                        "再见",
                        "再见啦",
                    ],
                }
            ]
            input_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            stats = build_general_multiturn_dataset(
                str(input_path),
                str(output_path),
                min_turns=6,
                max_turns=8,
                limit=10,
                seed=7,
            )

            self.assertEqual(stats["written"], 1)
            record = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["task_type"], "general_multiturn_dialogue")
            self.assertEqual(len(record["messages"]), 8)
            self.assertEqual(record["messages"][0]["role"], "user")
            self.assertEqual(record["messages"][1]["role"], "assistant")

    def test_build_general_multiturn_dataset_parses_instruction_records(self) -> None:
        with _temporary_workspace_dir() as root:
            input_path = root / "multiturn_chat.jsonl"
            output_path = root / "general_phase0_train_ms_swift.jsonl"

            lines = [
                {
                    "instruction": (
                        "Human:你好，你能陪我聊聊吗？\n"
                        "Assistant:当然可以，你现在最想聊什么？\n"
                        "Human:我最近准备考试，有点烦。\n"
                        "Assistant:听起来你最近一直绷着。"
                    ),
                    "input": "其实我最怕的是挂科。",
                    "output": "这种担心很真实。你现在最卡住的是复习进度，还是一想到考试就紧张？",
                }
            ]
            input_path.write_text(
                "\n".join(json.dumps(line, ensure_ascii=False) for line in lines),
                encoding="utf-8",
            )

            stats = build_general_multiturn_dataset(
                str(input_path),
                str(output_path),
                min_turns=6,
                max_turns=8,
                limit=10,
                seed=7,
            )

            self.assertEqual(stats["written"], 1)
            record = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["meta"]["source"], "instruction_multiturn")
            self.assertEqual(record["messages"][0]["content"], "你好，你能陪我聊聊吗？")
            self.assertEqual(record["messages"][-1]["role"], "assistant")
            self.assertTrue(any("挂科" in message["content"] for message in record["messages"]))


if __name__ == "__main__":
    unittest.main()
