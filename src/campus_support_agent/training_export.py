from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .config import Settings
from .logging_utils import configure_logging, get_logger
from .prompts import build_system_prompt
from .storage import SQLiteSessionStore


logger = get_logger("training_export")


def _detect_language(text: str) -> str:
    # 先做轻量语言识别，够用来区分中英双语训练样本。
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    if re.search(r"[A-Za-z]", text):
        return "en"
    return "unknown"


def _infer_stage_goal(response: dict[str, Any]) -> str:
    risk = (response.get("risk") or {}).get("level", "low")
    entropy = response.get("entropy") or {}
    entropy_level = int(entropy.get("level", 1) or 1)

    # 阶段目标优先服务于“先会聊，再会分析”的训练路线。
    if risk in {"high", "critical"}:
        return "safety_stabilization"
    if entropy_level >= 4:
        return "emotional_containment"
    if entropy_level == 3:
        return "gentle_probe_and_reframe"
    return "summary_and_next_step"


def _target_from_response(response: dict[str, Any]) -> dict[str, Any]:
    return {
        "risk": response.get("risk", {}),
        "entropy": response.get("entropy", {}),
        "entropy_reduction": response.get("entropy_reduction", {}),
        "assessment": response.get("assessment", {}),
        "plan": response.get("plan", {}),
        "campus_resources": response.get("campus_resources", []),
        "safety": response.get("safety", {}),
        "local_policy": response.get("local_policy", {}),
        "referral_decision": response.get("referral_decision", {}),
    }


def _build_record_sample(record: dict[str, Any]) -> dict[str, Any]:
    # record 格式更适合研究分析和后处理，字段尽量保留完整。
    history = list(record.get("conversation_history", []))
    history.append({"role": "user", "content": record.get("input_text", "")})
    response = record.get("response", {})
    language = _detect_language(record.get("input_text", ""))
    stage_goal = _infer_stage_goal(response)

    return {
        "record_id": record.get("response_id"),
        "session_id": record.get("session_id"),
        "source": record.get("source"),
        "created_at": record.get("created_at"),
        "language": language,
        "task_type": "analysis_support",
        "stage_goal": stage_goal,
        "student_context": record.get("student_context", {}),
        "messages": history,
        "target": _target_from_response(response),
        "assistant_json": json.dumps(_target_from_response(response), ensure_ascii=False),
    }


def _build_sft_sample(record: dict[str, Any], system_prompt: str) -> dict[str, Any]:
    # sft 格式直接面向指令微调，把结构化输出作为 assistant 内容。
    response = record.get("response", {})
    stage_goal = _infer_stage_goal(response)
    language = _detect_language(record.get("input_text", ""))

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(record.get("conversation_history", []))
    messages.append({"role": "user", "content": record.get("input_text", "")})
    target = _target_from_response(response)
    messages.append({"role": "assistant", "content": json.dumps(target, ensure_ascii=False)})
    return {
        "id": record.get("response_id"),
        "session_id": record.get("session_id"),
        "language": language,
        "task_type": "analysis_support",
        "stage_goal": stage_goal,
        "messages": messages,
        "student_context": record.get("student_context", {}),
        "meta": {
            "source": record.get("source"),
            "created_at": record.get("created_at"),
        },
    }


def export_training_dataset(
    *,
    db_path: str,
    output_path: str,
    export_format: str = "sft",
    session_id: str | None = None,
    limit: int | None = None,
) -> int:
    settings = Settings()
    configure_logging(settings)
    store = SQLiteSessionStore(db_path=db_path, max_messages=settings.max_history_turns * 2)
    records = store.list_support_responses(session_id=session_id, limit=limit)
    system_prompt = build_system_prompt(settings)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            if export_format == "record":
                sample = _build_record_sample(record)
            else:
                sample = _build_sft_sample(record, system_prompt)
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1

    logger.info(
        "Exported %s training samples to %s using format=%s session_id=%s",
        written,
        output,
        export_format,
        session_id or "-",
    )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Export campus support samples into JSONL training data.")
    parser.add_argument("--db", dest="db_path", default=Settings().database_path, help="SQLite database path.")
    parser.add_argument("--out", dest="output_path", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--format",
        dest="export_format",
        choices=["sft", "record"],
        default="sft",
        help="Export format. sft is for instruction fine-tuning; record keeps richer analysis fields.",
    )
    parser.add_argument("--session-id", dest="session_id", default=None, help="Optional session filter.")
    parser.add_argument("--limit", dest="limit", type=int, default=None, help="Optional max sample count.")
    args = parser.parse_args()

    count = export_training_dataset(
        db_path=args.db_path,
        output_path=args.output_path,
        export_format=args.export_format,
        session_id=args.session_id,
        limit=args.limit,
    )
    print(f"exported {count} samples")


if __name__ == "__main__":
    main()
