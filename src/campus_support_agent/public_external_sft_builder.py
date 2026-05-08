from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

from .logging_utils import get_logger


logger = get_logger("public_external_sft_builder")


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "public_training_datasets" / "batches"

DEFAULT_OUTPUT = ROOT / "data" / "training" / "public_external" / "public_external_candidate_sft_ms_swift.jsonl"
DEFAULT_SYSTEM = (
    "You are a warm, safe, and practical mental health support assistant. "
    "Do not diagnose. For crisis or self-harm risk, prioritize immediate safety and encourage real-world support."
)

SOURCE_LIMITS = {
    "mhdialog": 300,
    "mind_corpus": 80,
    "mental_health_therapy": 300,
    "prince_mental_health_conv": 250,
    "kurtis_mental_health_final": 250,
    "shivomh_support": 300,
    "zahrizhalali_conversation": 100,
}

LOW_VALUE_MARKERS = (
    "as an ai language model",
    "i cannot provide",
    "consult a professional for diagnosis",
)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _record(messages: list[dict[str, str]], source: str) -> dict[str, Any] | None:
    cleaned: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        content = _text(message.get("content"))
        if role in {"system", "user", "assistant"} and content:
            cleaned.append({"role": role, "content": content})

    roles = {message["role"] for message in cleaned}
    if "user" not in roles or "assistant" not in roles:
        return None

    assistant_text = "\n".join(message["content"] for message in cleaned if message["role"] == "assistant")
    if len(assistant_text) < 20:
        return None
    if any(marker in assistant_text.lower() for marker in LOW_VALUE_MARKERS):
        return None

    return {"messages": cleaned, "meta": {"source": source, "language": "en", "needs_review": True}}


def _dedupe_key(record: dict[str, Any]) -> str:
    return json.dumps(record.get("messages", []), ensure_ascii=False, sort_keys=True)


def _sample(records: list[dict[str, Any]], limit: int, rng: random.Random) -> list[dict[str, Any]]:
    if limit > 0 and len(records) > limit:
        return rng.sample(records, limit)
    return records


def _load_pandas():
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("pandas/pyarrow is required to convert parquet/csv external datasets.") from exc
    return pd


def _from_mhdialog(path: Path) -> list[dict[str, Any]]:
    pd = _load_pandas()
    df = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for raw in df["Dialogue"].dropna():
        try:
            turns = json.loads(raw)
        except json.JSONDecodeError:
            continue
        messages = [{"role": "system", "content": DEFAULT_SYSTEM}]
        for turn in turns:
            user = _text(turn.get("user"))
            supporter = _text(turn.get("supporter"))
            if user:
                messages.append({"role": "user", "content": user})
            if supporter:
                messages.append({"role": "assistant", "content": supporter})
        converted = _record(messages, "MHDialog")
        if converted:
            rows.append(converted)
    return rows


def _from_mind_corpus(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for item in data:
        messages = [{"role": "system", "content": DEFAULT_SYSTEM}]
        for turn in item.get("conversations", []):
            role = {"human": "user", "gpt": "assistant"}.get(turn.get("from"))
            if role:
                messages.append({"role": role, "content": turn.get("value", "")})
        converted = _record(messages, "Mind-Corpus")
        if converted:
            rows.append(converted)
    return rows


def _from_instruction_parquet(path: Path, *, source: str, input_col: str, output_col: str) -> list[dict[str, Any]]:
    pd = _load_pandas()
    df = pd.read_parquet(path)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        converted = _record(
            [
                {"role": "system", "content": DEFAULT_SYSTEM},
                {"role": "user", "content": row.get(input_col, "")},
                {"role": "assistant", "content": row.get(output_col, "")},
            ],
            source,
        )
        if converted:
            rows.append(converted)
    return rows


def _from_prince(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for item in data:
        question = _text(" ".join([_text(item.get("questionTitle")), _text(item.get("questionText"))]))
        converted = _record(
            [
                {"role": "system", "content": DEFAULT_SYSTEM},
                {"role": "user", "content": question},
                {"role": "assistant", "content": item.get("answerText", "")},
            ],
            "PrinceAyush_Mental_Health_conv",
        )
        if converted:
            rows.append(converted)
    return rows


def _from_shivomh_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in _read_jsonl(path):
        messages = []
        for message in item.get("messages", []):
            role = message.get("role")
            if role == "system":
                messages.append({"role": "system", "content": DEFAULT_SYSTEM})
            elif role in {"user", "assistant"}:
                messages.append({"role": role, "content": message.get("content", "")})
        converted = _record(messages, "ShivomH_MentalHealth-Support")
        if converted:
            rows.append(converted)
    return rows


def _from_zahrizhalali(path: Path) -> list[dict[str, Any]]:
    pd = _load_pandas()
    df = pd.read_parquet(path)
    rows: list[dict[str, Any]] = []
    pattern = re.compile(r"<HUMAN>:\s*(.*?)\s*<ASSISTANT>:\s*(.*)", re.S)
    for text in df["text"].dropna():
        match = pattern.search(str(text))
        if not match:
            continue
        converted = _record(
            [
                {"role": "system", "content": DEFAULT_SYSTEM},
                {"role": "user", "content": match.group(1)},
                {"role": "assistant", "content": match.group(2)},
            ],
            "ZahrizhalAli_mental_health_conversational_dataset",
        )
        if converted:
            rows.append(converted)
    return rows


def _source_loaders() -> dict[str, tuple[Path, Any]]:
    return {
        "mhdialog": (
            DATA_ROOT / "2026-05-07_quality_batch_04" / "MHDialog" / "train.csv",
            _from_mhdialog,
        ),
        "mind_corpus": (
            DATA_ROOT / "2026-05-07_quality_batch_04" / "Mind-Corpus" / "mindcorpus.json",
            _from_mind_corpus,
        ),
        "mental_health_therapy": (
            DATA_ROOT / "2026-05-07_quality_batch_04" / "mental_health_therapy" / "train.parquet",
            lambda path: _from_instruction_parquet(
                path,
                source="mental_health_therapy",
                input_col="input",
                output_col="output",
            ),
        ),
        "prince_mental_health_conv": (
            DATA_ROOT / "2026-05-07_quality_batch_04" / "PrinceAyush_Mental_Health_conv" / "cl_output_file.json",
            _from_prince,
        ),
        "kurtis_mental_health_final": (
            DATA_ROOT / "2026-05-07_quality_batch_05" / "kurtis_mental_health_final" / "train.parquet",
            lambda path: _from_instruction_parquet(
                path,
                source="kurtis_mental_health_final",
                input_col="question",
                output_col="answer",
            ),
        ),
        "shivomh_support": (
            DATA_ROOT / "2026-05-07_quality_batch_06" / "ShivomH_MentalHealth-Support" / "MH_final_train.jsonl",
            _from_shivomh_jsonl,
        ),
        "zahrizhalali_conversation": (
            DATA_ROOT
            / "2026-05-07_quality_batch_07"
            / "ZahrizhalAli_mental_health_conversational_dataset"
            / "train.parquet",
            _from_zahrizhalali,
        ),
    }


def build_public_external_sft_dataset(out: str, *, seed: int = 42) -> dict[str, Any]:
    rng = random.Random(seed)
    output_path = Path(out)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    stats: dict[str, dict[str, int | str]] = {}

    for name, (path, loader) in _source_loaders().items():
        if not path.exists():
            stats[name] = {"path": str(path), "missing": 1, "written": 0}
            continue
        loaded = loader(path)
        sampled = _sample(loaded, SOURCE_LIMITS[name], rng)
        written = 0
        for record in sampled:
            key = _dedupe_key(record)
            if key in seen:
                continue
            seen.add(key)
            records.append(record)
            written += 1
        stats[name] = {"path": str(path), "loaded": len(loaded), "written": written}

    rng.shuffle(records)
    written_total = _write_jsonl(output_path, records)
    result = {"output": str(output_path), "written": written_total, "sources": stats}
    logger.info("Built public external SFT candidates: %s", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert selected external mental-health datasets to ms-swift SFT candidate rows.")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build_public_external_sft_dataset(args.out, seed=args.seed), ensure_ascii=False))


if __name__ == "__main__":
    main()
