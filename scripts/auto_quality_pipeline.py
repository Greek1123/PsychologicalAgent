from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

STABLE_ADAPTER = ROOT / "training/ms_swift/outputs/refinement_pool_v5_peft/v0-20260520-215838/checkpoint-final"
BASE_MODEL = Path("D:/llm_cache/modelscope/models/Qwen/Qwen3-4B-Instruct-2507")


def run_command(args: list[str], *, timeout: int | None = None) -> dict[str, Any]:
    completed = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    return {
        "command": args,
        "returncode": completed.returncode,
        "output": completed.stdout.strip(),
    }


def require_success(result: dict[str, Any]) -> None:
    if result["returncode"] != 0:
        command = " ".join(result["command"])
        raise RuntimeError(f"Command failed ({result['returncode']}): {command}\n{result['output']}")


def parse_last_json(output: str) -> dict[str, Any]:
    start = output.find("{")
    end = output.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(output[start : end + 1])
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(payload, dict):
                return payload

    for line in reversed(output.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise RuntimeError(f"No JSON object found in command output:\n{output}")


def summarize_eval_jsonl(path: Path) -> dict[str, Any]:
    cases = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                cases.append(json.loads(line))

    scores: list[int] = []
    flag_counts: Counter[str] = Counter()
    low_score_examples: list[dict[str, Any]] = []
    for case in cases:
        for turn in case.get("turns", []):
            comparison = turn.get("comparison") or {}
            score = int(comparison.get("score") or 0)
            scores.append(score)
            flag_counts.update(comparison.get("flags") or [])
            if score < 65 and len(low_score_examples) < 8:
                low_score_examples.append(
                    {
                        "case_id": case.get("case_id"),
                        "title": case.get("title"),
                        "turn_index": turn.get("turn_index"),
                        "score": score,
                        "flags": comparison.get("flags") or [],
                        "user": turn.get("user", "")[:120],
                    }
                )

    avg_score = round(sum(scores) / len(scores), 2) if scores else 0.0
    return {
        "cases": len(cases),
        "turns": len(scores),
        "average_score": avg_score,
        "flag_counts": dict(flag_counts.most_common()),
        "low_score_examples": low_score_examples,
    }


def write_pipeline_report(report: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{stamp}_auto_quality_pipeline.md"
    lines = [
        "# 自动测评与训练流水线报告",
        "",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 模式：{report['mode']}",
        f"- 评估范围：start={report['start']}, limit={report['limit']}",
        "",
        "## 数据生成",
    ]
    for item in report.get("dataset_steps", []):
        lines.extend(
            [
                f"- `{item['name']}`：returncode={item['returncode']}",
                f"  - 输出：`{item.get('out', '')}`",
                f"  - 记录数：{item.get('records', 'unknown')}",
            ]
        )

    backend = report.get("backend_eval")
    if backend:
        lines.extend(
            [
                "",
                "## 后端 DOCX 评估",
                f"- Markdown：`{backend.get('markdown')}`",
                f"- JSONL：`{backend.get('jsonl')}`",
                f"- 案例数：{backend.get('summary', {}).get('cases')}",
                f"- 回复轮数：{backend.get('summary', {}).get('turns')}",
                f"- 平均启发式评分：{backend.get('summary', {}).get('average_score')}",
                "",
                "### 问题标签",
            ]
        )
        flag_counts = backend.get("summary", {}).get("flag_counts") or {}
        if flag_counts:
            for flag, count in flag_counts.items():
                lines.append(f"- `{flag}`：{count}")
        else:
            lines.append("- 无")

        lines.extend(["", "### 低分样例"])
        low_examples = backend.get("summary", {}).get("low_score_examples") or []
        if low_examples:
            for item in low_examples:
                lines.append(
                    f"- case={item['case_id']} turn={item['turn_index']} score={item['score']} "
                    f"flags={item['flags']} user={item['user']}"
                )
        else:
            lines.append("- 无")

    training = report.get("training")
    if training:
        lines.extend(
            [
                "",
                "## 模型训练",
                f"- 是否执行：{training.get('enabled')}",
                f"- 输出目录：`{training.get('out_dir', '')}`",
                f"- checkpoint：`{training.get('checkpoint', '')}`",
                f"- returncode：{training.get('returncode')}",
            ]
        )

    lines.extend(
        [
            "",
            "## 建议",
            "- 若 `misses_crisis_safety` 或 `misses_privacy_reassurance` 出现，优先补后端 guardrails，再扩充训练样本。",
            "- 若主要问题是 `weak_action_specificity`，优先补低压力具体动作模板和多轮上下文样本。",
            "- 新 LoRA 进入默认推荐前，应和稳定 LoRA 同题跑完整 checkpoint/DOCX 对照。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_datasets() -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    commands = [
        {
            "name": "targeted_refinement",
            "args": [
                PYTHON,
                "scripts/build_targeted_refinement_seed.py",
                "--out",
                "data/training/targeted_refinement/targeted_refinement_auto_pref.jsonl",
                "--ms-swift-out",
                "data/training/targeted_refinement/targeted_refinement_auto_sft.jsonl",
            ],
        },
        {
            "name": "docx_safety_refinement",
            "args": [
                PYTHON,
                "scripts/build_docx_safety_refinement_seed.py",
                "--out",
                "data/training/docx_safety_refinement/docx_safety_refinement_auto_pref.jsonl",
                "--ms-swift-out",
                "data/training/docx_safety_refinement/docx_safety_refinement_auto_sft.jsonl",
            ],
        },
    ]
    for item in commands:
        result = run_command(item["args"], timeout=60)
        require_success(result)
        payload = parse_last_json(result["output"])
        steps.append(
            {
                "name": item["name"],
                "returncode": result["returncode"],
                "out": payload.get("ms_swift_out") or payload.get("out"),
                "records": payload.get("records"),
            }
        )
    return steps


def build_oversampled_dataset(repeat: int) -> Path:
    base_path = ROOT / "data/training/targeted_refinement/targeted_refinement_auto_sft.jsonl"
    safety_path = ROOT / "data/training/docx_safety_refinement/docx_safety_refinement_auto_sft.jsonl"
    out_path = ROOT / "data/training/docx_safety_refinement/auto_safety_oversampled_sft.jsonl"

    rows: list[dict[str, Any]] = []
    for line in base_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    safety_rows = [json.loads(line) for line in safety_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for _ in range(repeat):
        rows.extend(safety_rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out_path


def run_backend_eval(limit: int, start: int, out_dir: Path) -> dict[str, Any]:
    result = run_command(
        [
            PYTHON,
            "scripts/evaluate_backend_docx_reference_cases.py",
            "--limit",
            str(limit),
            "--start",
            str(start),
            "--out-dir",
            str(out_dir),
            "--database-path",
            str(ROOT / "tmp_test_artifacts/auto_quality_backend_eval.db"),
        ],
        timeout=max(120, limit * 30),
    )
    require_success(result)
    payload = parse_last_json(result["output"])
    jsonl_path = Path(payload["jsonl"])
    if not jsonl_path.is_absolute():
        jsonl_path = ROOT / jsonl_path
    payload["summary"] = summarize_eval_jsonl(jsonl_path)
    return payload


def train_patch(dataset: Path, out_dir: Path, *, epochs: float, learning_rate: float) -> dict[str, Any]:
    result = run_command(
        [
            PYTHON,
            "scripts/train_eval_behavior_patch_peft.py",
            "--adapter",
            str(STABLE_ADAPTER),
            "--dataset",
            str(dataset),
            "--out-dir",
            str(out_dir),
            "--epochs",
            str(epochs),
            "--learning-rate",
            str(learning_rate),
            "--max-length",
            "640",
        ],
        timeout=1800,
    )
    require_success(result)
    payload = parse_last_json(result["output"])
    return {
        "enabled": True,
        "returncode": result["returncode"],
        "out_dir": str(out_dir),
        "checkpoint": payload.get("output", str(out_dir / "checkpoint-final")),
        "records": payload.get("records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run automatic DOCX evaluation, comparison report, and optional LoRA training.")
    parser.add_argument("--mode", choices=["backend", "full"], default="backend")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--train", action="store_true", help="Run LoRA patch training after data generation.")
    parser.add_argument("--safety-repeat", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=8e-6)
    parser.add_argument("--out-dir", default=str(ROOT / "reports/auto_quality_pipeline"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    report: dict[str, Any] = {
        "mode": args.mode,
        "limit": args.limit,
        "start": args.start,
        "dataset_steps": build_datasets(),
    }
    dataset = build_oversampled_dataset(args.safety_repeat)
    report["dataset_steps"].append(
        {
            "name": "combined_safety_oversampled",
            "returncode": 0,
            "out": str(dataset),
            "records": sum(1 for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()),
        }
    )

    report["backend_eval"] = run_backend_eval(args.limit, args.start, out_dir / "backend_docx")

    if args.train or args.mode == "full":
        train_out = ROOT / "training/ms_swift/outputs/auto_docx_safety_patch"
        report["training"] = train_patch(dataset, train_out, epochs=args.epochs, learning_rate=args.learning_rate)
    else:
        report["training"] = {"enabled": False}

    report_path = write_pipeline_report(report, out_dir)
    print(json.dumps({"report": str(report_path), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
