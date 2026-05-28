from __future__ import annotations

import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, Settings
from .deployment_readiness import build_deployment_readiness


@dataclass(slots=True)
class ReportArtifacts:
    docx_eval_jsonl: Path | None
    manual_check_json: Path | None
    test_summary: str


def find_latest_artifacts(root: Path = PROJECT_ROOT, *, test_summary: str = "not run in this report") -> ReportArtifacts:
    return ReportArtifacts(
        docx_eval_jsonl=_latest_file(root / "reports" / "auto_quality_pipeline", "*_extracted_docx_reference_eval.jsonl"),
        manual_check_json=_latest_file(root / "reports" / "manual_reply_checks", "*_manual_reply_check.json"),
        test_summary=test_summary,
    )


def build_acceptance_report(
    *,
    artifacts: ReportArtifacts,
    settings: Settings | None = None,
    generated_at: str | None = None,
) -> str:
    settings = settings or Settings()
    readiness = build_deployment_readiness(settings)
    docx_summary = summarize_docx_eval(artifacts.docx_eval_jsonl)
    manual_summary = summarize_manual_check(artifacts.manual_check_json)
    git_summary = get_git_summary()
    generated_at = generated_at or datetime.now().isoformat(timespec="seconds")

    return "\n".join(
        [
            "# 系统验收报告",
            "",
            f"- 生成时间：{generated_at}",
            f"- 当前分支/提交：{git_summary}",
            f"- 部署自检：{readiness['status']}，pass={readiness['summary']['pass']}，fail={readiness['summary']['fail']}",
            f"- 自动测试：{artifacts.test_summary}",
            "",
            "## 当前系统层级",
            "",
            "| 层级 | 状态 | 说明 |",
            "| --- | --- | --- |",
            "| 多模态输入层 | 已完成原型 | 文本与语音入口已接入同一 Agent 链路，语音保留基础音频信号。 |",
            "| 心理熵与策略层 | 已完成核心闭环 | 支持风险识别、心理熵评估、熵减策略、动态调整和校园资源匹配。 |",
            "| DOCX 质量评估层 | 已达标 | 对照长对话优秀回复做自动测评，并保留低分样例定位能力。 |",
            "| 前端交接层 | 已完成后端契约 | 提供 `/api/v1/frontend/contract` 和角色视图接口。 |",
            "| 人工干预层 | 已完成第一版闭环 | 支持 care queue、人工确认、升级、解决和关闭。 |",
            "| 隐私边界层 | 已完成第一版 | 学生、咨询师、研究、管理员四类视图由后端投影。 |",
            "| 部署运维层 | 已完成第一版 | 提供 readiness API 与终端自检脚本。 |",
            "",
            "## 关键验收指标",
            "",
            "| 指标 | 当前结果 | 来源 |",
            "| --- | --- | --- |",
            f"| DOCX 后端平均分 | {docx_summary['average_score']} | {docx_summary['source']} |",
            f"| DOCX 案例/轮次 | {docx_summary['cases']} / {docx_summary['turns']} | {docx_summary['source']} |",
            f"| DOCX 问题标签 | {docx_summary['flag_counts']} | 自动评估 JSONL |",
            f"| DOCX 低分样例数 | {docx_summary['low_score_count']} | 自动评估 JSONL |",
            f"| 手动抽检 PASS/WARN | {manual_summary['pass']} / {manual_summary['warn']} | {manual_summary['source']} |",
            f"| 手动抽检场景/轮次 | {manual_summary['scenarios']} / {manual_summary['turns']} | 手动抽检 JSON |",
            f"| 部署 readiness | {readiness['status']} | `scripts/check_deployment_readiness.py` |",
            f"| 单元/回归测试 | {artifacts.test_summary} | pytest |",
            "",
            "## 主要接口",
            "",
            "- `POST /api/v1/support/text`：文本心理支持入口。",
            "- `POST /api/v1/support/audio`：语音心理支持入口。",
            "- `GET /api/v1/frontend/contract`：前端交接契约。",
            "- `GET /api/v1/sessions/{session_id}/view?role=student|counselor|research|admin`：角色视图与隐私边界。",
            "- `GET /api/v1/analytics/care-queue`：人工关注队列。",
            "- `POST /api/v1/sessions/{session_id}/human-interventions`：人工处理记录。",
            "- `GET /api/v1/ops/readiness`：部署自检。",
            "",
            "## 推荐交付说明",
            "",
            "- 给前端组员：优先对接 `frontend/contract`、`support/text`、`support/audio` 和角色视图接口。",
            "- 给负责模型的组员：默认使用 README 中的 Qwen3 基础模型 + `refinement_pool_v5_peft` LoRA。",
            "- 给答辩/论文材料：使用 DOCX 自动测评、手动抽检、熵轨迹导出和后端对比实验作为实验支撑。",
            "- 给部署同学：启动前先运行 `python scripts\\check_deployment_readiness.py`。",
            "- 给演示准备：运行 `python scripts\\generate_demo_workspace.py` 生成 `docs/demo_workspace_report.md`。",
            "",
            "## 下一步建议",
            "",
            "1. 补正式咨询师工作台页面，把 care queue 和 human-interventions 可视化。",
            "2. 接真实 ASR 服务，并把语音停顿、音量、静音比例纳入多模态展示。",
            "3. 将该验收报告脚本纳入每轮迭代流程，形成固定答辩材料。",
            "",
        ]
    )


def summarize_docx_eval(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {
            "source": "missing",
            "cases": 0,
            "turns": 0,
            "average_score": "unknown",
            "flag_counts": {},
            "low_score_count": "unknown",
        }

    results = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    scores: list[int] = []
    flag_counts: Counter[str] = Counter()
    low_score_count = 0
    for case in results:
        for turn in case.get("turns", []):
            comparison = turn.get("comparison") or {}
            score = int(comparison.get("score") or 0)
            scores.append(score)
            flag_counts.update(comparison.get("flags") or [])
            if score < 65:
                low_score_count += 1
    return {
        "source": _relative(path),
        "cases": len(results),
        "turns": len(scores),
        "average_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
        "flag_counts": dict(flag_counts.most_common()),
        "low_score_count": low_score_count,
    }


def summarize_manual_check(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {"source": "missing", "scenarios": 0, "turns": 0, "pass": 0, "warn": 0}
    rows = json.loads(path.read_text(encoding="utf-8"))
    scenario_ids = {row.get("scenario_id") for row in rows}
    statuses = Counter(str(row.get("check_status") or "UNKNOWN") for row in rows)
    return {
        "source": _relative(path),
        "scenarios": len(scenario_ids),
        "turns": len(rows),
        "pass": statuses.get("PASS", 0),
        "warn": statuses.get("WARN", 0),
    }


def get_git_summary() -> str:
    try:
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=PROJECT_ROOT, text=True).strip()
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{branch}@{commit}"


def _latest_file(root: Path, pattern: str) -> Path | None:
    if not root.exists():
        return None
    files = [path for path in root.rglob(pattern) if path.is_file()]
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_mtime)


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)
