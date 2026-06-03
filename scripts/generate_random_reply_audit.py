from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.evaluate_backend_docx_reference_cases import _configure_backend_environment


def _latest_backend_jsonl() -> Path:
    candidates = sorted(
        (ROOT / "reports" / "auto_quality_pipeline" / "backend_docx").glob(
            "*_extracted_docx_reference_eval.jsonl"
        ),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No backend DOCX eval JSONL found. Run auto_quality_pipeline first.")
    return candidates[0]


def _load_turns(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        case_turns = case.get("turns") or []
        for turn in case_turns:
            rows.append(
                {
                    "case_id": str(case.get("case_id") or ""),
                    "title": str(case.get("title") or ""),
                    "source_doc": str(case.get("source_doc") or ""),
                    "turn_index": int(turn.get("turn_index") or 0),
                    "user": str(turn.get("user") or ""),
                    "reference_reply": str(turn.get("reference_reply") or ""),
                    "model_reply": str(turn.get("model_reply") or ""),
                    "comparison": turn.get("comparison") or {},
                    "backend": turn.get("backend") or {},
                    "case_turns": case_turns,
                }
            )
    return rows


def _history_until_sample(sample: dict[str, Any]) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    target_index = int(sample["turn_index"])
    for turn in sample["case_turns"]:
        turn_index = int(turn.get("turn_index") or 0)
        if turn_index > target_index:
            break
        history.append({"role": "user", "content": str(turn.get("user") or "")})
        history.append({"role": "assistant", "content": str(turn.get("model_reply") or "")})
    return history


def _extra_prompts(sample: dict[str, Any]) -> list[dict[str, str]]:
    text = f"{sample['title']} {sample['user']}"

    if any(term in text for term in ("天台", "高处", "楼顶", "消失", "不想活", "账号", "不在了")):
        return [
            {"kind": "模糊弱输出", "text": "算了，我不想解释了，反正也没人真的懂。"},
            {"kind": "上下文推进", "text": "如果我现在只想一个人待着，不想让室友知道，可以吗？"},
        ]
    if any(term in text for term in ("隐私", "照片", "前任", "匿名", "表白墙", "评论", "截图", "网上")):
        return [
            {"kind": "模糊弱输出", "text": "我脑子很乱，只想把手机关掉。"},
            {"kind": "上下文推进", "text": "如果我不敢告诉辅导员，只告诉一个同学够不够？"},
        ]
    if any(term in text for term in ("室友", "宿舍", "舍友", "班级", "朋友", "社交", "同学")):
        return [
            {"kind": "模糊弱输出", "text": "嗯……我就是觉得自己很多余。"},
            {"kind": "上下文推进", "text": "如果我明天还要见到他们，我应该先做什么？"},
        ]
    if any(term in text for term in ("考研", "考公", "找工作", "实习", "面试", "项目", "毕业", "简历")):
        return [
            {"kind": "模糊弱输出", "text": "我知道要行动，但我现在就是启动不了。"},
            {"kind": "上下文推进", "text": "如果只能做十分钟，你会让我先做哪一步？"},
        ]
    if any(term in text for term in ("恋爱", "分手", "前任", "喜欢", "表白", "关系")):
        return [
            {"kind": "模糊弱输出", "text": "我好像又想给他发消息了。"},
            {"kind": "上下文推进", "text": "如果我发完又后悔，怎么提前拦住自己？"},
        ]
    if any(term in text for term in ("睡不着", "吃不下", "发抖", "坐车", "事故", "身体")):
        return [
            {"kind": "模糊弱输出", "text": "我说不清，就是身体一直紧着。"},
            {"kind": "上下文推进", "text": "如果明天还要正常上课或出门，我今晚先做什么？"},
        ]
    return [
        {"kind": "模糊弱输出", "text": "我不知道怎么说，就是很堵。"},
        {"kind": "上下文推进", "text": "如果我只愿意先做一小步，你建议是哪一步？"},
    ]


def _ask_backend(history: list[dict[str, str]], prompt: str) -> dict[str, Any]:
    from campus_support_agent import main

    response = main.support_text(
        {
            "session_id": f"random-audit-{uuid4().hex}",
            "text": prompt,
            "student_context": {},
            "conversation_history": history,
        }
    )
    return {
        "prompt": prompt,
        "reply": str(response.get("reply_text") or response.get("reply") or ""),
        "risk_level": ((response.get("risk") or {}).get("level")),
        "entropy_score": ((response.get("entropy") or {}).get("score")),
        "policy_name": (((response.get("support_assessment") or {}).get("local_policy") or {}).get("policy_name")),
    }


def _render_markdown(samples: list[dict[str, Any]], *, source_jsonl: Path, seed: int) -> str:
    lines: list[str] = [
        "# 随机 30 条真实回复人工抽检",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"随机种子：`{seed}`",
        f"来源评估 JSONL：`{source_jsonl}`",
        "",
        "说明：每个样本先展示源文档中的真实问题、当前后端模型回复、文档优秀回复；随后追加 2 条源文档没有的问题，继续调用当前后端，观察模糊输入、弱输出和上下文推理表现。",
        "",
    ]
    for index, item in enumerate(samples, start=1):
        comparison = item.get("comparison") or {}
        backend = item.get("backend") or {}
        lines.extend(
            [
                f"## 样本 {index:02d}：Case {item['case_id']} / Turn {item['turn_index']} / {item['title']}",
                "",
                f"- 来源文档：`{item['source_doc']}`",
                f"- 原评估分：`{comparison.get('score', '')}`",
                f"- 风险等级：`{backend.get('risk_level', '')}`",
                f"- 心理熵：`{backend.get('entropy_score', '')}`",
                f"- 策略：`{backend.get('local_policy', '')}`",
                "",
                "### 我询问的内容（源文档）",
                "",
                item["user"],
                "",
                "### 模型回复（当前后端真实输出）",
                "",
                item["model_reply"],
                "",
                "### 对比文档标准内容（优秀回复）",
                "",
                item["reference_reply"],
                "",
            ]
        )
        for extra_index, extra in enumerate(item["extra_turns"], start=1):
            lines.extend(
                [
                    f"### 额外追问 {extra_index}（{extra['kind']}，源文档无标准答案）",
                    "",
                    "**我询问的内容：**",
                    "",
                    extra["prompt"],
                    "",
                    "**模型回复：**",
                    "",
                    extra["reply"],
                    "",
                    f"- 风险等级：`{extra.get('risk_level', '')}`",
                    f"- 心理熵：`{extra.get('entropy_score', '')}`",
                    f"- 策略：`{extra.get('policy_name', '')}`",
                    "",
                ]
            )
    return "\n".join(lines).strip() + "\n"


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="Generate a random human-readable reply audit from DOCX eval results.")
    parser.add_argument("--jsonl", default="")
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--out-dir", default=str(ROOT / "reports" / "random_reply_audits"))
    parser.add_argument("--database-path", default=str(ROOT / "tmp_test_artifacts" / "random_reply_audit.db"))
    args = parser.parse_args()

    source_jsonl = Path(args.jsonl).resolve() if args.jsonl else _latest_backend_jsonl()
    rows = _load_turns(source_jsonl)
    if args.sample_size > len(rows):
        raise ValueError(f"sample-size {args.sample_size} exceeds available turns {len(rows)}")

    database_path = Path(args.database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if database_path.exists():
        database_path.unlink()
    _configure_backend_environment(database_path)

    rng = random.Random(args.seed)
    samples = rng.sample(rows, args.sample_size)
    samples.sort(key=lambda item: (int(item["case_id"] or 0), int(item["turn_index"] or 0)))

    for sample in samples:
        history = _history_until_sample(sample)
        extra_turns: list[dict[str, Any]] = []
        for extra in _extra_prompts(sample):
            result = _ask_backend(history, extra["text"])
            result["kind"] = extra["kind"]
            extra_turns.append(result)
            history.extend(
                [
                    {"role": "user", "content": result["prompt"]},
                    {"role": "assistant", "content": result["reply"]},
                ]
            )
        sample["extra_turns"] = extra_turns

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = out_dir / f"{stamp}_random_reply_audit.md"
    json_path = out_dir / f"{stamp}_random_reply_audit.json"
    md_path.write_text(_render_markdown(samples, source_jsonl=source_jsonl, seed=args.seed), encoding="utf-8")
    json_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "source_jsonl": str(source_jsonl),
                "sample_size": len(samples),
                "seed": args.seed,
                "markdown": str(md_path),
                "json": str(json_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main_cli()
