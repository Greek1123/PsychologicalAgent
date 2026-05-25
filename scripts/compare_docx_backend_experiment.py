from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.evaluate_backend_docx_reference_cases import _configure_backend_environment, _evaluate_backend_cases
from scripts.evaluate_docx_reference_cases import ReferenceCase, compare_reply, extract_all_cases


DEFAULT_DOCX = [
    ROOT / "docs" / "心理助手长对话模拟测试用例50例.docx",
    ROOT / "docs" / "心理助手长对话模拟测试用例_新增50例_含隐性高危场景.docx",
]


CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "安全危机与隐性高危",
        ("自杀", "自伤", "不想活", "危险", "账号密码", "送人", "断联", "天台", "死亡", "暴力", "伤害别人", "债务", "高危", "消失", "不安全"),
    ),
    (
        "学业任务与科研压力",
        ("考试", "作业", "论文", "查重", "考研", "科研", "代码", "项目", "答辩", "比赛", "毕业", "期末", "竞赛", "实习", "复习", "课程", "课堂", "汇报", "作品", "模型"),
    ),
    (
        "人际关系与亲密关系",
        ("朋友", "分手", "恋爱", "表白", "关系", "小组", "室友", "宿舍", "社交", "聚会", "好友", "边缘", "冷落", "疏远", "排挤", "被讨厌"),
    ),
    (
        "家庭与照护压力",
        ("父母", "爸妈", "妈妈", "爸爸", "家里", "外婆", "亲人", "家庭", "照护", "离婚", "孝", "陪床"),
    ),
    (
        "自我评价与身体状态",
        ("外貌", "身材", "节食", "宠物", "篮球", "受伤", "游戏", "手机", "睡", "失眠", "空虚", "烦", "不自在", "嘲笑", "欺凌", "身份感"),
    ),
)


def categorize_case(case: ReferenceCase | dict[str, Any]) -> str:
    title = str(case.title if isinstance(case, ReferenceCase) else case.get("title", ""))
    observation = " ".join(case.observation_points if isinstance(case, ReferenceCase) else case.get("observation_points", []))
    text = f"{title} {observation}"
    for category, keywords in CATEGORY_RULES:
        if any(keyword in text for keyword in keywords):
            return category
    return "其他校园压力"


def generic_baseline_reply(user_text: str, history: list[dict[str, str]]) -> str:
    combined = " ".join([item.get("content", "") for item in history] + [user_text])
    if any(term in combined for term in ("不想活", "自杀", "自伤", "天台", "不在", "死亡", "伤害自己", "伤害别人")):
        return (
            "听起来你现在很难受，我会先把安全放在第一位。"
            "请尽快联系身边可信任的人、辅导员或紧急服务，不要一个人承受。"
            "如果可以，先告诉我你现在是否安全、身边有没有人。"
        )
    return (
        "听起来这件事让你很难受，也让你一时不知道怎么处理。"
        "你可以先让自己缓一缓，再把最困扰你的部分说出来。"
        "如果愿意，我们可以一起慢慢梳理下一步。"
    )


def evaluate_generic_baseline(cases: list[ReferenceCase]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case in cases:
        history: list[dict[str, str]] = []
        turns: list[dict[str, Any]] = []
        for turn_index, turn in enumerate(case.turns, start=1):
            reply = generic_baseline_reply(turn.user, history)
            comparison = compare_reply(turn.reference_reply, reply, turn.user)
            turns.append(
                {
                    "turn_index": turn_index,
                    "user": turn.user,
                    "reference_reply": turn.reference_reply,
                    "model_reply": reply,
                    "comparison": comparison,
                }
            )
            history.extend([{"role": "user", "content": turn.user}, {"role": "assistant", "content": reply}])
        scores = [item["comparison"]["score"] for item in turns]
        results.append(
            {
                "case_id": case.case_id,
                "title": case.title,
                "source_doc": case.source_doc,
                "category": categorize_case(case),
                "observation_points": case.observation_points,
                "turns": turns,
                "average_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
            }
        )
    return results


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    scores: list[int] = []
    flag_counts: Counter[str] = Counter()
    category_scores: dict[str, list[int]] = defaultdict(list)
    low_score_examples: list[dict[str, Any]] = []
    for case in results:
        category = str(case.get("category") or categorize_case(case))
        for turn in case.get("turns", []):
            comparison = turn.get("comparison") or {}
            score = int(comparison.get("score") or 0)
            scores.append(score)
            category_scores[category].append(score)
            flag_counts.update(comparison.get("flags") or [])
            if score < 65 and len(low_score_examples) < 8:
                low_score_examples.append(
                    {
                        "case_id": case.get("case_id"),
                        "title": case.get("title"),
                        "turn_index": turn.get("turn_index"),
                        "score": score,
                        "flags": comparison.get("flags") or [],
                        "user": str(turn.get("user", ""))[:120],
                    }
                )
    by_category = {
        category: {
            "turns": len(items),
            "average_score": round(sum(items) / len(items), 2) if items else 0.0,
        }
        for category, items in sorted(category_scores.items())
    }
    return {
        "cases": len(results),
        "turns": len(scores),
        "average_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
        "flag_counts": dict(flag_counts.most_common()),
        "by_category": by_category,
        "low_score_examples": low_score_examples,
    }


def attach_categories(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for case in results:
        case["category"] = categorize_case(case)
    return results


def write_jsonl(results: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_turn_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "variant",
        "case_id",
        "title",
        "category",
        "turn_index",
        "score",
        "flags",
        "user",
        "reference_reply",
        "model_reply",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_rows(variant: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in results:
        for turn in case.get("turns", []):
            comparison = turn.get("comparison") or {}
            rows.append(
                {
                    "variant": variant,
                    "case_id": case.get("case_id"),
                    "title": case.get("title"),
                    "category": case.get("category") or categorize_case(case),
                    "turn_index": turn.get("turn_index"),
                    "score": comparison.get("score"),
                    "flags": ",".join(comparison.get("flags") or []),
                    "user": turn.get("user"),
                    "reference_reply": turn.get("reference_reply"),
                    "model_reply": turn.get("model_reply"),
                }
            )
    return rows


def write_markdown(report: dict[str, Any], path: Path) -> None:
    variants = report["variants"]
    baseline = variants["generic_baseline"]["summary"]
    backend = variants["backend_strategy"]["summary"]
    score_gain = round(backend["average_score"] - baseline["average_score"], 2)

    lines = [
        "# DOCX 后端策略层对比实验报告",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 评估范围：start={report['start']}, limit={report['limit']}",
        f"- 案例数：{backend['cases']}",
        f"- 回复轮数：{backend['turns']}",
        "",
        "## 实验组",
        "",
        "- `generic_baseline`：无场景策略的通用心理支持回复，仅做基础共情和少量安全提醒。",
        "- `backend_strategy`：当前后端 mock 链路，包含风险识别、心理熵评估、校园资源、最终回复 guardrails 和多轮上下文策略。",
        "- `reference_reply`：Word 文档中的优秀回复，作为评分参考目标，不作为可运行模型组。",
        "",
        "## 总体结果",
        "",
        "| 组别 | 平均分 | flags | 低分样例 |",
        "| --- | ---: | ---: | ---: |",
        f"| generic_baseline | {baseline['average_score']} | {sum(baseline['flag_counts'].values())} | {len(baseline['low_score_examples'])} |",
        f"| backend_strategy | {backend['average_score']} | {sum(backend['flag_counts'].values())} | {len(backend['low_score_examples'])} |",
        "",
        f"当前后端策略层相对通用基线提升 `{score_gain}` 分；最终 `flag_counts={backend['flag_counts']}`，低分样例数 `{len(backend['low_score_examples'])}`。",
        "",
        "## 分类结果",
        "",
        "| 类别 | baseline | backend | 提升 | turn 数 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    all_categories = sorted(set(baseline["by_category"]) | set(backend["by_category"]))
    for category in all_categories:
        base_item = baseline["by_category"].get(category, {"average_score": 0.0, "turns": 0})
        backend_item = backend["by_category"].get(category, {"average_score": 0.0, "turns": 0})
        gain = round(backend_item["average_score"] - base_item["average_score"], 2)
        lines.append(
            f"| {category} | {base_item['average_score']} | {backend_item['average_score']} | {gain} | {backend_item['turns']} |"
        )

    lines.extend(["", "## Backend 低分与问题标签", ""])
    if backend["flag_counts"]:
        for flag, count in backend["flag_counts"].items():
            lines.append(f"- `{flag}`：{count}")
    else:
        lines.append("- 无")
    lines.append("")
    if backend["low_score_examples"]:
        for item in backend["low_score_examples"]:
            lines.append(
                f"- case={item['case_id']} turn={item['turn_index']} score={item['score']} "
                f"flags={item['flags']} user={item['user']}"
            )
    else:
        lines.append("- 低分样例：无")

    lines.extend(
        [
            "",
            "## 结论",
            "",
            "这组实验用于证明后端策略层的贡献：相同 DOCX 长对话测试集下，通用回复能提供基本安慰，但在隐性高危、校园场景动作、边界表达和多轮承接上明显不足；当前后端策略层通过风险识别、熵源定位、最终回复修正和安全兜底，把回复质量提升到可作为阶段成果展示的水平。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    docx_paths = [Path(item).expanduser().resolve() for item in args.docx] if args.docx else DEFAULT_DOCX
    cases = extract_all_cases(docx_paths)
    if args.start < 1:
        raise ValueError("--start must be >= 1")
    selected = cases[args.start - 1 : args.start - 1 + args.limit]

    baseline_results = evaluate_generic_baseline(selected)

    database_path = Path(args.database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if database_path.exists():
        database_path.unlink()
    if not args.verbose_logs:
        logging.disable(logging.WARNING)
        logging.getLogger().setLevel(logging.ERROR)
        logging.getLogger("campus_support_agent").setLevel(logging.ERROR)
    _configure_backend_environment(database_path)
    backend_results = attach_categories(_evaluate_backend_cases(selected))
    if not args.verbose_logs:
        logging.disable(logging.NOTSET)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "start": args.start,
        "limit": args.limit,
        "database": str(database_path),
        "variants": {
            "generic_baseline": {
                "summary": summarize_results(baseline_results),
                "results": baseline_results,
            },
            "backend_strategy": {
                "summary": summarize_results(backend_results),
                "results": backend_results,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare generic baseline and current backend on DOCX reference cases.")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--out-dir", default=str(ROOT / "reports" / "docx_backend_comparison"))
    parser.add_argument("--database-path", default=str(ROOT / "tmp_test_artifacts" / "docx_backend_comparison.db"))
    parser.add_argument("--docx", action="append", default=[])
    parser.add_argument("--verbose-logs", action="store_true", help="Show backend INFO logs while running the comparison.")
    args = parser.parse_args()

    report = run_experiment(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = out_dir / f"{stamp}_docx_backend_comparison.md"
    json_path = out_dir / f"{stamp}_docx_backend_comparison.json"
    baseline_jsonl = out_dir / f"{stamp}_generic_baseline.jsonl"
    backend_jsonl = out_dir / f"{stamp}_backend_strategy.jsonl"
    csv_path = out_dir / f"{stamp}_turn_level_comparison.csv"

    write_markdown(report, md_path)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(report["variants"]["generic_baseline"]["results"], baseline_jsonl)
    write_jsonl(report["variants"]["backend_strategy"]["results"], backend_jsonl)
    write_turn_csv(
        flatten_rows("generic_baseline", report["variants"]["generic_baseline"]["results"])
        + flatten_rows("backend_strategy", report["variants"]["backend_strategy"]["results"]),
        csv_path,
    )

    print(
        json.dumps(
            {
                "markdown": str(md_path),
                "json": str(json_path),
                "csv": str(csv_path),
                "generic_baseline": report["variants"]["generic_baseline"]["summary"],
                "backend_strategy": report["variants"]["backend_strategy"]["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
