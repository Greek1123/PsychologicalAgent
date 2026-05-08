from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .feedback_review_sheet import REVIEW_COLUMNS
from .logging_utils import get_logger


logger = get_logger("feedback_ai_reviewer")

DEFAULT_SYSTEM_PROMPT = """你是校园心理支持助手训练数据的中文标注员。
你的任务不是当心理医生，而是帮助改写模型的坏回复，让它更像一个温和、自然、尊重边界的校园心理支持助手。

请严格遵守：
1. 不诊断疾病，不吓唬用户。
2. 不暴露“心理熵、风险等级、模型分析、后端策略”等后台概念。
3. 如果用户不想细说，要尊重边界，不要逼问。
4. 如果用户担心隐私，要明确说明会尊重隐私和边界。
5. 如果用户只输入“？/嗯/1/2/算了/不想说”，要修复对话，不要机械跟随。
6. 回复要像正常中文对话，有支持感，有一点具体下一步，不要太短，不要模板化。
7. 不要使用 emoji、颜文字、网络撒娇语气，例如“抱抱你”“宝”“啦～”“呀”“??”。
8. 不要做绝对承诺，例如“不会挂科”“一定会好”“我不会告诉任何人”。
9. 隐私相关表达要谨慎：可以说“我会尊重你的隐私和边界”，但不能承诺任何情况下绝对保密。
10. 不要替用户下结论，不要说“你已经赢过很多人”“你就是太焦虑”。
11. 回复长度控制在 80-180 个中文字符；最多 2 段。
12. 输出必须是 JSON，不要输出 Markdown。
"""

DEFAULT_USER_TEMPLATE = """请审核下面这条模型回复，并给出更好的中文回复。

上下文/用户输入：
{input_text}

模型原回复：
{assistant_reply}

风险等级：{risk_level}
心理熵分数：{entropy_score}

请输出 JSON，字段如下：
{{
  "mark_bad": "1 或 0",
  "problem_tags": ["privacy_missed", "too_short", "repetitive", "pushy", "topic_drift", "number_following", "professional_jargon", "unsafe", "fake_identity", "time_hallucination"],
  "chosen": "如果原回复不好，写一条更好的中文回复；如果原回复已经可以，返回空字符串。chosen 不能包含 emoji、颜文字、绝对承诺、诊断、后台术语。",
  "review_note": "简短说明原回复的问题"
}}
"""

TAG_ALLOWLIST = {
    "privacy_missed",
    "too_short",
    "repetitive",
    "pushy",
    "topic_drift",
    "number_following",
    "professional_jargon",
    "unsafe",
    "fake_identity",
    "time_hallucination",
}

FORBIDDEN_CHOSEN_PATTERNS = [
    "抱抱你",
    "宝",
    "宝",
    "??",
    "不会挂科",
    "一定会好",
    "我不会告诉任何人",
    "不会跟任何人说",
    "绝对保密",
    "你已经赢过很多人",
]


def _post_json(url: str, payload: dict[str, Any], api_key: str, timeout_seconds: int) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Reviewer request failed: HTTP {exc.code} - {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Reviewer request failed: {exc.reason}") from exc


def _chat_completion(
    *,
    base_url: str,
    model: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    timeout_seconds: int,
    temperature: float,
    max_tokens: int,
) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    response = _post_json(url, payload, api_key, timeout_seconds)
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError("Reviewer request failed: missing choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Reviewer request failed: missing message content.")
    return content.strip()


def _parse_json_object(raw_text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, flags=re.S)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Reviewer output must be a JSON object.")
    return parsed


def _redact_private_text(text: str) -> str:
    redacted = text
    redacted = re.sub(r"1[3-9]\d{9}", "[手机号]", redacted)
    redacted = re.sub(r"\b\d{15,18}[\dXx]?\b", "[证件号]", redacted)
    redacted = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[邮箱]", redacted)
    redacted = re.sub(r"(微信|vx|VX|qq|QQ)[:：]?\s*[A-Za-z0-9_-]{5,}", r"\1:[账号]", redacted)
    return redacted


def _normalize_tags(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_tags = [item.strip() for item in value.replace("，", ",").split(",")]
    elif isinstance(value, list):
        raw_tags = [str(item).strip() for item in value]
    else:
        raw_tags = []
    return [tag for tag in raw_tags if tag in TAG_ALLOWLIST]


def _clean_chosen_reply(value: str) -> str:
    cleaned = str(value or "").strip()
    cleaned = cleaned.replace("??", "")
    cleaned = re.sub(r"[~～]{2,}", "。", cleaned)
    cleaned = re.sub(r"[\U00010000-\U0010ffff]", "", cleaned)
    cleaned = cleaned.replace("不会跟任何人说", "会尊重你的隐私和边界")
    cleaned = cleaned.replace("我不会告诉任何人", "我会尊重你的隐私和边界")
    cleaned = cleaned.replace("绝对保密", "尊重隐私和边界")
    return cleaned.strip()


def _has_forbidden_chosen_pattern(value: str) -> bool:
    return any(pattern in value for pattern in FORBIDDEN_CHOSEN_PATTERNS)


def _should_review(row: dict[str, str], *, only_empty: bool) -> bool:
    chosen = str(row.get("chosen") or "").strip()
    if only_empty and chosen and not _has_forbidden_chosen_pattern(chosen):
        return False
    return bool(str(row.get("input_text") or "").strip())


def _review_row(
    row: dict[str, str],
    *,
    base_url: str,
    model: str,
    api_key: str,
    system_prompt: str,
    timeout_seconds: int,
    temperature: float,
    max_tokens: int,
    redact: bool,
) -> dict[str, str]:
    input_text = str(row.get("input_text") or "")
    assistant_reply = str(row.get("assistant_reply") or "")
    if redact:
        input_text = _redact_private_text(input_text)
        assistant_reply = _redact_private_text(assistant_reply)

    user_prompt = DEFAULT_USER_TEMPLATE.format(
        input_text=input_text,
        assistant_reply=assistant_reply,
        risk_level=row.get("risk_level", ""),
        entropy_score=row.get("entropy_score", ""),
    )
    raw_output = _chat_completion(
        base_url=base_url,
        model=model,
        api_key=api_key,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    parsed = _parse_json_object(raw_output)
    tags = _normalize_tags(parsed.get("problem_tags"))
    chosen = _clean_chosen_reply(str(parsed.get("chosen") or "").strip())
    mark_bad = str(parsed.get("mark_bad") or "").strip()
    review_note = str(parsed.get("review_note") or "").strip()

    updated = dict(row)
    if mark_bad in {"1", "0"}:
        updated["mark_bad"] = mark_bad
    elif chosen:
        updated["mark_bad"] = "1"
    if tags:
        updated["problem_tags"] = ",".join(tags)
    if chosen and not _has_forbidden_chosen_pattern(chosen):
        updated["chosen"] = chosen
    elif chosen:
        updated["review_note"] = "AI draft skipped because it contained forbidden wording; please rewrite manually."
        updated["mark_bad"] = "1"
    if review_note:
        updated["review_note"] = review_note
    return updated


def autofill_review_sheet_with_ai(
    *,
    input_csv_path: str,
    output_csv_path: str,
    base_url: str,
    model: str,
    api_key: str,
    limit: int | None = None,
    only_empty: bool = True,
    timeout_seconds: int = 60,
    temperature: float = 0.2,
    max_tokens: int = 512,
    sleep_seconds: float = 0.0,
    redact: bool = True,
) -> dict[str, Any]:
    input_path = Path(input_csv_path)
    output_path = Path(output_csv_path)
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    reviewed = 0
    failed = 0
    output_rows: list[dict[str, str]] = []
    fieldnames = list(dict.fromkeys([*REVIEW_COLUMNS, *[key for row in rows for key in row.keys()]]))

    def save_progress() -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)

    for row in rows:
        should_review = _should_review(row, only_empty=only_empty) and (limit is None or reviewed < limit)
        if not should_review:
            output_rows.append(row)
            continue
        try:
            output_rows.append(
                _review_row(
                    row,
                    base_url=base_url,
                    model=model,
                    api_key=api_key,
                    system_prompt=DEFAULT_SYSTEM_PROMPT,
                    timeout_seconds=timeout_seconds,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    redact=redact,
                )
            )
            reviewed += 1
            save_progress()
            print(
                json.dumps(
                    {
                        "progress": reviewed,
                        "failed": failed,
                        "total_rows": len(rows),
                        "output": str(output_path),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        except Exception as exc:  # pragma: no cover - external service failures vary.
            failed += 1
            logger.warning("Failed to AI-review row id=%s: %s", row.get("id"), exc)
            output_rows.append(row)
            save_progress()

    save_progress()

    stats = {
        "input": str(input_path),
        "output": str(output_path),
        "reviewed": reviewed,
        "failed": failed,
        "total_rows": len(rows),
    }
    logger.info("AI-filled feedback review sheet: %s", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Use an external OpenAI-compatible LLM to draft review_sheet.csv.")
    parser.add_argument("--input", required=True, help="Input review_sheet.csv.")
    parser.add_argument("--out", required=True, help="Output review_sheet.csv.")
    parser.add_argument("--base-url", default=os.getenv("REVIEW_LLM_BASE_URL", ""))
    parser.add_argument("--model", default=os.getenv("REVIEW_LLM_MODEL", ""))
    parser.add_argument("--api-key", default=os.getenv("REVIEW_LLM_API_KEY", ""))
    parser.add_argument("--limit", type=int, default=None, help="Maximum rows to ask external AI to review.")
    parser.add_argument("--include-filled", action="store_true", help="Also overwrite rows that already have chosen.")
    parser.add_argument("--timeout-seconds", type=int, default=int(os.getenv("REVIEW_LLM_TIMEOUT_SECONDS", "60")))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("REVIEW_LLM_TEMPERATURE", "0.2")))
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("REVIEW_LLM_MAX_TOKENS", "512")))
    parser.add_argument("--sleep-seconds", type=float, default=float(os.getenv("REVIEW_LLM_SLEEP_SECONDS", "0")))
    parser.add_argument("--no-redact", action="store_true", help="Do not redact common private identifiers before sending.")
    args = parser.parse_args()

    if not args.base_url or not args.model or not args.api_key:
        raise SystemExit(
            "Missing reviewer config. Set REVIEW_LLM_BASE_URL, REVIEW_LLM_MODEL, REVIEW_LLM_API_KEY "
            "or pass --base-url --model --api-key."
        )

    stats = autofill_review_sheet_with_ai(
        input_csv_path=args.input,
        output_csv_path=args.out,
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        limit=args.limit,
        only_empty=not args.include_filled,
        timeout_seconds=args.timeout_seconds,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        sleep_seconds=args.sleep_seconds,
        redact=not args.no_redact,
    )
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
