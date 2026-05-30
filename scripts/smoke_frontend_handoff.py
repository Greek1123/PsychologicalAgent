from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = "明天早上考试，我现在完全睡不着，越想越慌。"


def _request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 15,
) -> tuple[int, dict[str, Any]]:
    url = f"{base_url.rstrip('/')}{path}"
    body = None
    headers: dict[str, str] = {}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=body, headers=headers, method=method)
    with urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return response.status, json.loads(raw) if raw else {}


def _run_step(name: str, fn) -> dict[str, Any]:
    try:
        status, data = fn()
        return {"name": name, "ok": 200 <= status < 300, "status": status, "data": data}
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return {"name": name, "ok": False, "status": exc.code, "error": raw}
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return {"name": name, "ok": False, "status": None, "error": str(exc)}


def run_smoke(base_url: str, session_id: str, prompt: str) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    steps.append(_run_step("health", lambda: _request_json(base_url, "/health")))
    steps.append(_run_step("frontend_contract", lambda: _request_json(base_url, "/api/v1/frontend/contract")))
    steps.append(_run_step("ops_readiness", lambda: _request_json(base_url, "/api/v1/ops/readiness")))
    steps.append(
        _run_step(
            "text_support",
            lambda: _request_json(
                base_url,
                "/api/v1/support/text",
                method="POST",
                payload={
                    "session_id": session_id,
                    "text": prompt,
                    "student_context": {"grade": "大二", "campus": "main"},
                    "conversation_history": [],
                },
            ),
        )
    )
    role_query = urlencode({"role": "student"})
    steps.append(
        _run_step(
            "student_role_view",
            lambda: _request_json(base_url, f"/api/v1/sessions/{session_id}/view?{role_query}"),
        )
    )
    steps.append(
        _run_step(
            "care_queue",
            lambda: _request_json(
                base_url,
                "/api/v1/analytics/care-queue?include_low_priority=true&include_resolved=true",
            ),
        )
    )
    return {
        "base_url": base_url.rstrip("/"),
        "session_id": session_id,
        "prompt": prompt,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "ok": all(step["ok"] for step in steps),
        "steps": steps,
    }


def _summarize_step(step: dict[str, Any]) -> list[str]:
    data = step.get("data") or {}
    lines = [f"### {step['name']}", "", f"- ok: `{step['ok']}`", f"- status: `{step.get('status')}`"]
    if not step["ok"]:
        lines.append(f"- error: `{step.get('error', '')[:300]}`")
        return lines
    if step["name"] == "health":
        lines.append(f"- service: `{data.get('status')}`")
        lines.append(f"- llm_provider: `{data.get('llm_provider')}`")
    elif step["name"] == "frontend_contract":
        endpoints = sorted((data.get("endpoints") or {}).keys())
        lines.append(f"- endpoints: `{', '.join(endpoints)}`")
    elif step["name"] == "ops_readiness":
        lines.append(f"- readiness: `{data.get('status')}`")
    elif step["name"] == "text_support":
        lines.append(f"- response_id: `{data.get('response_id')}`")
        lines.append(f"- risk: `{(data.get('risk') or {}).get('level')}`")
        lines.append(f"- entropy_score: `{(data.get('entropy') or {}).get('score')}`")
        reply = str(data.get("reply_text") or "").replace("\n", " ")
        lines.append(f"- reply_preview: {reply[:180]}")
    elif step["name"] == "student_role_view":
        latest = data.get("latest_response") or {}
        lines.append(f"- role: `{data.get('role')}`")
        lines.append(f"- has_reply_text: `{'reply_text' in latest}`")
    elif step["name"] == "care_queue":
        lines.append(f"- items: `{len(data.get('items') or [])}`")
        lines.append(f"- priority_counts: `{json.dumps(data.get('priority_counts') or {}, ensure_ascii=False)}`")
    return lines


def write_report(result: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{stamp}_frontend_handoff_smoke.md"
    lines = [
        "# Frontend Handoff Smoke Report",
        "",
        f"- base_url: `{result['base_url']}`",
        f"- session_id: `{result['session_id']}`",
        f"- generated_at: `{result['generated_at']}`",
        f"- overall_ok: `{result['ok']}`",
        "",
    ]
    for step in result["steps"]:
        lines.extend(_summarize_step(step))
        lines.append("")
    lines.append("## Raw Result")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(result, ensure_ascii=False, indent=2))
    lines.append("```")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the frontend/backend handoff endpoints.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--session-id", default="frontend-handoff-smoke")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--out-dir", default=str(ROOT / "reports" / "frontend_handoff_smoke"))
    args = parser.parse_args()

    result = run_smoke(args.base_url, args.session_id, args.prompt)
    report = write_report(result, Path(args.out_dir))
    print(json.dumps({"ok": result["ok"], "report": str(report)}, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
