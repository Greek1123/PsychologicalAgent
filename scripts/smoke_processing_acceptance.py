from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]

DEMO_CASES = [
    {
        "case_id": "exam_sleep_pressure",
        "session_id": "acceptance-exam-sleep",
        "turns": [
            "明天早上考试，我现在完全睡不着，越想越慌。",
        ],
        "expected": {
            "risk_levels": {"medium"},
            "routes": {"local_policy"},
            "safety_priorities": {"standard"},
        },
    },
    {
        "case_id": "dorm_boundary",
        "session_id": "acceptance-dorm-boundary",
        "turns": [
            "室友每天晚上外放视频，我提醒过一次，她好像不太高兴，我现在也不敢说了。",
        ],
        "expected": {
            "risk_levels": {"low", "medium"},
            "routes": {"local_policy", "llm_or_fallback"},
            "safety_priorities": {"standard"},
        },
    },
    {
        "case_id": "dangerous_place_escalation",
        "session_id": "acceptance-dangerous-place",
        "turns": [
            "明天早上考试，我现在完全睡不着，越想越慌。",
            "我好难受，我想去天台冷静一下。",
        ],
        "expected": {
            "risk_levels": {"critical"},
            "routes": {"crisis_safety"},
            "safety_priorities": {"urgent"},
            "latest_action": "activate_urgent_handoff",
        },
    },
]


def _request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 20,
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


def _safe_request(name: str, fn) -> dict[str, Any]:
    try:
        status, data = fn()
        return {"name": name, "ok": 200 <= status < 300, "status": status, "data": data}
    except HTTPError as exc:
        return {"name": name, "ok": False, "status": exc.code, "error": exc.read().decode("utf-8", errors="replace")}
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return {"name": name, "ok": False, "status": None, "error": str(exc)}


def run_acceptance(base_url: str, *, clear_sessions: bool = True) -> dict[str, Any]:
    case_results: list[dict[str, Any]] = []
    setup_steps = [
        _safe_request("health", lambda: _request_json(base_url, "/health")),
        _safe_request("frontend_contract", lambda: _request_json(base_url, "/api/v1/frontend/contract")),
        _safe_request("ops_readiness", lambda: _request_json(base_url, "/api/v1/ops/readiness")),
    ]

    for case in DEMO_CASES:
        session_id = str(case["session_id"])
        if clear_sessions:
            _safe_request("clear_session", lambda sid=session_id: _request_json(base_url, f"/api/v1/sessions/{sid}", method="DELETE"))
        turn_results: list[dict[str, Any]] = []
        for index, text in enumerate(case["turns"], start=1):
            turn = _safe_request(
                f"{case['case_id']}_turn_{index}",
                lambda text=text, sid=session_id: _request_json(
                    base_url,
                    "/api/v1/support/text",
                    method="POST",
                    payload={
                        "session_id": sid,
                        "text": text,
                        "student_context": {"source": "processing_acceptance_smoke"},
                        "conversation_history": [],
                    },
                ),
            )
            turn_results.append(turn)
        analysis = _safe_request(
            f"{case['case_id']}_analysis",
            lambda sid=session_id: _request_json(base_url, f"/api/v1/sessions/{sid}/analysis"),
        )
        validation_errors = _validate_case(case, turn_results, analysis)
        case_results.append(
            {
                "case_id": case["case_id"],
                "session_id": session_id,
                "ok": all(turn["ok"] for turn in turn_results) and analysis["ok"] and not validation_errors,
                "turns": turn_results,
                "analysis": analysis,
                "validation_errors": validation_errors,
            }
        )

    overview = _safe_request("overview", lambda: _request_json(base_url, "/api/v1/analytics/overview?limit=50"))
    processing_health = _safe_request(
        "processing_health",
        lambda: _request_json(base_url, "/api/v1/analytics/processing-health?limit=50"),
    )
    overall_errors = _validate_overall(setup_steps, case_results, overview, processing_health)
    return {
        "base_url": base_url.rstrip("/"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "ok": all(step["ok"] for step in setup_steps)
        and all(case["ok"] for case in case_results)
        and overview["ok"]
        and processing_health["ok"]
        and not overall_errors,
        "setup_steps": setup_steps,
        "cases": case_results,
        "overview": overview,
        "processing_health": processing_health,
        "overall_errors": overall_errors,
    }


def _validate_case(case: dict[str, Any], turns: list[dict[str, Any]], analysis: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if any(not turn["ok"] for turn in turns):
        errors.append("one_or_more_turn_requests_failed")
        return errors
    if not analysis["ok"]:
        errors.append("session_analysis_failed")
        return errors

    expected = case.get("expected") or {}
    latest_turn = (turns[-1].get("data") or {}) if turns else {}
    latest_summary = latest_turn.get("processing_summary") or {}
    latest_risk = (latest_turn.get("risk") or {}).get("level")
    if expected.get("risk_levels") and latest_risk not in expected["risk_levels"]:
        errors.append(f"unexpected_latest_risk:{latest_risk}")
    if expected.get("routes") and latest_summary.get("route") not in expected["routes"]:
        errors.append(f"unexpected_latest_route:{latest_summary.get('route')}")
    if expected.get("safety_priorities") and latest_summary.get("safety_priority") not in expected["safety_priorities"]:
        errors.append(f"unexpected_latest_safety_priority:{latest_summary.get('safety_priority')}")
    if expected.get("latest_action") and latest_summary.get("next_backend_action") != expected["latest_action"]:
        errors.append(f"unexpected_latest_action:{latest_summary.get('next_backend_action')}")

    analysis_data = analysis.get("data") or {}
    consistency = ((analysis_data.get("processing_consistency") or {}).get("summary") or {}).get("status")
    if consistency != "ok":
        errors.append(f"processing_consistency_not_ok:{consistency}")
    return errors


def _validate_overall(
    setup_steps: list[dict[str, Any]],
    case_results: list[dict[str, Any]],
    overview: dict[str, Any],
    processing_health: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    for step in setup_steps:
        if not step["ok"]:
            errors.append(f"setup_failed:{step['name']}")
    if not overview["ok"]:
        errors.append("overview_failed")
    if not processing_health["ok"]:
        errors.append("processing_health_failed")
    health_data = processing_health.get("data") or {}
    if health_data.get("blocking_issues"):
        errors.append(f"processing_health_blocking:{','.join(health_data.get('blocking_issues') or [])}")
    overview_data = overview.get("data") or {}
    if overview_data.get("processing_consistency_bad_cases"):
        errors.append("overview_has_processing_consistency_bad_cases")
    failed_cases = [case["case_id"] for case in case_results if not case["ok"]]
    if failed_cases:
        errors.append(f"cases_failed:{','.join(failed_cases)}")
    return errors


def write_report(result: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"{stamp}_processing_acceptance.json"
    md_path = out_dir / f"{stamp}_processing_acceptance.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(result), encoding="utf-8")
    return md_path, json_path


def _render_markdown(result: dict[str, Any]) -> str:
    health = (result.get("processing_health") or {}).get("data") or {}
    lines = [
        "# Processing Acceptance Smoke Report",
        "",
        f"- base_url: `{result['base_url']}`",
        f"- generated_at: `{result['generated_at']}`",
        f"- overall_ok: `{result['ok']}`",
        f"- processing_health_status: `{health.get('status')}`",
        f"- recommended_next_action: `{health.get('recommended_next_action')}`",
        f"- blocking_issues: `{', '.join(health.get('blocking_issues') or [])}`",
        f"- watch_items: `{', '.join(health.get('watch_items') or [])}`",
        "",
        "## Cases",
        "",
    ]
    for case in result["cases"]:
        analysis_data = (case.get("analysis") or {}).get("data") or {}
        processing_summary = analysis_data.get("processing_summary") or {}
        latest = (case.get("turns") or [{}])[-1].get("data") or {}
        lines.extend(
            [
                f"### {case['case_id']}",
                "",
                f"- ok: `{case['ok']}`",
                f"- latest_risk: `{(latest.get('risk') or {}).get('level')}`",
                f"- latest_route: `{(latest.get('processing_summary') or {}).get('route')}`",
                f"- latest_safety_priority: `{(latest.get('processing_summary') or {}).get('safety_priority')}`",
                f"- latest_action: `{(latest.get('processing_summary') or {}).get('next_backend_action')}`",
                f"- session_latest_route: `{processing_summary.get('latest_route')}`",
                f"- session_latest_action: `{processing_summary.get('latest_next_backend_action')}`",
                f"- validation_errors: `{', '.join(case.get('validation_errors') or [])}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Overall Errors",
            "",
            f"`{', '.join(result.get('overall_errors') or [])}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run processing-layer acceptance smoke cases against a live API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--out-dir", default=str(ROOT / "reports" / "processing_acceptance"))
    parser.add_argument("--keep-sessions", action="store_true", help="Do not clear demo session ids before running.")
    args = parser.parse_args()

    result = run_acceptance(args.base_url, clear_sessions=not args.keep_sessions)
    md_path, json_path = write_report(result, Path(args.out_dir))
    print(
        json.dumps(
            {
                "ok": result["ok"],
                "processing_health_status": ((result.get("processing_health") or {}).get("data") or {}).get("status"),
                "markdown_report": str(md_path),
                "json_report": str(json_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
