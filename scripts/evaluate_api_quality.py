from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_chat_quality import GLOBAL_FORBIDDEN_TERMS, SCENARIOS


def _post_json(url: str, payload: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"API request failed: HTTP {exc.code} - {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"API request failed: {exc.reason}") from exc


def _evaluate_api(base_url: str, timeout_seconds: int, pause_seconds: float) -> list[dict[str, Any]]:
    endpoint = f"{base_url.rstrip('/')}/api/v1/support/text"
    results: list[dict[str, Any]] = []

    for scenario in SCENARIOS:
        session_id = f"eval-{scenario.case_id}-{uuid4().hex[:8]}"
        replies: list[str] = []
        failures: list[str] = []

        for turn in scenario.turns:
            payload = {
                "session_id": session_id,
                "text": turn.user,
                "student_context": {"source": "api_quality_eval"},
            }
            response = _post_json(endpoint, payload, timeout_seconds)
            reply = str(response.get("reply_text", "")).strip()
            replies.append(reply)
            if pause_seconds > 0:
                time.sleep(pause_seconds)

        combined = "\n".join(replies)
        for term in scenario.expected_terms:
            if term not in combined:
                failures.append(f"missing expected term: {term}")
        for term in (*GLOBAL_FORBIDDEN_TERMS, *scenario.forbidden_terms):
            if term in combined:
                failures.append(f"contains forbidden term: {term}")
        if replies and len(replies[-1]) < scenario.min_last_reply_chars:
            failures.append(f"last reply too short: {len(replies[-1])} chars")

        results.append(
            {
                "case_id": scenario.case_id,
                "title": scenario.title,
                "passed": not failures,
                "failures": failures,
                "turns": [
                    {
                        "user": turn.user,
                        "reply": reply,
                    }
                    for turn, reply in zip(scenario.turns, replies, strict=True)
                ],
            }
        )

    return results


def _write_report(results: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": {
            "total": len(results),
            "passed": sum(1 for item in results if item["passed"]),
            "failed": sum(1 for item in results if not item["passed"]),
        },
        "results": results,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _print_summary(results: list[dict[str, Any]], output: Path) -> None:
    passed = sum(1 for item in results if item["passed"])
    total = len(results)
    print(f"API quality evaluation: {passed}/{total} passed")
    for item in results:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"[{status}] {item['case_id']} - {item['title']}")
        for failure in item["failures"]:
            print(f"  - {failure}")
    print(f"Report written to: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the running FastAPI support/text endpoint.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--pause-seconds", type=float, default=0.0)
    parser.add_argument("--output", default=str(ROOT / "reports" / "api_quality_eval.json"))
    args = parser.parse_args()

    results = _evaluate_api(args.base_url, args.timeout_seconds, args.pause_seconds)
    output = Path(args.output)
    _write_report(results, output)
    _print_summary(results, output)

    if any(not item["passed"] for item in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
