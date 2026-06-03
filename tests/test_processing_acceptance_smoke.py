from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import smoke_processing_acceptance as smoke


class ProcessingAcceptanceSmokeTests(unittest.TestCase):
    def test_validate_case_accepts_crisis_escalation(self) -> None:
        case = {
            "case_id": "dangerous_place_escalation",
            "expected": {
                "risk_levels": {"critical"},
                "routes": {"crisis_safety"},
                "safety_priorities": {"urgent"},
                "latest_action": "activate_urgent_handoff",
            },
        }
        turns = [
            {
                "ok": True,
                "data": {
                    "risk": {"level": "critical"},
                    "processing_summary": {
                        "route": "crisis_safety",
                        "safety_priority": "urgent",
                        "next_backend_action": "activate_urgent_handoff",
                    },
                },
            }
        ]
        analysis = {
            "ok": True,
            "data": {
                "processing_consistency": {"summary": {"status": "ok"}},
            },
        }

        self.assertEqual(smoke._validate_case(case, turns, analysis), [])

    def test_validate_case_flags_route_mismatch(self) -> None:
        case = {
            "case_id": "dangerous_place_escalation",
            "expected": {
                "risk_levels": {"critical"},
                "routes": {"crisis_safety"},
                "safety_priorities": {"urgent"},
            },
        }
        turns = [
            {
                "ok": True,
                "data": {
                    "risk": {"level": "critical"},
                    "processing_summary": {
                        "route": "local_policy",
                        "safety_priority": "standard",
                    },
                },
            }
        ]
        analysis = {
            "ok": True,
            "data": {
                "processing_consistency": {"summary": {"status": "needs_review"}},
            },
        }

        errors = smoke._validate_case(case, turns, analysis)

        self.assertIn("unexpected_latest_route:local_policy", errors)
        self.assertIn("unexpected_latest_safety_priority:standard", errors)
        self.assertIn("processing_consistency_not_ok:needs_review", errors)

    def test_write_report_creates_markdown_and_json(self) -> None:
        result = {
            "base_url": "http://127.0.0.1:8000",
            "generated_at": "2026-06-03T12:00:00",
            "ok": True,
            "setup_steps": [],
            "cases": [
                {
                    "case_id": "exam_sleep_pressure",
                    "ok": True,
                    "turns": [
                        {
                            "data": {
                                "risk": {"level": "medium"},
                                "processing_summary": {
                                    "route": "local_policy",
                                    "safety_priority": "standard",
                                    "next_backend_action": "continue_supportive_monitoring",
                                },
                            }
                        }
                    ],
                    "analysis": {
                        "data": {
                            "processing_summary": {
                                "latest_route": "local_policy",
                                "latest_next_backend_action": "continue_supportive_monitoring",
                            }
                        }
                    },
                    "validation_errors": [],
                }
            ],
            "overview": {"ok": True},
            "processing_health": {
                "ok": True,
                "data": {
                    "status": "ok",
                    "recommended_next_action": "continue_development_or_frontend_integration",
                    "blocking_issues": [],
                    "watch_items": [],
                },
            },
            "overall_errors": [],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            md_path, json_path = smoke.write_report(result, Path(tmp_dir))

            self.assertTrue(md_path.exists())
            self.assertTrue(json_path.exists())
            self.assertIn("Processing Acceptance Smoke Report", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
