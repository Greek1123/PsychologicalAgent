from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.processing_consistency import build_processing_consistency_report


class ProcessingConsistencyTests(unittest.TestCase):
    def test_clean_local_and_crisis_routes_are_consistent(self) -> None:
        report = build_processing_consistency_report(
            [
                {
                    "response_id": "resp-local",
                    "risk_level": "medium",
                    "referral_should_refer": False,
                    "processing_summary": {
                        "route": "local_policy",
                        "safety_priority": "standard",
                        "next_backend_action": "continue_supportive_monitoring",
                        "risk_level": "medium",
                        "reply_source": "local_policy",
                        "should_refer": False,
                        "referral_urgency": "none",
                    },
                },
                {
                    "response_id": "resp-crisis",
                    "risk_level": "critical",
                    "referral_should_refer": True,
                    "processing_summary": {
                        "route": "crisis_safety",
                        "safety_priority": "urgent",
                        "next_backend_action": "activate_urgent_handoff",
                        "risk_level": "critical",
                        "reply_source": "crisis_template",
                        "should_refer": True,
                        "referral_urgency": "urgent",
                    },
                },
            ]
        )

        self.assertEqual(report["summary"]["status"], "ok")
        self.assertEqual(report["summary"]["inconsistent_turns"], 0)
        self.assertTrue(all(item["consistent"] for item in report["timeline"]))

    def test_detects_high_risk_processing_mismatch(self) -> None:
        report = build_processing_consistency_report(
            [
                {
                    "response_id": "resp-bad",
                    "risk_level": "critical",
                    "referral_should_refer": True,
                    "referral_urgency": "urgent",
                    "processing_summary": {
                        "route": "local_policy",
                        "safety_priority": "standard",
                        "next_backend_action": "continue_supportive_monitoring",
                        "risk_level": "critical",
                        "reply_source": "local_policy",
                        "should_refer": True,
                        "referral_urgency": "urgent",
                    },
                }
            ]
        )

        issues = report["timeline"][0]["issues"]
        self.assertEqual(report["summary"]["status"], "needs_review")
        self.assertIn("high_risk_not_crisis_route", issues)
        self.assertIn("critical_risk_not_urgent", issues)
        self.assertIn("referral_marked_but_monitoring_only", issues)
        self.assertIn("urgent_referral_not_urgent_priority", issues)

    def test_marks_legacy_records_missing_processing_summary(self) -> None:
        report = build_processing_consistency_report([{"response_id": "legacy", "risk_level": "low"}])

        self.assertEqual(report["summary"]["status"], "needs_review")
        self.assertEqual(report["timeline"][0]["issues"], ["missing_processing_summary"])


if __name__ == "__main__":
    unittest.main()
