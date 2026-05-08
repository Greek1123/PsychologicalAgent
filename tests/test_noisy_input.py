from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.noisy_input import analyze_noisy_distress_text
from campus_support_agent.safety import evaluate_text_risk
from campus_support_agent.schemas import RiskLevel


class NoisyInputTests(unittest.TestCase):
    def test_crisis_typo_is_routed_as_critical(self) -> None:
        risk = evaluate_text_risk("我真的不想或了")

        self.assertEqual(risk.level, RiskLevel.CRITICAL)
        self.assertTrue(risk.needs_human_followup)
        self.assertIn("不想活", risk.trigger_terms)

    def test_near_sound_death_typo_is_routed_as_critical(self) -> None:
        risk = evaluate_text_risk("我好累，我想似")

        self.assertEqual(risk.level, RiskLevel.CRITICAL)
        self.assertIn("想死", risk.trigger_terms)

    def test_escalation_typo_is_routed_as_high(self) -> None:
        risk = evaluate_text_risk("我快奔溃了，真的撑不主")

        self.assertEqual(risk.level, RiskLevel.HIGH)
        self.assertTrue(risk.needs_human_followup)
        self.assertIn("快崩溃", risk.trigger_terms)

    def test_distress_typos_are_kept_for_analysis_text(self) -> None:
        analysis = analyze_noisy_distress_text("我鸭力好大，睡不召")

        self.assertTrue(analysis.has_inference)
        self.assertIn("压力", analysis.inferred_terms)
        self.assertIn("睡不着", analysis.inferred_terms)
        self.assertIn("系统推断关键词", analysis.analysis_text)


if __name__ == "__main__":
    unittest.main()
