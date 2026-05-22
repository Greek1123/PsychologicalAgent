from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.agent import CampusSupportAgent
from campus_support_agent.config import Settings
from campus_support_agent.providers import MockLLMProvider, MockSTTProvider
from campus_support_agent.retrieval import CampusKnowledgeRetriever
from campus_support_agent.strategy_layer_evaluator import DEFAULT_STRATEGY_EVAL_CASES, evaluate_strategy_layer


class StrategyLayerEvaluatorTests(unittest.TestCase):
    def test_default_strategy_cases_pass(self) -> None:
        settings = Settings(llm_provider="mock")
        agent = CampusSupportAgent(
            settings=settings,
            llm_provider=MockLLMProvider(),
            stt_provider=MockSTTProvider(),
            retriever=CampusKnowledgeRetriever(settings),
        )

        results = evaluate_strategy_layer(agent)
        failures = {item["case_id"]: item["failures"] for item in results if not item["passed"]}

        self.assertEqual(len(results), len(DEFAULT_STRATEGY_EVAL_CASES))
        self.assertEqual(failures, {})


if __name__ == "__main__":
    unittest.main()
