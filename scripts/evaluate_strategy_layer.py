from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.agent import CampusSupportAgent
from campus_support_agent.config import Settings
from campus_support_agent.providers import MockLLMProvider, MockSTTProvider
from campus_support_agent.retrieval import CampusKnowledgeRetriever
from campus_support_agent.strategy_layer_evaluator import evaluate_strategy_layer, write_strategy_eval_report


def _build_agent() -> CampusSupportAgent:
    settings = Settings(llm_provider="mock")
    return CampusSupportAgent(
        settings=settings,
        llm_provider=MockLLMProvider(),
        stt_provider=MockSTTProvider(),
        retriever=CampusKnowledgeRetriever(settings),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate state_profile -> intervention_strategy -> reply quality.")
    parser.add_argument("--out", default=str(ROOT / "reports" / "strategy_layer_eval.json"))
    parser.add_argument("--csv-out", default=str(ROOT / "reports" / "strategy_layer_eval.csv"))
    args = parser.parse_args()

    results = evaluate_strategy_layer(_build_agent())
    json_path = Path(args.out)
    csv_path = Path(args.csv_out) if args.csv_out else None
    write_strategy_eval_report(results, json_path, csv_path)

    passed = sum(1 for item in results if item["passed"])
    total = len(results)
    print(json.dumps({"passed": passed, "failed": total - passed, "total": total, "out": str(json_path)}, ensure_ascii=False))
    for item in results:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"[{status}] {item['case_id']} state={item['actual_primary_state']} strategy={item['actual_strategy_id']}")
        for failure in item["failures"]:
            print(f"  - {failure}")

    if passed != total:
        sys.exit(1)


if __name__ == "__main__":
    main()
