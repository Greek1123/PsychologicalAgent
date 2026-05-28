from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.acceptance_report import build_acceptance_report, find_latest_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Markdown acceptance report for demos and defense.")
    parser.add_argument("--out", default=str(ROOT / "docs" / "system_acceptance_report.md"))
    parser.add_argument("--test-summary", default="not run in this report")
    args = parser.parse_args()

    artifacts = find_latest_artifacts(ROOT, test_summary=args.test_summary)
    report = build_acceptance_report(artifacts=artifacts)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
