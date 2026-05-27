from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.config import Settings
from campus_support_agent.deployment_readiness import build_deployment_readiness


def main() -> int:
    readiness = build_deployment_readiness(Settings())
    print(json.dumps(readiness, ensure_ascii=False, indent=2))
    return 0 if readiness["status"] in {"ready", "degraded"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
