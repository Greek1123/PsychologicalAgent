from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.config import Settings
from campus_support_agent.deployment_readiness import build_deployment_readiness


def test_deployment_readiness_passes_for_mock_provider(tmp_path: Path) -> None:
    kb_path = tmp_path / "campus_knowledge.json"
    kb_path.write_text("[]", encoding="utf-8")
    settings = Settings(
        llm_provider="mock",
        stt_provider="mock",
        database_path=str(tmp_path / "agent.db"),
        log_file_path=str(tmp_path / "app.log"),
        campus_kb_path=str(kb_path),
    )

    readiness = build_deployment_readiness(settings)

    assert readiness["status"] == "ready"
    assert readiness["summary"]["fail"] == 0
    assert {check["name"] for check in readiness["checks"]} >= {
        "llm_provider",
        "stt_provider",
        "database_path",
        "log_file_path",
        "campus_kb_path",
        "python_runtime",
    }


def test_deployment_readiness_blocks_missing_local_checkpoint(tmp_path: Path) -> None:
    kb_path = tmp_path / "campus_knowledge.json"
    kb_path.write_text("[]", encoding="utf-8")
    settings = Settings(
        llm_provider="local_checkpoint",
        stt_provider="mock",
        database_path=str(tmp_path / "agent.db"),
        log_file_path=str(tmp_path / "app.log"),
        campus_kb_path=str(kb_path),
        local_checkpoint_path=str(tmp_path / "missing-checkpoint"),
        local_base_model_path=str(tmp_path / "missing-base"),
    )

    readiness = build_deployment_readiness(settings)
    local_check = next(check for check in readiness["checks"] if check["name"] == "local_checkpoint")

    assert readiness["status"] == "blocked"
    assert local_check["status"] == "fail"
    assert local_check["details"]["missing"] == ["LOCAL_CHECKPOINT_PATH", "LOCAL_BASE_MODEL_PATH"]
