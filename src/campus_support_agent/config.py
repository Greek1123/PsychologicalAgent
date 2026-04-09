from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


def load_env_file(path: Path = DEFAULT_ENV_FILE, *, override: bool = False) -> None:
    # 只做最小化的 .env 解析，避免额外依赖。
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        env_key = key.strip()
        env_value = value.strip()

        if not env_key:
            continue
        if not override and env_key in os.environ:
            continue
        if env_value and env_value[0] == env_value[-1] and env_value[0] in {"'", '"'}:
            env_value = env_value[1:-1]

        os.environ[env_key] = env_value


load_env_file()


def _split_csv(value: str | None, fallback: list[str]) -> list[str]:
    if not value:
        return fallback
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(slots=True)
class Settings:
    app_env: str = field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    app_port: int = field(default_factory=lambda: int(os.getenv("APP_PORT", "8000")))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_file_path: str = field(
        default_factory=lambda: os.getenv("LOG_FILE_PATH", str(PROJECT_ROOT / "logs" / "app.log"))
    )
    database_path: str = field(
        default_factory=lambda: os.getenv("DATABASE_PATH", str(PROJECT_ROOT / "data" / "campus_agent.db"))
    )

    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "mock"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "EmoLLM-2.0"))
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "http://127.0.0.1:23333/v1"))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_timeout_seconds: int = field(default_factory=lambda: int(os.getenv("LLM_TIMEOUT_SECONDS", "60")))

    stt_provider: str = field(default_factory=lambda: os.getenv("STT_PROVIDER", "mock"))
    stt_model: str = field(default_factory=lambda: os.getenv("STT_MODEL", "whisper-1"))
    stt_base_url: str = field(default_factory=lambda: os.getenv("STT_BASE_URL", "http://127.0.0.1:8001/v1"))
    stt_api_key: str = field(default_factory=lambda: os.getenv("STT_API_KEY", ""))
    stt_language: str = field(default_factory=lambda: os.getenv("STT_LANGUAGE", "zh"))

    campus_name: str = field(default_factory=lambda: os.getenv("CAMPUS_NAME", "示例大学"))
    campus_counseling_center: str = field(
        default_factory=lambda: os.getenv("CAMPUS_COUNSELING_CENTER", "学生心理中心")
    )
    campus_counseling_hotline: str = field(
        default_factory=lambda: os.getenv("CAMPUS_COUNSELING_HOTLINE", "请替换为你学校心理热线")
    )
    campus_counseling_email: str = field(
        default_factory=lambda: os.getenv("CAMPUS_COUNSELING_EMAIL", "psy-center@example.edu")
    )
    crisis_contacts: list[str] = field(
        default_factory=lambda: _split_csv(
            os.getenv("CRISIS_CONTACTS"),
            ["当地紧急电话", "校心理中心值班电话", "可信任辅导员/班主任/家长"],
        )
    )
    campus_kb_path: str = field(
        default_factory=lambda: os.getenv(
            "CAMPUS_KB_PATH",
            str(PROJECT_ROOT / "data" / "campus_knowledge.json"),
        )
    )

    support_language: str = "zh-CN"
    max_history_turns: int = 6
