from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOGGER_NAME = "campus_support_agent"


def configure_logging(settings: object) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(_resolve_level(getattr(settings, "log_level", "INFO")))

    # 同时输出到控制台和文件，便于开发排错与留痕。
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        log_path = Path(str(getattr(settings, "log_file_path", "logs/app.log")))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def _resolve_level(level_name: str) -> int:
    return getattr(logging, str(level_name).upper(), logging.INFO)
