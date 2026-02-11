"""
Logging configuration utilities.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from ..config import get_settings


def configure_logging() -> None:
    """Configure application-wide logging."""
    settings = get_settings()
    log_dir = settings.logs_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "forecast.log"

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Avoid duplicate handlers on repeated configuration
    if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        root.addHandler(handler)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(console)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return configured logger."""
    configure_logging()
    return logging.getLogger(name)


__all__ = ["get_logger", "configure_logging"]

