"""Explicit logging configuration without import-time side effects."""

from __future__ import annotations

import logging
import os


def configure_logging(level: str | None = None) -> None:
    """Configure process logging from an argument, environment, or safe default."""
    resolved = (level or os.getenv("SHOPEE_MATCH_LOG_LEVEL") or "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, resolved, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
