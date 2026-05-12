"""Read the project version from pyproject.toml once at import time."""
from __future__ import annotations

import tomllib
from pathlib import Path

import structlog

logger = structlog.stdlib.get_logger(__name__)


def _load() -> str:
    pyproject = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
    try:
        with pyproject.open("rb") as f:
            return tomllib.load(f)["project"]["version"]
    except (OSError, KeyError, tomllib.TOMLDecodeError) as exc:
        logger.warning("Could not read app version from %s: %s", pyproject, exc)
        return ""


APP_VERSION = _load()
