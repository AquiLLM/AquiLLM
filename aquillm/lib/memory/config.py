"""
Memory system configuration from environment variables.
"""

import logging
from os import getenv

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int, min_value: int = 1) -> int:
    """Parse integer from environment with validation."""
    try:
        value = int(getenv(name, str(default)).strip())
    except Exception:
        return default
    return value if value >= min_value else min_value


EPISODIC_TOP_K = _env_int("EPISODIC_TOP_K", 5, min_value=1)
EPISODIC_MEMORY_MAX_CHARS = _env_int("EPISODIC_MEMORY_MAX_CHARS", 240, min_value=80)
MEMORY_BACKEND = getenv("MEMORY_BACKEND", "local").strip().lower()
MEM0_DUAL_WRITE_LOCAL = getenv("MEM0_DUAL_WRITE_LOCAL", "1").strip().lower() in ("1", "true", "yes", "on")
MEM0_TIMEOUT_SECONDS = _env_int("MEM0_TIMEOUT_SECONDS", 5, min_value=1)


def use_mem0() -> bool:
    """Check if Mem0 backend is configured."""
    return MEMORY_BACKEND == "mem0"


__all__ = [
    'EPISODIC_TOP_K',
    'EPISODIC_MEMORY_MAX_CHARS',
    'MEMORY_BACKEND',
    'MEM0_DUAL_WRITE_LOCAL',
    'MEM0_TIMEOUT_SECONDS',
    'use_mem0',
]
