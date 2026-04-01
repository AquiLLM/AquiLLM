"""
Memory system configuration from environment variables.
"""

import structlog
from os import getenv

logger = structlog.stdlib.get_logger(__name__)


def _env_int(name: str, default: int, min_value: int = 1) -> int:
    """Parse integer from environment with validation."""
    try:
        value = int(getenv(name, str(default)).strip())
    except Exception:
        return default
    return value if value >= min_value else min_value


def _env_bool(name: str, default: bool) -> bool:
    """Parse boolean from environment variables."""
    raw = getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_optional_str(name: str) -> str | None:
    """Read optional string env value and normalize blanks to None."""
    value = (getenv(name, "") or "").strip()
    return value or None


def _env_optional_float(name: str) -> float | None:
    """Read optional float env value; invalid input returns None."""
    value = _env_optional_str(name)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        logger.warning("Invalid %s=%r; ignoring.", name, value)
        return None


EPISODIC_TOP_K = _env_int("EPISODIC_TOP_K", 5, min_value=1)
EPISODIC_MEMORY_MAX_CHARS = _env_int("EPISODIC_MEMORY_MAX_CHARS", 240, min_value=80)
MEMORY_BACKEND = getenv("MEMORY_BACKEND", "local").strip().lower()
MEM0_DUAL_WRITE_LOCAL = _env_bool("MEM0_DUAL_WRITE_LOCAL", True)
MEM0_TIMEOUT_SECONDS = _env_int("MEM0_TIMEOUT_SECONDS", 5, min_value=1)
MEM0_GRAPH_ENABLED = _env_bool("MEM0_GRAPH_ENABLED", False)
MEM0_GRAPH_PROVIDER = (getenv("MEM0_GRAPH_PROVIDER", "") or "").strip().lower()
MEM0_GRAPH_URL = (getenv("MEM0_GRAPH_URL", "bolt://memgraph:7687") or "").strip()
MEM0_GRAPH_USERNAME = (getenv("MEM0_GRAPH_USERNAME", "memgraph") or "").strip()
MEM0_GRAPH_PASSWORD = (getenv("MEM0_GRAPH_PASSWORD", "") or "").strip()
MEM0_GRAPH_DATABASE = _env_optional_str("MEM0_GRAPH_DATABASE")
MEM0_GRAPH_CUSTOM_PROMPT = _env_optional_str("MEM0_GRAPH_CUSTOM_PROMPT")
MEM0_GRAPH_THRESHOLD = _env_optional_float("MEM0_GRAPH_THRESHOLD")
MEM0_GRAPH_FAIL_OPEN = _env_bool("MEM0_GRAPH_FAIL_OPEN", True)
MEM0_GRAPH_ADD_ENABLED = _env_bool("MEM0_GRAPH_ADD_ENABLED", True)
MEM0_GRAPH_SEARCH_ENABLED = _env_bool("MEM0_GRAPH_SEARCH_ENABLED", True)


def use_mem0() -> bool:
    """Check if Mem0 backend is configured."""
    return MEMORY_BACKEND == "mem0"


__all__ = [
    'EPISODIC_TOP_K',
    'EPISODIC_MEMORY_MAX_CHARS',
    'MEMORY_BACKEND',
    'MEM0_DUAL_WRITE_LOCAL',
    'MEM0_TIMEOUT_SECONDS',
    'MEM0_GRAPH_ENABLED',
    'MEM0_GRAPH_PROVIDER',
    'MEM0_GRAPH_URL',
    'MEM0_GRAPH_USERNAME',
    'MEM0_GRAPH_PASSWORD',
    'MEM0_GRAPH_DATABASE',
    'MEM0_GRAPH_CUSTOM_PROMPT',
    'MEM0_GRAPH_THRESHOLD',
    'MEM0_GRAPH_FAIL_OPEN',
    'MEM0_GRAPH_ADD_ENABLED',
    'MEM0_GRAPH_SEARCH_ENABLED',
    'use_mem0',
]
