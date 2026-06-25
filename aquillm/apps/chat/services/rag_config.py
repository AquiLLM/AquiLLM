"""Environment-backed configuration for direct RAG and tool-loop fallbacks."""
from __future__ import annotations

from os import getenv


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (getenv(name, "") or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    try:
        value = int(getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


def is_direct_rag_enabled() -> bool:
    return _env_bool("RAG_DIRECT_ENABLED", default=False)


def attach_tools_when_collections_selected() -> bool:
    return _env_bool("RAG_ATTACH_TOOLS_WHEN_COLLECTIONS_SELECTED", default=True)


def direct_rag_top_k() -> int:
    return _env_int("RAG_DIRECT_TOP_K", 10, minimum=1)


def evidence_token_budget() -> int:
    return _env_int("RAG_EVIDENCE_TOKEN_BUDGET", 3500, minimum=256)


def synthesis_max_tokens() -> int:
    return _env_int("RAG_SYNTHESIS_MAX_TOKENS", 8192, minimum=256)


def max_snippets_per_doc() -> int:
    return _env_int("RAG_MAX_SNIPPETS_PER_DOC", 3, minimum=1)


def max_figures_per_turn() -> int:
    return _env_int("RAG_MAX_FIGURES_PER_TURN", 3, minimum=0)


def tool_default_top_k() -> int:
    return _env_int("RAG_TOOL_DEFAULT_TOP_K", 10, minimum=1)


def query_rewrite_enabled() -> bool:
    return _env_bool("RAG_QUERY_REWRITE_ENABLED", default=False)


def direct_stage_logs_enabled() -> bool:
    return _env_bool("RAG_DIRECT_STAGE_LOGS", default=True)


__all__ = [
    "attach_tools_when_collections_selected",
    "direct_rag_top_k",
    "direct_stage_logs_enabled",
    "evidence_token_budget",
    "is_direct_rag_enabled",
    "max_figures_per_turn",
    "max_snippets_per_doc",
    "query_rewrite_enabled",
    "synthesis_max_tokens",
    "tool_default_top_k",
]
