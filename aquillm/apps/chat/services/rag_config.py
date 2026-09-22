"""Environment-backed configuration for direct RAG and tool-loop fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
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
    return min(15, _env_int("RAG_DIRECT_TOP_K", 10, minimum=1))


def direct_rag_candidate_top_k() -> int:
    """Leave room for document balancing within the retrieval tool's 15-row limit."""
    final_limit = direct_rag_top_k()
    return min(15, final_limit * 3) if final_limit > 1 else 1


def direct_rag_max_queries() -> int:
    return min(3, _env_int("RAG_DIRECT_MAX_QUERIES", 3, minimum=1))


def evidence_token_budget() -> int:
    return _env_int("RAG_EVIDENCE_TOKEN_BUDGET", 3500, minimum=256)


def synthesis_max_tokens() -> int:
    return _env_int("RAG_SYNTHESIS_MAX_TOKENS", 4096, minimum=256)


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


@dataclass(frozen=True, slots=True)
class EvidenceSelectionConfig:
    mode: str = "legacy"
    score_timeout_ms: int = 3000
    shadow_scoring: bool = False
    error: str | None = None


def evidence_selection_config() -> EvidenceSelectionConfig:
    """Read one validated rollout configuration; malformed values retain legacy."""
    mode = getenv("RAG_EVIDENCE_SELECTION_MODE", "legacy").strip().lower()
    shadow = getenv("RAG_EVIDENCE_SELECTION_SHADOW_SCORING", "0").strip().lower()
    truthy = ("1", "true", "yes", "on")
    falsy = ("0", "false", "no", "off")
    try:
        timeout = int(getenv("RAG_EVIDENCE_SELECTION_SCORE_TIMEOUT_MS", "3000"))
        if (
            mode not in ("legacy", "shadow", "adaptive")
            or not 100 <= timeout <= 3000
            or shadow not in (*truthy, *falsy)
        ):
            raise ValueError("invalid evidence selection configuration")
    except (TypeError, ValueError):
        return EvidenceSelectionConfig(
            error="invalid_evidence_selection_configuration",
        )
    return EvidenceSelectionConfig(mode, timeout, shadow in truthy)


def selection_mode() -> str:
    return evidence_selection_config().mode


def selection_scoring_timeout_ms() -> int:
    return evidence_selection_config().score_timeout_ms


def shadow_scoring_enabled() -> bool:
    return evidence_selection_config().shadow_scoring


__all__ = [
    "attach_tools_when_collections_selected",
    "direct_rag_top_k",
    "direct_rag_candidate_top_k",
    "direct_rag_max_queries",
    "direct_stage_logs_enabled",
    "evidence_token_budget",
    "EvidenceSelectionConfig",
    "evidence_selection_config",
    "selection_mode",
    "selection_scoring_timeout_ms",
    "shadow_scoring_enabled",
    "is_direct_rag_enabled",
    "max_figures_per_turn",
    "max_snippets_per_doc",
    "query_rewrite_enabled",
    "synthesis_max_tokens",
    "tool_default_top_k",
]
