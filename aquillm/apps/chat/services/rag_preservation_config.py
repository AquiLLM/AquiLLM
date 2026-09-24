"""Independent, default-off evidence-preservation rollout switches.

Parsing is strict so a malformed opt-in cannot silently enable a different
experiment. Runtime selector compatibility is enforced by the controller in
Task 3; this config also accepts an explicit availability check for callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from os import getenv

_TRUE = frozenset(("1", "true", "yes", "on"))
_FALSE = frozenset(("0", "false", "no", "off"))


def _choice(name: str, default: str, allowed: set[str]) -> str:
    value = getenv(name, default).strip().lower()
    if value not in allowed:
        raise ValueError(name)
    return value


def _bool(name: str) -> bool:
    value = getenv(name, "0").strip().lower()
    if value not in _TRUE | _FALSE:
        raise ValueError(name)
    return value in _TRUE


@dataclass(frozen=True, slots=True)
class PreservationConfig:
    rerank_text_mode: str = "legacy"
    evidence_text_mode: str = "legacy"
    document_capacity_mode: str = "legacy"
    document_hard_cap: int = 0
    followup_evidence_enabled: bool = False
    iterative_retrieval_enabled: bool = False
    shadow_scoring: bool = False
    error: str | None = None

    @property
    def active(self) -> bool:
        return self.error is None and (
            self.rerank_text_mode == "windowed"
            or self.evidence_text_mode == "source"
            or self.document_capacity_mode == "budgeted"
            or self.followup_evidence_enabled
            or self.iterative_retrieval_enabled
            or (self.rerank_text_mode == "shadow" and self.shadow_scoring)
        )

    def document_cap(
        self, *, final_passage_limit: int, legacy_per_doc_limit: int
    ) -> int:
        if final_passage_limit <= 0 or legacy_per_doc_limit <= 0:
            raise ValueError("passage limits must be positive")
        if self.document_capacity_mode == "legacy":
            return min(final_passage_limit, legacy_per_doc_limit)
        return min(final_passage_limit, self.document_hard_cap or final_passage_limit)


def preservation_config(
    *, shared_selector_available: bool = True
) -> PreservationConfig:
    try:
        rerank = _choice(
            "RAG_RERANK_TEXT_MODE", "legacy", {"legacy", "shadow", "windowed"}
        )
        evidence = _choice("RAG_EVIDENCE_TEXT_MODE", "legacy", {"legacy", "source"})
        capacity = _choice(
            "RAG_DOCUMENT_CAPACITY_MODE", "legacy", {"legacy", "budgeted"}
        )
        raw_cap = getenv("RAG_DOCUMENT_HARD_CAP", "0").strip()
        hard_cap = int(raw_cap)
        if hard_cap < 0:
            raise ValueError("RAG_DOCUMENT_HARD_CAP")
        followup = _bool("RAG_FOLLOWUP_EVIDENCE_ENABLED")
        iterative = _bool("RAG_ITERATIVE_RETRIEVAL_ENABLED")
        shadow = _bool("RAG_RERANK_SHADOW_SCORING_ENABLED")
        if (followup or iterative) and evidence != "source":
            raise ValueError(
                "source evidence required for follow-up or iterative retrieval"
            )
        if shadow and rerank != "shadow":
            raise ValueError("shadow scoring requires shadow rerank mode")
        result = PreservationConfig(
            rerank, evidence, capacity, hard_cap, followup, iterative, shadow
        )
        from .rag_config import evidence_selection_config

        if result.active and evidence_selection_config().error:
            raise ValueError("invalid shared selector configuration")
        if (
            result.active
            and getenv("RAG_EVIDENCE_SELECTION_MODE", "legacy").strip().lower()
            == "shadow"
        ):
            raise ValueError("active preservation requires a served shared selector")
        if result.active and not shared_selector_available:
            raise ValueError("shared selector unavailable")
        return result
    except (TypeError, ValueError):
        return PreservationConfig(error="invalid_rag_preservation_configuration")


__all__ = ["PreservationConfig", "preservation_config"]
