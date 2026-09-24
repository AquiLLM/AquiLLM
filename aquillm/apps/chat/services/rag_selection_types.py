"""Immutable, internal inputs and result for direct-RAG evidence selection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from lib.retrieval.evidence import PreparedEvidence


@dataclass(frozen=True)
class SelectionCandidate:
    chunk_id: int
    doc_id: str
    chunk_number: int
    text: str
    relevance: float
    fused_rank: int
    source_fingerprint: str
    row: Mapping[str, object]
    prepared_evidence: PreparedEvidence | None = None
    token_cost: int | None = None


@dataclass(frozen=True)
class SelectionLimits:
    max_passages: int
    max_per_document: int
    token_budget: int


@dataclass(frozen=True)
class SelectionProfile:
    name: str
    relevance_weight: float
    gap_allowance: float
    version: str


@dataclass(frozen=True)
class EvidenceSelection:
    candidates: tuple[SelectionCandidate, ...]
    estimated_tokens: int
    profile: SelectionProfile
    score_status: str
