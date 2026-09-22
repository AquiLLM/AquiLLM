"""Immutable, authorization-bound numerical rerank results."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import isfinite
from typing import Literal
from uuid import UUID

type ScoringKind = Literal["pointwise", "listwise", "rank_only"]
type ScoreStatus = Literal["complete", "unavailable"]
# Current hydrated identity, not a cache grant. The last field is the actual
# successful pair fingerprint, which may differ from the canonical prepared pair.
type AuthorizedScoreIdentity = tuple[int, UUID, int, str, str]


def fingerprint_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def fingerprint_pair(query: str, document: str) -> str:
    return fingerprint_text(
        json.dumps((query, document), ensure_ascii=False, separators=(",", ":"))
    )


def fingerprint_pool(candidate_identities: tuple[tuple[int, str, str], ...]) -> str:
    return fingerprint_text(json.dumps(candidate_identities, separators=(",", ":")))


def score_identity_for_chunk(
    chunk, *, effective_pair_fingerprint: str
) -> AuthorizedScoreIdentity:
    """Bind an authorized, current row to its exact successful scorer input."""
    return (
        chunk.pk,
        chunk.doc_id,
        chunk.chunk_number,
        fingerprint_text(chunk.content),
        effective_pair_fingerprint,
    )


@dataclass(frozen=True)
class PassageScore:
    chunk_pk: int
    document_id: UUID
    chunk_number: int
    source_fingerprint: str
    effective_pair_fingerprint: str
    value: float


@dataclass(frozen=True)
class RerankScoreSet:
    schema_version: str
    query_fingerprint: str
    scorer_fingerprint: str
    pool_fingerprint: str
    scoring_kind: ScoringKind
    status: ScoreStatus
    candidate_order: tuple[int, ...]
    scores: tuple[PassageScore, ...]


@dataclass(frozen=True)
class ScoredRerankResult:
    ranked_ids: tuple[int, ...]
    score_set: RerankScoreSet


def rank_only_result_for_chunks(query: str, chunks, top_k: int) -> ScoredRerankResult:
    """Represent an upstream skipped rerank without inventing relevance values."""
    rows = tuple(chunks)
    candidate_order = tuple(chunk.pk for chunk in rows)
    identities = tuple(
        (
            chunk.pk,
            fingerprint_text(chunk.content),
            fingerprint_pair(query, chunk.content),
        )
        for chunk in rows
    )
    return ScoredRerankResult(
        candidate_order[:top_k],
        RerankScoreSet(
            "v2",
            fingerprint_text(query),
            "",
            fingerprint_pool(identities),
            "rank_only",
            "unavailable",
            candidate_order,
            (),
        ),
    )


def order_scored_pairs(
    pairs: tuple[tuple[int, float], ...], candidate_ids: tuple[int, ...]
) -> tuple[tuple[int, float], ...]:
    """Sort complete indexed scores with original candidate order as tie break."""
    if len(pairs) != len(candidate_ids) or len(set(candidate_ids)) != len(
        candidate_ids
    ):
        raise ValueError("score pairs must cover unique candidates")
    seen = set()
    ordered = []
    for index, value in pairs:
        if (
            type(index) is not int
            or index < 0
            or index >= len(candidate_ids)
            or index in seen
        ):
            raise ValueError("invalid or duplicate score index")
        if type(value) not in (int, float) or not isfinite(float(value)):
            raise ValueError("score must be finite and numeric")
        seen.add(index)
        ordered.append((index, float(value)))
    if seen != set(range(len(candidate_ids))):
        raise ValueError("score pairs must cover every candidate")
    ordered.sort(key=lambda pair: (-pair[1], pair[0]))
    return tuple((candidate_ids[index], value) for index, value in ordered)


def validate_score_set(
    score_set: RerankScoreSet,
    *,
    authorized_identities: tuple[AuthorizedScoreIdentity, ...],
    expected_query_fingerprint: str,
    expected_scorer_fingerprint: str,
) -> RerankScoreSet:
    """Reject cached/model metadata that is stale or outside the current scope."""
    if not isinstance(score_set, RerankScoreSet) or score_set.schema_version != "v2":
        raise ValueError("unsupported score set")
    if (
        score_set.query_fingerprint != expected_query_fingerprint
        or score_set.scorer_fingerprint != expected_scorer_fingerprint
    ):
        raise ValueError("stale score set")
    if score_set.scoring_kind not in (
        "pointwise",
        "listwise",
        "rank_only",
    ) or score_set.status not in ("complete", "unavailable"):
        raise ValueError("invalid score set status")
    if score_set.scoring_kind == "rank_only" and score_set.status != "unavailable":
        raise ValueError("rank-only score set must be unavailable")
    identities = tuple(authorized_identities)
    if any(
        not isinstance(identity, tuple)
        or len(identity) != 5
        or type(identity[0]) is not int
        or identity[0] <= 0
        or not isinstance(identity[1], UUID)
        or type(identity[2]) is not int
        or identity[2] < 0
        or not isinstance(identity[3], str)
        or not identity[3]
        or not isinstance(identity[4], str)
        or not identity[4]
        for identity in identities
    ):
        raise ValueError("invalid authorized identity")
    by_id = {identity[0]: identity for identity in identities}
    if len(by_id) != len(identities) or len(set(score_set.candidate_order)) != len(
        score_set.candidate_order
    ):
        raise ValueError("duplicate candidates")
    if any(type(pk) is not int or pk not in by_id for pk in score_set.candidate_order):
        raise ValueError("unrelated candidates")
    if score_set.status == "unavailable" or score_set.scoring_kind == "rank_only":
        if score_set.scores:
            raise ValueError("unavailable scores must be empty")
        return score_set
    seen: set[int] = set()
    for score in score_set.scores:
        if (
            not isinstance(score, PassageScore)
            or type(score.value) not in (int, float)
            or not isfinite(float(score.value))
            or not isinstance(score.document_id, UUID)
            or type(score.chunk_number) is not int
        ):
            raise ValueError("invalid score")
        pk = score.chunk_pk
        if type(pk) is not int or pk in seen or pk not in score_set.candidate_order:
            raise ValueError("duplicate or unrelated score")
        seen.add(pk)
        if (
            pk,
            score.document_id,
            score.chunk_number,
            score.source_fingerprint,
            score.effective_pair_fingerprint,
        ) != by_id[pk]:
            raise ValueError("stale score identity")
    if seen != set(score_set.candidate_order):
        raise ValueError("incomplete scores")
    return score_set


__all__ = [
    "AuthorizedScoreIdentity",
    "PassageScore",
    "RerankScoreSet",
    "ScoredRerankResult",
    "fingerprint_pair",
    "fingerprint_pool",
    "fingerprint_text",
    "order_scored_pairs",
    "rank_only_result_for_chunks",
    "score_identity_for_chunk",
    "validate_score_set",
]
