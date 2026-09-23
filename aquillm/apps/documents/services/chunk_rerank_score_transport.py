"""Strict private transport for authorized rerank score envelopes."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from uuid import UUID

from apps.documents.services.chunk_rerank_results import (
    PassageScore,
    RerankScoreSet,
    WindowCoverage,
)


def _window_fields(score):
    coverage = score.window_coverage
    if coverage is None:
        return {}
    return {
        "window_coverage": {
            "required_window_ids": list(coverage.required_window_ids),
            "required_pair_fingerprints": list(coverage.required_pair_fingerprints),
            "successful_pair_fingerprints": list(coverage.successful_pair_fingerprints),
            "prepared_input_scoring_coverage": coverage.prepared_input_scoring_coverage,
            "aggregation_version": coverage.aggregation_version,
        }
    }


def _parse_window(raw):
    if not isinstance(raw, Mapping) or set(raw) != {
        "required_window_ids",
        "required_pair_fingerprints",
        "successful_pair_fingerprints",
        "prepared_input_scoring_coverage",
        "aggregation_version",
    }:
        raise ValueError("invalid window coverage")
    arrays = tuple(
        raw[k]
        for k in (
            "required_window_ids",
            "required_pair_fingerprints",
            "successful_pair_fingerprints",
        )
    )
    if any(
        type(a) is not list or len(a) > 135 or any(type(x) is not str for x in a)
        for a in arrays
    ):
        raise ValueError("invalid window identities")
    coverage = WindowCoverage(
        *(tuple(a) for a in arrays),
        raw["prepared_input_scoring_coverage"],
        raw["aggregation_version"],
    )
    if not coverage.is_complete():
        raise ValueError("incomplete window coverage")
    return coverage


def serialize_score_set(score_set: RerankScoreSet) -> dict[str, object]:
    """Return JSON-compatible scalar data for the private tool sidecar."""
    return {
        "schema_version": score_set.schema_version,
        "query_fingerprint": score_set.query_fingerprint,
        "scorer_fingerprint": score_set.scorer_fingerprint,
        "pool_fingerprint": score_set.pool_fingerprint,
        "scoring_kind": score_set.scoring_kind,
        "status": score_set.status,
        "candidate_order": list(score_set.candidate_order),
        "scores": [
            {
                "chunk_pk": score.chunk_pk,
                "document_id": str(score.document_id),
                "chunk_number": score.chunk_number,
                "source_fingerprint": score.source_fingerprint,
                "effective_pair_fingerprint": score.effective_pair_fingerprint,
                "value": score.value,
                **_window_fields(score),
            }
            for score in score_set.scores
        ],
    }


def deserialize_score_set(value: object) -> RerankScoreSet | None:
    """Reject malformed or extraneous score fields without exposing their values."""
    if type(value) is RerankScoreSet:
        try:
            value = serialize_score_set(value)
        except (AttributeError, TypeError, ValueError):
            return None
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "query_fingerprint",
        "scorer_fingerprint",
        "pool_fingerprint",
        "scoring_kind",
        "status",
        "candidate_order",
        "scores",
    }:
        return None
    try:
        strings = tuple(
            value[key]
            for key in (
                "schema_version",
                "query_fingerprint",
                "scorer_fingerprint",
                "pool_fingerprint",
                "scoring_kind",
                "status",
            )
        )
        if any(type(item) is not str or not item for item in strings):
            return None
        schema, query, scorer, pool, kind, status = strings
        if schema not in ("v2", "v3-window") or kind not in (
            "pointwise",
            "listwise",
            "rank_only",
        ):
            return None
        if status not in ("complete", "unavailable"):
            return None
        if kind == "rank_only" and status == "complete":
            return None
        order = value["candidate_order"]
        raw_scores = value["scores"]
        if type(order) is not list or type(raw_scores) is not list:
            return None
        if any(type(pk) is not int or pk <= 0 for pk in order):
            return None
        if len(set(order)) != len(order):
            return None
        scores: list[PassageScore] = []
        for raw in raw_scores:
            expected = {
                "chunk_pk",
                "document_id",
                "chunk_number",
                "source_fingerprint",
                "effective_pair_fingerprint",
                "value",
            }
            if schema == "v3-window":
                expected.add("window_coverage")
            if not isinstance(raw, Mapping) or set(raw) != expected:
                return None
            pk, number, amount = raw["chunk_pk"], raw["chunk_number"], raw["value"]
            if (
                type(pk) is not int
                or pk <= 0
                or type(number) is not int
                or number < 0
                or type(amount) not in (int, float)
                or not isfinite(float(amount))
                or type(raw["document_id"]) is not str
                or type(raw["source_fingerprint"]) is not str
                or type(raw["effective_pair_fingerprint"]) is not str
            ):
                return None
            scores.append(
                PassageScore(
                    pk,
                    UUID(raw["document_id"]),
                    number,
                    raw["source_fingerprint"],
                    raw["effective_pair_fingerprint"],
                    float(amount),
                    _parse_window(raw["window_coverage"])
                    if schema == "v3-window"
                    else None,
                )
            )
        if len({score.chunk_pk for score in scores}) != len(scores):
            return None
        if any(score.chunk_pk not in order for score in scores):
            return None
        if status == "complete" and kind != "rank_only":
            if {score.chunk_pk for score in scores} != set(order):
                return None
        elif scores:
            return None
        return RerankScoreSet(
            schema, query, scorer, pool, kind, status, tuple(order), tuple(scores)
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


__all__ = ["deserialize_score_set", "serialize_score_set"]
