"""Split live execution facts into frozen metric and trace artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .task21_hybrid_eval import TASK21_HYBRID_ARMS
from .task21_hybrid_live_trace import (
    validate_live_trace,
    validate_live_trace_observations,
)

_OBSERVATION_FIELDS = (
    "case_id",
    "ranked_chunk_ids",
    "graph_chunk_ids",
    "citation_evidence_chunk_ids",
    "seed_chunk_ids",
    "mapped_seed_chunk_ids",
    "projected_ranks",
    "repeated_projected_ranks",
    "adversarial_candidate_chunk_ids",
    "inaccessible_result_chunk_ids",
    "latency_ms",
    "reranker_calls",
    "comparison_snapshot_signature",
)
_TRACE_FIELDS = (
    "case_id",
    "candidate_trace",
    "timing_trace",
    "authorization_status",
    "graph_scheduled",
    "inaccessible_candidate_count",
)
_FRESHNESS_FIELDS = (
    "generation_key",
    "projection_checksum",
    "age_seconds",
    "max_age_seconds",
)
_FRESHNESS_TRACE_FIELDS = (
    "projection_keys",
    "generation_keys",
    "graph_checksums",
    "ready_bundle_checksums",
    "ontology_version",
    "ontology_checksum",
)
_PARITY_FIELDS = tuple(
    f"{backend}_{kind}_sha256"
    for backend in ("postgres", "memgraph")
    for kind in ("snapshot", "scores", "trace", "ties")
) + ("postgres_projected_ranks", "memgraph_projected_ranks")


@dataclass(frozen=True, slots=True)
class LiveEvidencePayloads:
    observations: dict[str, list[dict[str, object]]]
    freshness: dict[str, object]
    backend_parity: dict[str, object]
    live_trace: dict[str, object]
    case_ids: tuple[str, ...]
    fixture_chunk_ids: tuple[str, ...]


def _exact_subset(row, fields, context):
    if not isinstance(row, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return {field: row[field] for field in fields}


def _split_observations(combined):
    if not isinstance(combined, Mapping) or tuple(combined) != TASK21_HYBRID_ARMS:
        raise ValueError("combined observations require the exact five arms")
    expected = set(_OBSERVATION_FIELDS) | set(_TRACE_FIELDS)
    observations = {arm: [] for arm in TASK21_HYBRID_ARMS}
    traces = {arm: [] for arm in TASK21_HYBRID_ARMS}
    for arm in TASK21_HYBRID_ARMS:
        for row in combined[arm]:
            if not isinstance(row, Mapping) or set(row) != expected:
                raise ValueError("combined live observation fields are not exact")
            observations[arm].append(_exact_subset(row, _OBSERVATION_FIELDS, arm))
            traces[arm].append(_exact_subset(row, _TRACE_FIELDS, arm))
    case_ids = tuple(row["case_id"] for row in observations["vector_only"])
    return observations, traces, case_ids


def build_live_evidence_payloads(
    *,
    run_id: str,
    source_commit: str,
    manifest,
    combined_observations,
    combined_freshness,
    combined_backend_parity,
) -> LiveEvidencePayloads:
    """Create exact evaluator rows plus their closed live-trace companion."""

    observations, arms, case_ids = _split_observations(combined_observations)
    if set(combined_freshness) != set(
        _FRESHNESS_FIELDS + _FRESHNESS_TRACE_FIELDS
    ):
        raise ValueError("combined live freshness fields are not exact")
    if set(combined_backend_parity) != set(_PARITY_FIELDS) | {"comparison_inputs"}:
        raise ValueError("combined live parity fields are not exact")
    freshness = _exact_subset(
        combined_freshness, _FRESHNESS_FIELDS, "live freshness"
    )
    backend_parity = _exact_subset(
        combined_backend_parity, _PARITY_FIELDS, "live parity"
    )
    fixture_chunk_ids = tuple(manifest.chunks)
    trace = {
        "schema": "task21-hybrid-live-trace-v1",
        "run_id": run_id,
        "source_commit": source_commit,
        "fixture_id": manifest.fixture_id,
        "fixture_checksum": manifest.fixture_checksum,
        "manifest_checksum": manifest.manifest_checksum,
        "arms": arms,
        "freshness_attestation": _exact_subset(
            combined_freshness, _FRESHNESS_TRACE_FIELDS, "freshness trace"
        ),
        "backend_parity_inputs": combined_backend_parity["comparison_inputs"],
    }
    validate_live_trace(
        trace,
        expected_case_ids=case_ids,
        fixture_chunk_ids=fixture_chunk_ids,
    )
    validate_live_trace_observations(trace, observations)
    return LiveEvidencePayloads(
        observations,
        freshness,
        backend_parity,
        trace,
        case_ids,
        fixture_chunk_ids,
    )


__all__ = ["LiveEvidencePayloads", "build_live_evidence_payloads"]
