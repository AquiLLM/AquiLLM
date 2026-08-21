"""Strict validation and metric helpers for Task21 hybrid cloud observations."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType

from .task21_hybrid_live_trace import validate_candidate_trace

PARITY_KINDS = ("snapshot", "scores", "trace", "ties")


class Task21HybridEvalError(ValueError):
    """Raised when a Task21 cloud observation cannot support a measured claim."""


def exact_sha(value: object, context: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Task21HybridEvalError(f"{context} must be an exact SHA-256")
    return value


def exact_ids(value: object, context: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise Task21HybridEvalError(f"{context} must be a sequence")
    result = tuple(value)
    if any(type(item) is not str or not item for item in result):
        raise Task21HybridEvalError(f"{context} contains an invalid identifier")
    if len(set(result)) != len(result):
        raise Task21HybridEvalError(f"{context} must be unique")
    return result


def exact_number(value: object, context: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise Task21HybridEvalError(f"{context} must be finite")
    result = float(value)
    if result < 0:
        raise Task21HybridEvalError(f"{context} must be nonnegative")
    return result


def validate_cases(
    cases: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for case in cases:
        case_id = case.get("id")
        if type(case_id) is not str or not case_id or case_id in result:
            raise Task21HybridEvalError("case ids must be nonempty and unique")
        collections = set(
            exact_ids(case.get("accessible_collection_ids"), "collections")
        )
        accessible: set[str] = set()
        inaccessible_fixture: list[str] = []
        for document in case.get("documents", ()):
            if not isinstance(document, Mapping):
                raise Task21HybridEvalError("documents must be mappings")
            authorized = document.get("collection_id") in collections
            for chunk in document.get("chunks", ()):
                if not isinstance(chunk, Mapping):
                    raise Task21HybridEvalError("chunks must be mappings")
                chunk_id = chunk.get("chunk_id")
                if type(chunk_id) is not str or not chunk_id:
                    raise Task21HybridEvalError("fixture chunk id is invalid")
                if authorized:
                    accessible.add(chunk_id)
                else:
                    inaccessible_fixture.append(chunk_id)
        fixture_ids = exact_ids(
            tuple(sorted(inaccessible_fixture)), "inaccessible fixture chunks"
        )
        quality_tags = exact_ids(case.get("quality_tags", ()), "quality tags")
        privacy_intent = case.get("privacy_intent", "")
        if type(privacy_intent) is not str:
            raise Task21HybridEvalError("privacy intent must be exact text")
        privacy_required = bool(fixture_ids) or (
            "inaccessible_neighbor" in quality_tags
            or "security-sensitive" in privacy_intent.lower()
        )
        if privacy_required and not fixture_ids:
            raise Task21HybridEvalError(
                f"{case_id} intended privacy case requires its own privacy fixture"
            )
        expected = exact_ids(
            case.get("expected_retrieval_chunk_ids"), "expected chunks"
        )
        if not set(expected).issubset(accessible):
            raise Task21HybridEvalError("expected chunks must be authorized")
        raw_distances = case.get("expected_min_semantic_distance", {})
        if not isinstance(raw_distances, Mapping):
            raise Task21HybridEvalError("semantic distances must be a mapping")
        distances = {str(key): value for key, value in raw_distances.items()}
        if any(
            type(value) is not int or not 0 <= value <= 2
            for value in distances.values()
        ):
            raise Task21HybridEvalError("semantic distances must be integers in [0, 2]")
        result[case_id] = {
            "accessible": accessible,
            "inaccessible_fixture": fixture_ids,
            "expected": expected,
            "distances": distances,
            "quality_tags": quality_tags,
            "privacy_required": privacy_required,
        }
    if not result:
        raise Task21HybridEvalError("at least one case is required")
    return result


def validate_freshness(value: Mapping[str, object]) -> Mapping[str, object]:
    if set(value) != {
        "generation_key",
        "projection_checksum",
        "age_seconds",
        "max_age_seconds",
    }:
        raise Task21HybridEvalError("freshness fields are not exact")
    age = exact_number(value["age_seconds"], "projection age")
    maximum = exact_number(value["max_age_seconds"], "maximum projection age")
    if age > maximum:
        raise Task21HybridEvalError("projection freshness gate rejected stale data")
    return MappingProxyType(
        {
            "generation_key": exact_sha(value["generation_key"], "generation key"),
            "projection_checksum": exact_sha(
                value["projection_checksum"], "projection checksum"
            ),
            "age_seconds": age,
            "max_age_seconds": maximum,
            "valid": True,
        }
    )


def validate_parity(value: Mapping[str, object]) -> Mapping[str, object]:
    expected = {
        *(
            f"{backend}_{kind}_sha256"
            for backend in ("postgres", "memgraph")
            for kind in PARITY_KINDS
        ),
        "postgres_projected_ranks",
        "memgraph_projected_ranks",
    }
    if set(value) != expected:
        raise Task21HybridEvalError("backend parity fields are not exact")
    canonical = dict(value)
    for kind in PARITY_KINDS:
        postgres = exact_sha(value[f"postgres_{kind}_sha256"], f"PostgreSQL {kind}")
        memgraph = exact_sha(value[f"memgraph_{kind}_sha256"], f"Memgraph {kind}")
        if postgres != memgraph:
            raise Task21HybridEvalError(f"PostgreSQL/Memgraph {kind} parity failed")
    postgres_ranks = exact_ids(
        value["postgres_projected_ranks"], "PostgreSQL projected ranks"
    )
    memgraph_ranks = exact_ids(
        value["memgraph_projected_ranks"], "Memgraph projected ranks"
    )
    if postgres_ranks != memgraph_ranks:
        raise Task21HybridEvalError("PostgreSQL/Memgraph projected rank parity failed")
    canonical.update(
        postgres_projected_ranks=postgres_ranks,
        memgraph_projected_ranks=memgraph_ranks,
        valid=True,
    )
    return MappingProxyType(canonical)


def score_case(
    case: Mapping[str, object], observation: Mapping[str, object], *, arm: str
) -> dict[str, object]:
    ranked = exact_ids(observation["ranked_chunk_ids"], "ranked chunks")
    try:
        candidates = validate_candidate_trace(observation["candidate_trace"], arm=arm)
    except ValueError as error:
        raise Task21HybridEvalError("candidate trace is invalid") from error
    candidate_ids = exact_ids(
        tuple(candidate["chunk_id"] for candidate in candidates),
        "candidate trace chunks",
    )
    if candidate_ids != ranked:
        raise Task21HybridEvalError("candidate trace order differs from ranked chunks")
    graph = exact_ids(observation["graph_chunk_ids"], "graph chunks")
    citations = set(exact_ids(observation["citation_evidence_chunk_ids"], "citations"))
    seeds = set(exact_ids(observation["seed_chunk_ids"], "seed chunks"))
    mapped = set(exact_ids(observation["mapped_seed_chunk_ids"], "mapped seeds"))
    accessible = set(case["accessible"])
    adversarial = exact_ids(
        observation["adversarial_candidate_chunk_ids"], "adversarial candidates"
    )
    inaccessible = exact_ids(
        observation["inaccessible_result_chunk_ids"], "inaccessible results"
    )
    observed_inaccessible = tuple(item for item in ranked if item not in accessible)
    if adversarial != case["inaccessible_fixture"]:
        raise Task21HybridEvalError(
            "adversarial candidates differ from the inaccessible fixture"
        )
    if inaccessible != observed_inaccessible:
        raise Task21HybridEvalError("inaccessible result observations are inconsistent")
    if set(adversarial).intersection(accessible):
        raise Task21HybridEvalError("adversarial candidates must be inaccessible")
    if observed_inaccessible:
        count = len(observed_inaccessible)
        raise Task21HybridEvalError(
            f"authorization observed {count} inaccessible result(s)"
        )
    if not set(graph).issubset(set(ranked)) or not mapped.issubset(seeds):
        raise Task21HybridEvalError("graph/seed observations are inconsistent")
    expected = set(case["expected"])
    top = ranked[:10]
    recall = 1.0 if not expected else len(expected.intersection(top)) / len(expected)
    dcg = sum(
        1 / math.log2(index + 1)
        for index, item in enumerate(top, 1)
        if item in expected
    )
    ideal = sum(
        1 / math.log2(index + 1) for index in range(1, min(10, len(expected)) + 1)
    )
    novel = tuple(item for item in graph if item not in seeds)
    distances = case["distances"]
    return {
        "ranked_chunk_ids": ranked,
        "graph_chunk_ids": graph,
        "recall_at_10": recall,
        "ndcg_at_10": 1.0 if not ideal else dcg / ideal,
        "graph_hit_rate": 1.0 if graph else 0.0,
        "adversarial_candidate_count": len(adversarial),
        "inaccessible_result_count": len(observed_inaccessible),
        "citation_evidence_coverage": 1.0
        if not ranked
        else len(set(ranked).intersection(citations)) / len(ranked),
        "seed_coverage": 1.0
        if not seeds
        else len(seeds.intersection(mapped)) / len(seeds),
        "distance_2_novel_fraction": 0.0
        if not novel
        else sum(distances.get(item) == 2 for item in novel) / len(novel),
        "latency_ms": exact_number(observation["latency_ms"], "latency"),
        "quality_tags": case["quality_tags"],
    }
