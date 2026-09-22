"""Vector-seeded extended branch for the production hybrid runtime."""

from __future__ import annotations

from collections import defaultdict
from math import fsum, isfinite

from apps.knowledge_graph.retrieval.branch_contracts import ExtendedBranchFailureReason
from apps.knowledge_graph.retrieval.projected_ppr import ppr_projected_v1
from apps.knowledge_graph.retrieval.scheduler_support import (
    LocalBranchSchedulerFailure,
    failed_branch,
)
from apps.knowledge_graph.retrieval.topology.contracts import (
    HybridBranchKind,
    ProjectedSeedV1,
)

from .production_runtime_support import (
    graph_candidates,
    ppr_config,
    ppr_failure_envelope,
    success_envelope,
    topology_caps,
)


def _projection_repository(runtime):
    if runtime.projection_repository_factory is not None:
        return runtime.projection_repository_factory()
    from .extended_seed_repository import ExtendedSeedRepository

    return ExtendedSeedRepository(using="projection_source")


def _source_failure(error):
    raise LocalBranchSchedulerFailure(
        HybridBranchKind.EXTENDED,
        ExtendedBranchFailureReason.EXTENDED_SEED_INVALID,
    ) from error


def prepare_extended_branch(
    runtime, *, baseline, shared, authorization, settings, deadline
):
    runtime._exact_request(authorization, settings)
    scope = runtime._shared_scope(shared)
    if settings.graph_extended_enabled is not True:
        return ExtendedBranchFailureReason.EXTENDED_NO_SEEDS
    graph_seeds = getattr(baseline, "graph_seeds", None)
    candidates = getattr(baseline, "baseline_candidates", None)
    if (
        type(graph_seeds) is not tuple
        or type(candidates) is not tuple
        or not graph_seeds
    ):
        return ExtendedBranchFailureReason.EXTENDED_NO_SEEDS
    if runtime.clock() >= deadline:
        return ExtendedBranchFailureReason.EXTENDED_TOPOLOGY_TIMEOUT
    by_pk = {getattr(row, "pk", None): row for row in candidates}
    authority_by_document = {
        document: row
        for row in scope.projections
        for document, _artifact in row.documents
    }
    requested: dict[int, float] = {}
    chunks_by_projection = defaultdict(list)
    for seed in graph_seeds[: settings.graph_extended_max_seeds]:
        candidate = by_pk.get(getattr(seed, "chunk_id", None))
        authority = authority_by_document.get(getattr(candidate, "doc_id", None))
        weight = getattr(seed, "restart_weight", None)
        if (
            authority is None
            or type(weight) is not float
            or not isfinite(weight)
            or weight <= 0.0
        ):
            return ExtendedBranchFailureReason.EXTENDED_SEED_INVALID
        if candidate.pk in requested:
            return ExtendedBranchFailureReason.EXTENDED_SEED_INVALID
        requested[candidate.pk] = weight
        chunks_by_projection[authority.projection_id].append(
            (candidate.pk, candidate.doc_id)
        )
    identities: dict[str, list[float]] = defaultdict(list)
    try:
        repository = _projection_repository(runtime)
    except Exception as error:
        _source_failure(error)
    for authority in scope.projections:
        chunks = tuple(chunks_by_projection.get(authority.projection_id, ()))
        if not chunks:
            continue
        if runtime.clock() >= deadline:
            return ExtendedBranchFailureReason.EXTENDED_TOPOLOGY_TIMEOUT
        try:
            by_chunk = repository.load_seed_identities(
                authority=authority,
                chunks=chunks,
                authorization=authorization,
                codec=runtime.codec,
                max_rows=min(4999, getattr(settings, "graph_extended_max_nodes", 4999)),
            )
            if set(by_chunk) - {pk for pk, _ in chunks}:
                raise ValueError("extended source returned unrelated chunks")
        except Exception as error:
            _source_failure(error)
        if runtime.clock() >= deadline:
            return ExtendedBranchFailureReason.EXTENDED_TOPOLOGY_TIMEOUT
        for chunk_id, entity_keys in by_chunk.items():
            if not entity_keys:
                continue
            share = requested[chunk_id] / len(entity_keys)
            for identity_key in entity_keys:
                identities[identity_key].append(share)
    if not identities:
        return ExtendedBranchFailureReason.EXTENDED_NO_SEEDS
    masses = {key: fsum(values) for key, values in identities.items()}
    selected = sorted(masses, key=lambda key: (-masses[key], key))[
        : settings.graph_extended_max_seeds
    ]
    total = fsum(masses[key] for key in selected)
    return tuple(
        sorted(
            (ProjectedSeedV1(key, masses[key] / total) for key in selected),
            key=lambda row: row.identity_key,
        )
    )


def run_extended_branch(
    runtime, *, prepared, shared, authorization, settings, deadline
):
    runtime._exact_request(authorization, settings)
    scope, started = runtime._shared_scope(shared), runtime.clock()
    if type(prepared) is ExtendedBranchFailureReason:
        return failed_branch(HybridBranchKind.EXTENDED, prepared)
    seeds = prepared
    caps = topology_caps(settings, HybridBranchKind.EXTENDED)
    snapshot = runtime.topology_loader.load(
        ready=scope.ready, seeds=seeds, caps=caps, deadline=deadline
    )
    try:
        result = ppr_projected_v1(
            snapshot=snapshot,
            seeds=seeds,
            config=ppr_config(snapshot, caps.max_results),
        )
        candidates = graph_candidates(
            snapshot=snapshot,
            identity_scores=result.scores,
            maximum=caps.max_results,
        )
    except (TypeError, ValueError):
        return ppr_failure_envelope(
            HybridBranchKind.EXTENDED,
            ExtendedBranchFailureReason.EXTENDED_PPR_INVALID,
            seed_count=len(seeds),
            snapshot=snapshot,
            elapsed_ms=min(
                settings.graph_extended_timeout_ms,
                int((runtime.clock() - started) * 1000),
            ),
        )
    return success_envelope(
        HybridBranchKind.EXTENDED,
        ready=scope.ready,
        seeds=seeds,
        snapshot=snapshot,
        candidates=candidates,
        settings=settings,
        elapsed_ms=max(0, int((runtime.clock() - started) * 1000)),
    )


__all__ = ["prepare_extended_branch", "run_extended_branch"]
