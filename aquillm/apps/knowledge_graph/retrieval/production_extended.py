"""Vector-seeded extended branch for the production hybrid runtime."""

from __future__ import annotations

from collections import defaultdict
from math import fsum, isfinite

from apps.knowledge_graph.retrieval.branch_contracts import ExtendedBranchFailureReason
from apps.knowledge_graph.retrieval.ppr_policy import classify_ppr_intent
from apps.knowledge_graph.retrieval.ppr_seed_support import (
    PreparedPPRSeedsV1,
    summarize_extended_support,
)
from apps.knowledge_graph.retrieval.production_ppr_policy import rank_projected_for_mode
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

# Each projection's complete chunk-to-identity inputs have a bounded source
# budget; final seed selection and topology admission retain smaller caps.
_MAX_SEED_SOURCE_ROWS_PER_PROJECTION = 4_999


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


def _prepare_extended_branch(
    runtime, *, query, baseline, shared, authorization, settings, deadline, with_policy
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
    mapped_chunk_ids: set[int] = set()
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
                max_rows=_MAX_SEED_SOURCE_ROWS_PER_PROJECTION,
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
            mapped_chunk_ids.add(chunk_id)
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
    seeds = tuple(
        sorted(
            (ProjectedSeedV1(key, masses[key] / total) for key in selected),
            key=lambda row: row.identity_key,
        )
    )
    if not with_policy:
        return seeds
    support = summarize_extended_support(
        ranked_seeds=graph_seeds,
        mapped_chunk_ids=frozenset(mapped_chunk_ids),
        vector_chunk_ids=getattr(baseline, "vector_chunk_ids", None),
        trigram_chunk_ids=getattr(baseline, "trigram_chunk_ids", None),
        exact_chunk_ids=getattr(baseline, "exact_chunk_ids", None),
        retained_identity_count=len(seeds),
        max_seeds=settings.graph_extended_max_seeds,
    )
    return PreparedPPRSeedsV1(seeds, support, classify_ppr_intent(query))


def prepare_extended_branch(
    runtime, *, baseline, shared, authorization, settings, deadline
):
    return _prepare_extended_branch(
        runtime,
        query="",
        baseline=baseline,
        shared=shared,
        authorization=authorization,
        settings=settings,
        deadline=deadline,
        with_policy=False,
    )


def prepare_extended_with_policy(
    runtime, *, query, baseline, shared, authorization, settings, deadline
):
    return _prepare_extended_branch(
        runtime,
        query=query,
        baseline=baseline,
        shared=shared,
        authorization=authorization,
        settings=settings,
        deadline=deadline,
        with_policy=True,
    )


def run_extended_branch(
    runtime, *, prepared, shared, authorization, settings, deadline
):
    runtime._exact_request(authorization, settings)
    scope, started = runtime._shared_scope(shared), runtime.clock()
    if type(prepared) is ExtendedBranchFailureReason:
        return failed_branch(HybridBranchKind.EXTENDED, prepared)
    mode = getattr(settings, "ppr_restart_mode", "fixed")
    if mode != "fixed" and type(prepared) is not PreparedPPRSeedsV1:
        return failed_branch(
            HybridBranchKind.EXTENDED, ExtendedBranchFailureReason.EXTENDED_SEED_INVALID
        )
    seeds = prepared if mode == "fixed" else prepared.seeds
    caps = topology_caps(settings, HybridBranchKind.EXTENDED)
    snapshot = runtime.topology_loader.load(
        ready=scope.ready, seeds=seeds, caps=caps, deadline=deadline
    )
    try:
        result, execution_signature = rank_projected_for_mode(
            snapshot=snapshot,
            seeds=seeds,
            config=ppr_config(snapshot, caps.max_results),
            prepared=None if mode == "fixed" else prepared,
            mode=mode,
            branch=HybridBranchKind.EXTENDED,
            deadline_check=lambda: runtime._check_branch_deadline(deadline),
        )
        candidates = graph_candidates(
            snapshot=snapshot,
            identity_scores=result.scores,
            maximum=caps.max_results,
        )
    except TimeoutError:
        return ppr_failure_envelope(
            HybridBranchKind.EXTENDED,
            ExtendedBranchFailureReason.EXTENDED_TOPOLOGY_TIMEOUT,
            seed_count=len(seeds),
            snapshot=snapshot,
            elapsed_ms=settings.graph_extended_timeout_ms,
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
        execution_algorithm_signature=execution_signature,
    )


__all__ = [
    "prepare_extended_branch",
    "prepare_extended_with_policy",
    "run_extended_branch",
]
