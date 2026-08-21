"""Vector-seeded extended branch for the production hybrid runtime."""

from __future__ import annotations

from collections import defaultdict
from math import fsum

from apps.knowledge_graph.projection.identifiers import ProjectionIdentifierDomain
from apps.knowledge_graph.projection.serialization import projection_checksum
from apps.knowledge_graph.retrieval.branch_contracts import ExtendedBranchFailureReason
from apps.knowledge_graph.retrieval.projected_ppr import ppr_projected_v1
from apps.knowledge_graph.retrieval.scheduler_support import failed_branch
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
    from apps.knowledge_graph.projection.django_projection_source import (
        DjangoProjectionRowSource,
    )
    from apps.knowledge_graph.projection.postgres_repository import (
        PostgresProjectionRepository,
    )

    source = DjangoProjectionRowSource(
        runtime.authorization.database_alias,
        state_using="projection_state",
        identifier_key=(
            runtime.settings.projection_identifier_hmac_key.get_secret_value().encode()
        ),
        identifier_key_version=runtime.settings.projection_identifier_key_version,
        schema_version=runtime.settings.projection_schema_version,
        projection_version=runtime.settings.projection_format_version,
    )
    return PostgresProjectionRepository(using="projection_state", source=source)


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
    generation_by_projection = dict(scope.generation_keys_by_projection)
    requested: dict[str, float] = {}
    for seed in graph_seeds[: settings.graph_extended_max_seeds]:
        candidate = by_pk.get(getattr(seed, "chunk_id", None))
        authority = authority_by_document.get(getattr(candidate, "doc_id", None))
        weight = getattr(seed, "restart_weight", None)
        if authority is None or type(weight) is not float or weight <= 0.0:
            return ExtendedBranchFailureReason.EXTENDED_SEED_INVALID
        chunk_key = runtime.codec.encode(
            ProjectionIdentifierDomain.CHUNK,
            generation=authority.generation_id,
            source=candidate.pk,
        ).value
        if chunk_key in requested:
            return ExtendedBranchFailureReason.EXTENDED_SEED_INVALID
        requested[chunk_key] = weight
    identities: dict[str, list[float]] = defaultdict(list)
    repository = _projection_repository(runtime)
    for authority in scope.projections:
        if runtime.clock() >= deadline:
            return ExtendedBranchFailureReason.EXTENDED_TOPOLOGY_TIMEOUT
        bundle = repository.load_projection_bundle(
            projection_id=authority.projection_id,
            batch_size=runtime.settings.projection_batch_size,
            purpose="audit",
        )
        if runtime.clock() >= deadline:
            return ExtendedBranchFailureReason.EXTENDED_TOPOLOGY_TIMEOUT
        if (
            projection_checksum(bundle) != authority.graph_checksum
            or bundle.generation.generation_key
            != generation_by_projection[authority.projection_id]
        ):
            raise ValueError("extended seed projection provenance is stale")
        membership = {
            row.entity_key: row.automatic_membership_key or row.entity_key
            for row in bundle.automatic_memberships
        }
        by_chunk: dict[str, set[str]] = defaultdict(set)
        for mention in bundle.entity_mentions:
            if mention.chunk_key in requested:
                by_chunk[mention.chunk_key].add(membership[mention.entity_key])
        for chunk_key, entity_keys in by_chunk.items():
            share = requested[chunk_key] / len(entity_keys)
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
