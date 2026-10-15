"""Direct-query seed preparation for the production hybrid runtime."""

from __future__ import annotations

from apps.knowledge_graph.retrieval.branch_contracts import (
    DirectBranchFailureReason,
)
from apps.knowledge_graph.retrieval.direct_seed_contracts import (
    DirectResolutionSpanInputV1,
)
from apps.knowledge_graph.retrieval.direct_seed_repository import (
    DirectSeedRepository,
    DirectSeedScopeV1,
)
from apps.knowledge_graph.retrieval.direct_seed_resolution import (
    resolve_direct_seed_components,
)
from apps.knowledge_graph.retrieval.direct_seed_sql import DirectSeedReadTimeout
from apps.knowledge_graph.retrieval.ppr_policy import classify_ppr_intent
from apps.knowledge_graph.retrieval.ppr_seed_support import (
    PreparedPPRSeedsV1,
    summarize_direct_support,
)
from apps.knowledge_graph.retrieval.scheduler_support import (
    LocalBranchSchedulerFailure,
)
from apps.knowledge_graph.retrieval.stage_diagnostics import graph_stage
from apps.knowledge_graph.retrieval.topology.contracts import (
    HybridBranchKind,
    ProjectedSeedV1,
)
from lib.knowledge_graph.query_extractor.client import (
    QueryExtractorClientError,
    reconstruct_entity_texts,
)


def _local(reason, error):
    raise LocalBranchSchedulerFailure(HybridBranchKind.DIRECT, reason) from error


def _prepare_direct(runtime, *, query, scope, deadline, with_policy, read_budget=None, metadata=None, group=None):
    from apps.knowledge_graph.retrieval.query_ontology import load_query_ontology

    if runtime.clock() >= deadline:
        return DirectBranchFailureReason.DIRECT_BRANCH_TIMEOUT
    artifact_ids = tuple(sorted({row.artifact_id for row in scope.projections}))
    try:
        with graph_stage(branch="direct", stage="ontology", clock=runtime.clock):
            if metadata is not None:
                from .native_query_ontology import load_native_ontology
                ontology_outcome = load_native_ontology(metadata, group=group)
            else:
                ontology_outcome = load_query_ontology(
                    selected_artifact_ids=artifact_ids,
                    using=runtime.authorization.database_alias,
                    **({"read_budget": read_budget} if read_budget is not None else {}),
                )
    except Exception as error:
        _local(DirectBranchFailureReason.DIRECT_SEED_INVALID, error)
    if runtime.clock() >= deadline:
        return DirectBranchFailureReason.DIRECT_BRANCH_TIMEOUT
    ontology = ontology_outcome.ontology
    expected_ontologies = {
        (row.ontology_version, row.ontology_checksum) for row in scope.projections
    }
    if (
        ontology is None
        or len(expected_ontologies) != 1
        or (ontology.version, ontology.checksum) != next(iter(expected_ontologies))
    ):
        return DirectBranchFailureReason.MIXED_ONTOLOGY
    try:
        with graph_stage(branch="direct", stage="extraction", clock=runtime.clock):
            response = runtime._extractor(scope=scope, ontology=ontology).extract(
                query=query, ontology=ontology, deadline=deadline
            )
    except QueryExtractorClientError as error:
        return DirectBranchFailureReason(error.reason.value)
    except Exception as error:
        _local(DirectBranchFailureReason.EXTRACTOR_PROVENANCE, error)
    try:
        surfaces = reconstruct_entity_texts(query=query, response=response)
        span_inputs = tuple(
            DirectResolutionSpanInputV1(span, text)
            for span, text in zip(response.spans, surfaces, strict=True)
        )
    except (TypeError, ValueError) as error:
        _local(DirectBranchFailureReason.EXTRACTOR_PROVENANCE, error)
    generation_by_projection = dict(scope.generation_keys_by_projection)
    generation_by_artifact = tuple(
        sorted(
            (row.artifact_id, generation_by_projection[row.projection_id])
            for row in scope.projections
        )
    )
    try:
        with graph_stage(
            branch="direct", stage="entity_resolution", clock=runtime.clock
        ):
            direct_scope = DirectSeedScopeV1(
                scope.ready.bundle_checksum,
                tuple(sorted(row.collection_id for row in scope.projections)),
                artifact_ids,
                scope.selected_document_ids,
                tuple(
                    sorted(
                        artifact
                        for row in scope.projections
                        for _document, artifact in row.documents
                    )
                ),
                generation_by_artifact,
                tuple(
                    sorted(
                        (row.artifact_id, row.generation_id)
                        for row in scope.projections
                    )
                ),
                ontology.checksum,
                scope.projections[0].resolver_version,
                scope.projections[0].embedding_model_signature,
            )
            from .native_direct_seed_repository import _NativeDirectSeedRepository
            repository_type = DirectSeedRepository if metadata is None else _NativeDirectSeedRepository
            repository = repository_type(
                **({} if metadata is None else {"metadata": metadata, "group": group}),
                scope=direct_scope,
                codec=runtime.codec,
                span_inputs=span_inputs,
                deadline=deadline,
                clock=runtime.clock,
                using=runtime.authorization.database_alias,
                **({"read_budget": read_budget} if read_budget is not None else {}),
            )
            outcome = resolve_direct_seed_components(
                spans=response.spans,
                repository=repository,
                ready=scope.ready,
                settings=runtime.settings,
                deadline=deadline,
                clock=runtime.clock,
            )
    except DirectSeedReadTimeout:
        return DirectBranchFailureReason.DIRECT_BRANCH_TIMEOUT
    except Exception as error:
        _local(DirectBranchFailureReason.DIRECT_SEED_INVALID, error)
    if outcome.failure_reason is not None:
        return DirectBranchFailureReason(outcome.failure_reason.value)
    seeds = tuple(
        sorted(
            (ProjectedSeedV1(row.component_key, row.mass) for row in outcome.seeds),
            key=lambda row: row.identity_key,
        )
    )
    if not with_policy:
        return seeds
    return PreparedPPRSeedsV1(
        seeds,
        summarize_direct_support(
            outcome, max_seeds=runtime.settings.graph_direct_max_seeds
        ),
        classify_ppr_intent(query),
    )


def prepare_direct_seeds(runtime, *, query, scope, deadline, read_budget=None):
    return _prepare_direct(
        runtime,
        query=query,
        scope=scope,
        deadline=deadline,
        with_policy=False,
        read_budget=read_budget,
    )


def prepare_direct_with_policy(runtime, *, query, scope, deadline):
    return _prepare_direct(
        runtime, query=query, scope=scope, deadline=deadline, with_policy=True
    )


__all__ = ["prepare_direct_seeds", "prepare_direct_with_policy"]

_NATIVE_METADATA_DEPENDENCIES = (_prepare_direct, DirectSeedRepository, resolve_direct_seed_components, reconstruct_entity_texts)

_NATIVE_PREPARE_DIRECT_SEEDS = prepare_direct_seeds
