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
from apps.knowledge_graph.retrieval.scheduler_support import (
    LocalBranchSchedulerFailure,
)
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


def prepare_direct_seeds(runtime, *, query, scope, deadline):
    from apps.knowledge_graph.retrieval.query_ontology import load_query_ontology

    if runtime.clock() >= deadline:
        return DirectBranchFailureReason.EXTRACTOR_TIMEOUT
    artifact_ids = tuple(sorted({row.artifact_id for row in scope.projections}))
    try:
        ontology_outcome = load_query_ontology(
            selected_artifact_ids=artifact_ids,
            using=runtime.authorization.database_alias,
        )
    except Exception as error:
        _local(DirectBranchFailureReason.DIRECT_SEED_INVALID, error)
    if runtime.clock() >= deadline:
        return DirectBranchFailureReason.EXTRACTOR_TIMEOUT
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
            ontology.checksum,
            scope.projections[0].resolver_version,
            scope.projections[0].embedding_model_signature,
        )
        repository = DirectSeedRepository(
            scope=direct_scope,
            codec=runtime.codec,
            span_inputs=span_inputs,
            using=runtime.authorization.database_alias,
        )
        outcome = resolve_direct_seed_components(
            spans=response.spans,
            repository=repository,
            ready=scope.ready,
            settings=runtime.settings,
            deadline=deadline,
        )
    except Exception as error:
        _local(DirectBranchFailureReason.DIRECT_SEED_INVALID, error)
    if outcome.failure_reason is not None:
        return DirectBranchFailureReason(outcome.failure_reason.value)
    return tuple(
        sorted(
            (ProjectedSeedV1(row.component_key, row.mass) for row in outcome.seeds),
            key=lambda row: row.identity_key,
        )
    )


__all__ = ["prepare_direct_seeds"]
