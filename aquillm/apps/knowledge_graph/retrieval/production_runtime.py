"""Request-bound production runtime for both projected graph branches."""

from __future__ import annotations

from pathlib import Path
from time import monotonic

from apps.collections.services.retrieval_authorization import (
    RetrievalAuthorizationContext,
)
from apps.knowledge_graph.retrieval.branch_contracts import (
    BranchStatusV1,
    DirectBranchFailureReason,
    SharedBranchFailureReason,
)
from apps.knowledge_graph.retrieval.production_extended import (
    prepare_extended_branch,
    run_extended_branch,
)
from apps.knowledge_graph.retrieval.production_runtime_support import (
    ProductionSharedScopeV1,
    graph_candidates,
    ppr_config,
    ppr_failure_envelope,
    success_envelope,
    topology_caps,
)
from apps.knowledge_graph.retrieval.projected_ppr import ppr_projected_v1
from apps.knowledge_graph.retrieval.ready_scope import (
    ReadyScopeError,
    ReadyScopeFailureReason,
    SelectedReadyScopeV1,
)
from apps.knowledge_graph.retrieval.scheduler_support import (
    SharedSchedulerFailure,
    failed_branch,
)
from lib.knowledge_graph.query_extractor.client import (
    QueryExtractorClient,
    QueryExtractorClientError,
    reconstruct_entity_texts,
)
from lib.knowledge_graph.query_extractor.config import QueryExtractorSettings

from .ready_materialization import materialize_selected_ready_chunks
from .ready_scope_repository import load_selected_ready_scope
from .topology.contracts import HybridBranchKind, ProjectedSeedV1


class ProductionHybridBranchRuntime:
    def __init__(
        self,
        *,
        authorization,
        settings,
        topology_loader,
        codec,
        scope_loader=load_selected_ready_scope,
        projection_repository_factory=None,
        extractor_factory=None,
        clock=monotonic,
    ) -> None:
        if type(authorization) is not RetrievalAuthorizationContext:
            raise TypeError("authorization must be exact")
        self.authorization, self.settings = authorization, settings
        self.topology_loader, self.codec = topology_loader, codec
        self.scope_loader = scope_loader
        self.projection_repository_factory = projection_repository_factory
        self.extractor_factory = extractor_factory
        self.clock = clock
        self._shared: ProductionSharedScopeV1 | None = None

    def _exact_request(self, authorization, settings) -> None:
        if authorization is not self.authorization or settings is not self.settings:
            raise ValueError("runtime request binding changed")

    def prepare_shared(self, *, authorization, settings, deadline):
        self._exact_request(authorization, settings)
        if self.clock() >= deadline:
            raise TimeoutError("overall graph deadline expired")
        try:
            scope = self.scope_loader(authorization=authorization, settings=settings)
        except ReadyScopeError as error:
            reason = (
                SharedBranchFailureReason.AUTHORIZATION_CONTEXT_INVALID
                if error.reason is ReadyScopeFailureReason.AUTHORIZATION_CONTEXT_INVALID
                else SharedBranchFailureReason.READINESS_MISMATCH
            )
            raise SharedSchedulerFailure(reason) from error
        if type(scope) is not SelectedReadyScopeV1:
            raise TypeError("ready scope loader returned an invalid scope")
        if self.clock() >= deadline:
            raise TimeoutError("overall graph deadline expired")
        self._shared = ProductionSharedScopeV1(scope)
        return self._shared

    def _shared_scope(self, shared) -> SelectedReadyScopeV1:
        if type(shared) is not ProductionSharedScopeV1 or shared is not self._shared:
            raise ValueError("shared scope is not request bound")
        return shared.scope

    def _extractor(self, *, scope, ontology):
        if self.extractor_factory is not None:
            return self.extractor_factory(
                scope=scope, ontology=ontology, settings=self.settings
            )
        settings = self.settings
        client_settings = QueryExtractorSettings(
            url=settings.query_extractor_url,
            bearer_token=settings.query_extractor_bearer_token,
            model_identifier=settings.query_extractor_model,
            model_revision=settings.query_extractor_model_revision,
            build_hash=settings.query_extractor_build_hash,
            schema_version=settings.query_extractor_expected_schema_version,
            schema_checksum=settings.query_extractor_expected_schema_checksum,
            ontology_path=Path("ontology.yaml"),
            ontology_checksum=ontology.checksum,
            timeout_ms=settings.query_extractor_timeout_ms,
            max_query_utf8_bytes=settings.query_max_bytes,
            max_query_code_points=settings.query_max_codepoints,
            max_spans=settings.query_max_spans,
        )
        return QueryExtractorClient(client_settings)

    def _direct_seeds(self, *, query, scope, deadline):
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
        from apps.knowledge_graph.retrieval.query_ontology import load_query_ontology

        if self.clock() >= deadline:
            return DirectBranchFailureReason.EXTRACTOR_TIMEOUT
        artifact_ids = tuple(sorted({row.artifact_id for row in scope.projections}))
        ontology_outcome = load_query_ontology(
            selected_artifact_ids=artifact_ids, using=self.authorization.database_alias
        )
        if self.clock() >= deadline:
            return DirectBranchFailureReason.EXTRACTOR_TIMEOUT
        ontology = ontology_outcome.ontology
        expected_ontologies = {
            (row.ontology_version, row.ontology_checksum) for row in scope.projections
        }
        if (
            ontology is None
            or len(expected_ontologies) != 1
            or ontology.checksum != scope.projections[0].ontology_checksum
        ):
            return DirectBranchFailureReason.MIXED_ONTOLOGY
        try:
            response = self._extractor(scope=scope, ontology=ontology).extract(
                query=query, ontology=ontology, deadline=deadline
            )
        except QueryExtractorClientError as error:
            return DirectBranchFailureReason(error.reason.value)
        surfaces = reconstruct_entity_texts(query=query, response=response)
        span_inputs = tuple(
            DirectResolutionSpanInputV1(span, text)
            for span, text in zip(response.spans, surfaces, strict=True)
        )
        generation_by_projection = dict(scope.generation_keys_by_projection)
        generation_by_artifact = tuple(
            sorted(
                (row.artifact_id, generation_by_projection[row.projection_id])
                for row in scope.projections
            )
        )
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
            codec=self.codec,
            span_inputs=span_inputs,
            using=self.authorization.database_alias,
        )
        outcome = resolve_direct_seed_components(
            spans=response.spans,
            repository=repository,
            ready=scope.ready,
            settings=self.settings,
            deadline=deadline,
        )
        if outcome.failure_reason is not None:
            return DirectBranchFailureReason(outcome.failure_reason.value)
        return tuple(
            sorted(
                (ProjectedSeedV1(row.component_key, row.mass) for row in outcome.seeds),
                key=lambda row: row.identity_key,
            )
        )

    def run_direct(self, *, query, shared, authorization, settings, deadline):
        self._exact_request(authorization, settings)
        scope, started = self._shared_scope(shared), self.clock()
        if settings.graph_direct_enabled is not True:
            return failed_branch(
                HybridBranchKind.DIRECT, DirectBranchFailureReason.DIRECT_NO_SEEDS
            )
        seeds = self._direct_seeds(query=query, scope=scope, deadline=deadline)
        if type(seeds) is DirectBranchFailureReason:
            return failed_branch(HybridBranchKind.DIRECT, seeds)
        caps = topology_caps(settings, HybridBranchKind.DIRECT)
        snapshot = self.topology_loader.load(
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
                HybridBranchKind.DIRECT,
                DirectBranchFailureReason.DIRECT_PPR_INVALID,
                seed_count=len(seeds),
                snapshot=snapshot,
                elapsed_ms=min(
                    settings.graph_direct_timeout_ms,
                    int((self.clock() - started) * 1000),
                ),
            )
        return success_envelope(
            HybridBranchKind.DIRECT,
            ready=scope.ready,
            seeds=seeds,
            snapshot=snapshot,
            candidates=candidates,
            settings=settings,
            elapsed_ms=max(0, int((self.clock() - started) * 1000)),
        )

    def prepare_extended(self, *, baseline, shared, authorization, settings, deadline):
        return prepare_extended_branch(
            self,
            baseline=baseline,
            shared=shared,
            authorization=authorization,
            settings=settings,
            deadline=deadline,
        )

    def run_extended(self, *, prepared, shared, authorization, settings, deadline):
        return run_extended_branch(
            self,
            prepared=prepared,
            shared=shared,
            authorization=authorization,
            settings=settings,
            deadline=deadline,
        )

    def materialize(self, *, chunk_keys, authorization, outcome):
        self._exact_request(authorization, self.settings)
        if self._shared is None:
            raise ValueError("ready scope has not been prepared")
        successful = tuple(
            row
            for row in (outcome.direct, outcome.extended)
            if row.status is BranchStatusV1.SUCCEEDED
        )
        if any(
            row.result.provenance.ready_bundle_checksum
            != self._shared.scope.ready.bundle_checksum
            for row in successful
        ):
            raise ValueError("materialization provenance differs from ready scope")
        return materialize_selected_ready_chunks(
            scope=self._shared.scope, chunk_keys=chunk_keys, authorization=authorization
        )


__all__ = ["ProductionHybridBranchRuntime", "graph_candidates"]
