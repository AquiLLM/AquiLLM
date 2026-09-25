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
    ExtendedBranchFailureReason,
    SharedBranchFailureReason,
)
from apps.knowledge_graph.retrieval.ppr_seed_support import PreparedPPRSeedsV1
from apps.knowledge_graph.retrieval.production_direct import (
    prepare_direct_seeds,
    prepare_direct_with_policy,
)
from apps.knowledge_graph.retrieval.production_extended import (
    prepare_extended_branch,
    prepare_extended_with_policy,
    run_extended_branch,
)
from apps.knowledge_graph.retrieval.production_ppr_policy import rank_projected_for_mode
from apps.knowledge_graph.retrieval.production_runtime_support import (
    ProductionSharedScopeV1,
    graph_candidates,
    ppr_config,
    ppr_failure_envelope,
    success_envelope,
    topology_caps,
)
from apps.knowledge_graph.retrieval.ready_scope import (
    ReadyScopeError,
    ReadyScopeFailureReason,
    SelectedReadyScopeV1,
)
from apps.knowledge_graph.retrieval.scheduler_support import (
    SharedSchedulerFailure,
    failed_branch,
)
from lib.knowledge_graph.query_extractor.client import QueryExtractorClient
from lib.knowledge_graph.query_extractor.config import QueryExtractorSettings

from .ready_materialization import materialize_selected_ready_chunks
from .ready_scope_repository import load_selected_ready_scope
from .topology.contracts import HybridBranchKind


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
        return prepare_direct_seeds(self, query=query, scope=scope, deadline=deadline)

    def run_direct(self, *, query, shared, authorization, settings, deadline):
        self._exact_request(authorization, settings)
        scope, started = self._shared_scope(shared), self.clock()
        if settings.graph_direct_enabled is not True:
            return failed_branch(
                HybridBranchKind.DIRECT, DirectBranchFailureReason.DIRECT_NO_SEEDS
            )
        mode = getattr(settings, "ppr_restart_mode", "fixed")
        if mode == "fixed":
            prepared = self._direct_seeds(query=query, scope=scope, deadline=deadline)
        else:
            try:
                prepared = prepare_direct_with_policy(
                    self, query=query, scope=scope, deadline=deadline
                )
            except (TypeError, ValueError):
                return failed_branch(
                    HybridBranchKind.DIRECT,
                    DirectBranchFailureReason.DIRECT_SEED_INVALID,
                )
        if type(prepared) is DirectBranchFailureReason:
            return failed_branch(HybridBranchKind.DIRECT, prepared)
        if mode != "fixed" and type(prepared) is not PreparedPPRSeedsV1:
            return failed_branch(
                HybridBranchKind.DIRECT, DirectBranchFailureReason.DIRECT_SEED_INVALID
            )
        seeds = prepared if mode == "fixed" else prepared.seeds
        caps = topology_caps(settings, HybridBranchKind.DIRECT)
        snapshot = self.topology_loader.load(
            ready=scope.ready, seeds=seeds, caps=caps, deadline=deadline
        )
        try:
            result, execution_signature = rank_projected_for_mode(
                snapshot=snapshot,
                seeds=seeds,
                config=ppr_config(snapshot, caps.max_results),
                prepared=None if mode == "fixed" else prepared,
                mode=mode,
                branch=HybridBranchKind.DIRECT,
                deadline_check=lambda: self._check_branch_deadline(deadline),
            )
            candidates = graph_candidates(
                snapshot=snapshot,
                identity_scores=result.scores,
                maximum=caps.max_results,
            )
        except TimeoutError:
            return failed_branch(
                HybridBranchKind.DIRECT,
                DirectBranchFailureReason.EXTRACTOR_TIMEOUT,
                elapsed_ms=settings.graph_direct_timeout_ms,
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
            execution_algorithm_signature=execution_signature,
        )

    def _check_branch_deadline(self, deadline):
        if self.clock() >= deadline:
            raise TimeoutError("graph branch deadline expired")

    def prepare_extended(
        self, *, query, baseline, shared, authorization, settings, deadline
    ):
        if getattr(settings, "ppr_restart_mode", "fixed") == "fixed":
            return prepare_extended_branch(
                self,
                baseline=baseline,
                shared=shared,
                authorization=authorization,
                settings=settings,
                deadline=deadline,
            )
        self._exact_request(authorization, settings)
        self._shared_scope(shared)
        try:
            return prepare_extended_with_policy(
                self,
                query=query,
                baseline=baseline,
                shared=shared,
                authorization=authorization,
                settings=settings,
                deadline=deadline,
            )
        except (TypeError, ValueError):
            return ExtendedBranchFailureReason.EXTENDED_SEED_INVALID

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
