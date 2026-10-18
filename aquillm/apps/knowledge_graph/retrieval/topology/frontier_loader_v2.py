"""One bounded breadth-first canonical frontier across captured generations."""

from apps.knowledge_graph.projection.frontier_reads_v2 import schema_error
from apps.knowledge_graph.projection.frontier_snapshot_v2 import FrontierAssemblyV2
from apps.knowledge_graph.retrieval.frontier_ranking_validation_v2 import (
    CONSUMED_POLICY, POLICY, FrontierRankingContextV2, frontier_tags,
)
from apps.knowledge_graph.retrieval.graph_handoff import GraphCoverageV2
from apps.knowledge_graph.retrieval.graph_query_contracts import (
    BoundedGraphTopologyV2,
    GraphRelationRequestV2,
)

from .contracts import (
    TopologyCapsV1,
    projected_seed_checksum,
    validate_projected_seed_sequence,
)
from .frontier_contracts_v2 import FrontierBudgetExceeded, FrontierBudgetV2
from .frontier_selection_v2 import FrontierSelectionV2


class FrontierUnavailableV2(Exception):
    """Known bounded unavailability, with no authority-looking open snapshot."""

    def __init__(self, coverage, budget):
        self.coverage = coverage
        self.accounting = tuple(
            getattr(budget, n)
            for n in ("identities", "edges", "source_rows", "queries", "wire_bytes")
        )
        self.reserved_source_rows = budget._reserved_rows
        self.work_uncertain = budget._uncertain
        super().__init__(",".join(coverage.reasons))


def canonical_frontier_loader_v2(*, selection, codec, adapter, acquisition_policy=POLICY):
    return CanonicalFrontierLoaderV2(
        adapter, FrontierSelectionV2(selection, codec), acquisition_policy
    )


class CanonicalFrontierLoaderV2:
    def __init__(self, adapter, binding, acquisition_policy=POLICY):
        if type(binding) is not FrontierSelectionV2:
            raise TypeError("use the captured-selection frontier factory")
        self.adapter, self.binding, self._contexts = adapter, binding, {}
        frontier_tags(acquisition_policy)
        self.acquisition_policy = acquisition_policy

    def ranking_context(self, result):
        if type(result) is not BoundedGraphTopologyV2:
            raise TypeError("frontier result must be exact")
        context = self._contexts.get(result.snapshot.algorithm.algorithm_signature)
        if context is None or context.group_checksum != result.authority_group_checksum:
            raise ValueError("result was not produced by this captured loader")
        return context

    def load(self, *, group, seeds, request, caps, budget, deadline):
        self.binding.require_group(group)
        if (
            type(request) is not GraphRelationRequestV2
            or request.ontology_checksum != group.ontology_checksum
        ):
            raise ValueError("frontier request ontology differs from captured group")
        if type(caps) is not TopologyCapsV1 or type(budget) is not FrontierBudgetV2:
            raise TypeError("frontier requires exact original caps and budget")
        validate_projected_seed_sequence(
            seeds,
            maximum=caps.max_seeds,
            expected_checksum=projected_seed_checksum(seeds),
        )
        context = FrontierRankingContextV2(
            group.group_checksum,
            budget.max_identities,
            budget.max_edges,
            budget.max_source_rows,
            budget.max_queries,
            budget.max_wire_bytes,
            self.acquisition_policy,
        )
        session = self.adapter.open(
            group=group, request=request, budget=budget, deadline=deadline
        )
        if session.acquisition_policy != context.acquisition_policy:
            raise ValueError("frontier session/context acquisition policy differs")
        assembly = FrontierAssemblyV2(session, self.binding, caps, context)
        reason = None
        count = len(group.scope.selected_document_ids)
        try:
            session.initialize()
            if not all(session.indexed.values()):
                raise FrontierUnavailableV2(
                    GraphCoverageV2(count, 0, "unavailable", ("not_ready",), False),
                    budget,
                )
            from .frontier_acquisition_v2 import acquire_frontier

            acquire_frontier(session, assembly, seeds, caps, budget)
        except FrontierBudgetExceeded:
            reason = "frontier_budget"
        except TimeoutError:
            reason = "deadline"
        except (KeyError, TypeError, ValueError) as exc:
            raise schema_error() from exc
        expected_scopes = len(session.ready.selected_generations) + len(
            session.ready.authorized_documents
        )
        actual_scopes = sum(
            sum(label == "ArtifactProvenance" for label, _ in rows)
            for rows in session.scope_rows.values()
        )
        closed = actual_scopes == expected_scopes
        if self.acquisition_policy == CONSUMED_POLICY:
            closed = set(session.scope_rows) == set(session.selected) and all(
                {(r.scope_type, r.scope_key, r.collection_key, r.artifact_key)
                 for (label, _), r in session.scope_rows[g].items()
                 if label == "ArtifactProvenance"} == {
                    ("collection", selected.collection_key, selected.collection_key,
                     selected.active_artifact_key)
                }
                for g, selected in session.selected.items()
            )
        if not closed or reason == "deadline":
            raise FrontierUnavailableV2(
                GraphCoverageV2(
                    count, 0, "unavailable", (reason or "not_ready",), False
                ),
                budget,
            )
        try:
            session.final_attestation()
        except (FrontierBudgetExceeded, TimeoutError) as exc:
            reason = (
                "frontier_budget"
                if isinstance(exc, FrontierBudgetExceeded)
                else "deadline"
            )
            raise FrontierUnavailableV2(
                GraphCoverageV2(count, 0, "unavailable", (reason,), False), budget
            ) from exc
        try:
            snapshot, candidates = assembly.build(request)
        except (KeyError, TypeError, ValueError) as exc:
            raise schema_error() from exc
        coverage = GraphCoverageV2(
            count,
            count,
            "partial" if reason else "complete",
            (reason,) if reason else (),
            False,
        )
        result = BoundedGraphTopologyV2(
            snapshot, candidates, coverage, session.request_digest, group.group_checksum
        )
        try:
            budget.check_active()
            if budget.clock() >= deadline:
                raise TimeoutError
        except TimeoutError as exc:
            reason = (
                "frontier_budget"
                if isinstance(exc, FrontierBudgetExceeded)
                else "deadline"
            )
            raise FrontierUnavailableV2(
                GraphCoverageV2(count, 0, "unavailable", (reason,), False), budget
            ) from exc
        self._contexts[snapshot.algorithm.algorithm_signature] = context
        return result
