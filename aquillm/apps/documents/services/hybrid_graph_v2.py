"""Private V2 graph handoff into the original frozen retrieval scope."""

from copy import copy
from dataclasses import asdict
from time import perf_counter

from apps.knowledge_graph.retrieval.graph_runtime_contracts_v2 import (
    GraphRuntimeEnvelopeV2,
)
from apps.knowledge_graph.retrieval.ready_materialization_v2 import (
    materialize_graph_candidates_v2,
    validate_captured_selection,
)
from apps.knowledge_graph.retrieval.ready_scope_v2 import graph_selection_digest_v2

from .chunk_search_legacy_graph import graph_diagnostics
from .chunk_search_scope import current_scope, validate_requested_authorization
from .chunk_search_scoped_rows import validate_rows
from .hybrid_graph_authorization import reauthorized_baseline
from .source_deadline import check_source_deadline, source_preparation_scope


def bounded_pool_v2(ordered, baseline):
    from .candidate_rank_fusion import MAX_CANDIDATES

    if len(baseline) > MAX_CANDIDATES or len({r.pk for r in baseline}) != len(baseline):
        raise ValueError("captured baseline exceeds original pool contract")
    return tuple(ordered[:MAX_CANDIDATES])


def hybrid_graph_candidate_pool_v2(snapshot, query, authorization, dependencies):
    started = perf_counter()
    scope, settings = current_scope(), dependencies.settings
    coverage = None
    frontier_observation = None
    frontier_invocation = None

    def diagnostics(status, count=0):
        return {
            **graph_diagnostics(
                started_at=started,
                seed_count=len(snapshot.graph_seeds),
                candidate_count=count,
                status=status,
                algorithm_signature=None,
                version_signature=None,
            ),
            **({"graph_coverage_v2": asdict(coverage)} if coverage else {}),
            **(
                {"frontier_budget": asdict(frontier_observation)}
                if frontier_observation
                else {}
            ),
            **(
                {"frontier_invocations": [asdict(frontier_invocation)]}
                if frontier_invocation
                else {}
            ),
        }

    def fallback():
        try:
            rows, _ = reauthorized_baseline(snapshot.baseline_candidates, authorization)
            if scope is not None:
                rows = tuple(
                    r
                    for r in validate_rows(rows, scope=scope)
                    if r.doc_id in scope.requested_document_ids
                )
            return bounded_pool_v2(rows, rows), diagnostics("error")
        except Exception:
            return (), diagnostics("error")

    from apps.knowledge_graph.retrieval.graph_candidate_operation_v2 import (
        _try_closed_graph_pool,
    )

    closed = _try_closed_graph_pool(
        snapshot,
        query,
        authorization,
        dependencies,
        diagnostics=diagnostics,
        fallback=fallback,
    )
    if closed is not None:
        return closed

    try:
        if getattr(settings, "graph_remediation_mode", None) != "v2":
            raise ValueError("production graph mode is not v2")
        if scope is None or scope.runtime is None:
            raise ValueError("V2 requires the original frozen retrieval scope")
        budget = scope.runtime.budget
        timeout = settings.graph_overall_timeout_ms
        if type(timeout) is not int or not 1 <= timeout <= 5000:
            raise ValueError("graph timeout is invalid")
        deadline = min(started + timeout / 1000, started + budget.remaining_ms() / 1000)
        with source_preparation_scope(deadline, perf_counter, budget), scope.database():
            validate_requested_authorization(
                scope.runtime, authorization, snapshot.documents
            )
            scope.validate()
            baseline = tuple(
                r
                for r in validate_rows(snapshot.baseline_candidates, scope=scope)
                if r.doc_id in scope.requested_document_ids
            )
            # The capability was built for the incoming query subset. Its source
            # operations must retain the original turn authorization and lifetime.
            runtime = copy(dependencies.runtime)
            runtime.authorization = scope.authorization
            envelope = runtime.run_v2(
                query=query,
                authorization=scope.authorization,
                representation_snapshot=scope.representation,
                budget=budget,
                settings=settings,
                deadline=deadline,
                baseline=snapshot,
                requested_document_ids=scope.requested_document_ids,
            )
            if type(envelope) is not GraphRuntimeEnvelopeV2:
                raise TypeError("V2 runtime returned a noncaptured envelope")
            envelope.__post_init__()
            selected, handoff = envelope.selection, envelope.handoff
            frontier_observation = envelope.frontier_budget
            frontier_invocation = envelope.frontier_invocation
            if (
                selected.representation_snapshot is not scope.representation
                or selected.requested_document_ids != scope.requested_document_ids
                or selected.authorization_context_signature
                != scope.authorization.authorization_context_signature
            ):
                raise ValueError("V2 replaced original retrieval scope")
            coverage = handoff.coverage
            rows = materialize_graph_candidates_v2(
                selected=selected,
                refs=handoff.candidates,
                authorization=scope.authorization,
            )
            if len(rows) != len(handoff.candidates) or any(
                r.doc_id not in scope.requested_document_ids for r in rows
            ):
                raise ValueError("V2 final materialization lost exact authority")
            current = validate_rows(rows, scope=scope)
            if current != rows:
                raise ValueError("V2 evidence changed after selection hydration")
            baseline_ids = {r.pk for r in baseline}
            combined = baseline + tuple(r for r in rows if r.pk not in baseline_ids)
            from .chunk_search_rank_policy import ordered_candidates, snapshot_channels

            ordered, _ = ordered_candidates(
                combined,
                snapshot_channels(snapshot),
                query=query,
                exact_rows=snapshot.exact_results,
                scope=scope,
                graph_authority=graph_selection_digest_v2(selected),
                graph_policy=handoff.algorithm_signature,
                preserved_ranks={
                    "graph_v2": tuple((r.pk, rank) for rank, r in enumerate(rows, 1))
                },
            )
            ordered = bounded_pool_v2(ordered, baseline)
            validate_captured_selection(
                selected=selected,
                authorization=scope.authorization,
                settings=settings,
                codec=runtime.codec,
            )
            final = validate_rows(ordered, scope=scope)
            if final != ordered:
                raise ValueError("V2 final source fence changed")
            check_source_deadline()
            if not budget.can_publish():
                raise TimeoutError("V2 handoff deadline")
            added = len({r.pk for r in final} - baseline_ids)
            return final, diagnostics("hit" if added else "miss", added)
    except Exception:
        return fallback()
