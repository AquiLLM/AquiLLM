"""Native-only V2 pool: full fresh authority at both ends, no intermediate positives."""

from copy import copy
from dataclasses import fields, is_dataclass

from apps.documents.models import TextChunk as TextChunk
from apps.documents.services.chunk_search_scope import (
    current_scope as current_scope,
)
from apps.documents.services.chunk_search_scope import (
    validate_requested_authorization,
)
from apps.documents.services.chunk_search_source_operation import (
    _closed_header as _closed_header,
)
from apps.documents.services.graph_admission_adapter import (
    captured_admission as captured_admission,
)
from apps.documents.services.graph_admission_adapter import (
    deferred_baseline,
    order_graph_pool,
)
from apps.documents.services.graph_admission_adapter import (
    observe_published_graph as observe_published_graph,
)
from apps.documents.services.graph_admission_seeds import validate_final_native_seeds
from apps.documents.services.source_deadline import (
    source_preparation_scope as source_preparation_scope,
)
from apps.documents.services.source_deadline import (
    source_remaining_ms as source_remaining_ms,
)
from apps.documents.services.source_loading import (
    bounded_source_database as bounded_source_database,
)

from .closed_graph_attempt_v2 import (
    _capture_closed_graph_attempt,
    _ClosedGraphLimited,
    _graph_observations,
)
from .graph_candidate_stage_v2 import _GraphSourceStage as _GraphSourceStage
from .production_runtime import (
    _NATIVE_GRAPH_RUN_V2 as _NATIVE_GRAPH_RUN_V2,
)
from .production_runtime import (
    ProductionHybridBranchRuntime as ProductionHybridBranchRuntime,
)


def _settings_values(settings):
    return (
        {f.name: getattr(settings, f.name) for f in fields(settings)}
        if is_dataclass(settings)
        else vars(settings)
    )


def _same_references(captured, current):
    return all(a is b for a, b in zip(captured, current, strict=True))


def _limit_observations(exc):
    from .graph_frontier_observation_v2 import (
        FrontierBudgetObservationV2,
        GraphFrontierInvocationV2,
    )
    from .graph_handoff import GraphCoverageV2

    try:
        if type(exc) is not _ClosedGraphLimited:
            return {}
        reason, raw = exc.reason, exc.observations
        if (type(reason) is not str or reason not in {"deadline", "frontier_budget"}
                or type(raw) is not dict):
            return {}
        snapshot = {}
        if raw:
            if (any(type(key) is not str for key in raw) or set(raw) != {
                    "graph_coverage_v2", "frontier_budget", "frontier_invocations"}):
                return {}
            invocations = raw["frontier_invocations"]
            if type(invocations) is not list or len(invocations) != 1:
                return {}
            invocation = invocations[0]
            if (type(invocation) is not dict
                    or any(type(key) is not str for key in invocation)):
                return {}
            coverage = raw["graph_coverage_v2"]
            budgets = (raw["frontier_budget"], invocation["frontier_budget"])
            for value in (coverage, invocation, *budgets):
                if type(value) is not dict or any(type(k) is not str for k in value):
                    return {}
            coverage = GraphCoverageV2(**coverage)
            copies = []
            for budget in budgets:
                copied_invocation = GraphFrontierInvocationV2(**{
                    **invocation,
                    "frontier_budget": FrontierBudgetObservationV2(**budget),
                })
                copies.append(_graph_observations(coverage, copied_invocation))
            if copies[0] != copies[1]:
                return {}
            snapshot = copies[0]
        return {**snapshot, "graph_failure_reason": reason}
    except Exception:
        return {}  # Malformed/forged exceptions remain generic graph errors.


def _try_closed_graph_pool(
    snapshot, query, authorization, dependencies, *, diagnostics, fallback
):
    from apps.documents.services.hybrid_graph_deadline import graph_rerank_reserve_ms

    attempt = _capture_closed_graph_attempt(
        snapshot, query, authorization, dependencies, diagnostics=diagnostics
    )
    if attempt is None:
        return None
    outcome = attempt.run(downstream_reserve_ms=graph_rerank_reserve_ms())
    if outcome.kind == "success":
        return outcome.pool, outcome.diagnostics
    if outcome.kind == "invalid":
        return (), outcome.diagnostics
    pool, observation = fallback(admit_deferred=outcome.admit_deferred)
    reason = outcome.diagnostics.get("graph_failure_reason")
    if type(reason) is str and reason in {"deadline", "frontier_budget"}:
        keys = ("graph_failure_reason", "graph_coverage_v2", "frontier_budget",
                "frontier_invocations")
        observation = {**observation, **{
            key: outcome.diagnostics[key] for key in keys if key in outcome.diagnostics
        }}
    return pool, observation


def _run_closed_graph(
    stage, snapshot, query, authorization, graph, settings, deadline, check,
    diagnostics, admission, *, native_closure=False,
):
    from apps.documents.services.chunk_search_scoped_rows import _retain_dense
    from apps.documents.services.hybrid_graph_v2 import bounded_pool_v2
    from apps.knowledge_graph.resolution.vocabulary_authority import validate_ready

    from .graph_evidence_repository_v2 import _load_selection_inputs_v2
    from .graph_hydration_operation_v2 import _hydrate_closed_selection_rows
    from .graph_runtime_contracts_v2 import GraphRuntimeEnvelopeV2
    from .graph_runtime_execution_v2 import _execute_captured_graph_v2
    from .graph_runtime_v2 import _run_graph_v2
    from .frontier_ranking_validation_v2 import CONSUMED_POLICY, POLICY
    from .ready_materialization_v2 import _materialize_closed_rows, _validate_closed_captured_selection
    from .ready_scope_repository_v2 import _select_closed_ready_scope_v2
    from .ready_scope_v2 import graph_selection_digest_v2

    scope, budget = stage.scope, stage.budget
    validate_requested_authorization(stage.runtime, authorization, snapshot.documents)
    stage.initial_keys = frozenset(scope.validate())
    baseline, _ = stage.current(tuple(snapshot.baseline_candidates), stage.initial_keys)
    baseline = tuple(
        row for row in baseline if row.doc_id in scope.requested_document_ids
    )
    selected = _select_closed_ready_scope_v2(
        authorization=scope.authorization,
        settings=settings,
        representation_snapshot=scope.representation,
        requested_document_ids=scope.requested_document_ids,
    )
    check()
    runtime = copy(graph)
    runtime.authorization = scope.authorization
    seed_validation = (
        stage.seed_validation(snapshot, check, runtime=runtime, selected=selected)
        if admission is not None
        else None
    )
    native = seed_validation is not None and seed_validation._native_metadata is not None
    acquisition_policy = CONSUMED_POLICY if native else POLICY

    def hydrate(*, selected, pairs, runtime, document_codec, deadline):
        check()
        for group in selected.groups:
            validate_ready(group.scope, scope.authorization)
        rows, fingerprints = _hydrate_closed_selection_rows(
            selected=selected,
            pairs=pairs,
            runtime=runtime,
            scope=scope,
            codec=graph.codec,
            document_codec=document_codec,
            deadline=deadline,
            stage=stage,
            admission=admission,
        )
        stage.fingerprints(fingerprints)
        check()
        return rows

    def load(**kwargs):
        return _load_selection_inputs_v2(**kwargs, hydrate=hydrate)

    def execute(**kwargs):
        return _execute_captured_graph_v2(**kwargs, selected=selected, hydrate=load,
            seed_validation=seed_validation, acquisition_policy=acquisition_policy,
        )
    envelope = _run_graph_v2(
        query=query,
        runtime=runtime,
        authorization=scope.authorization,
        representation_snapshot=scope.representation,
        budget=budget,
        settings=settings,
        deadline=deadline,
        baseline=snapshot,
        requested_document_ids=scope.requested_document_ids,
        execute=execute,
        acquisition_policy=acquisition_policy,
    )
    if (
        type(envelope) is not GraphRuntimeEnvelopeV2
        or envelope.selection is not selected
    ):
        raise ValueError("V2 replaced captured selection")
    envelope.__post_init__()
    observations = _graph_observations(
        envelope.handoff.coverage, envelope.frontier_invocation
    )
    try:
        check()
    except _ClosedGraphLimited as exc:
        if type(exc) is not _ClosedGraphLimited:
            raise
        raise _ClosedGraphLimited(
            exc.reason,
            coverage=envelope.handoff.coverage,
            invocation=envelope.frontier_invocation,
        ) from None
    if native_closure and not envelope.handoff.candidates:
        for reason in ("deadline", "frontier_budget"):
            if reason in envelope.handoff.coverage.reasons:
                raise _ClosedGraphLimited(
                    reason,
                    coverage=envelope.handoff.coverage,
                    invocation=envelope.frontier_invocation,
                )
    rows = _materialize_closed_rows(
        selected=selected,
        refs=envelope.handoff.candidates,
        authorization=scope.authorization,
        codec=graph.codec,
        stage=stage,
    )
    if len(rows) != len(envelope.handoff.candidates) or any(
        row.doc_id not in scope.requested_document_ids for row in rows
    ):
        raise ValueError("V2 final materialization lost exact authority")
    current, _ = stage.current(rows, stage.initial_keys)
    if current != rows:
        raise ValueError("V2 evidence changed after selection hydration")
    baseline_ids = {row.pk for row in baseline}
    baseline = deferred_baseline(admission, stage, baseline)
    combined = baseline + tuple(
        row for row in rows if row.pk not in {b.pk for b in baseline}
    )
    ordered = order_graph_pool(
        snapshot,
        combined,
        query=query,
        scope=stage.staged,
        graph_authority=graph_selection_digest_v2(selected),
        graph_policy=envelope.handoff.algorithm_signature,
        preserved_ranks={
            "graph_v2": tuple((row.pk, rank) for rank, row in enumerate(rows, 1))
        },
    )
    ordered = bounded_pool_v2(ordered, baseline)
    _validate_closed_captured_selection(
        selected=selected,
        authorization=scope.authorization,
        settings=settings,
        codec=graph.codec,
    )
    keys = stage.initial_keys & set(scope.validate())
    stage.final_keys = frozenset(keys)
    validate_final_native_seeds(snapshot, keys)
    final, valid_fts = stage.current(ordered, keys)
    final = tuple(_retain_dense(final, stage.staged, keys, valid_fts))
    if final != ordered:
        raise ValueError("V2 final source fence changed")
    check()
    added = len(({row.pk for row in final} & {row.pk for row in rows}) - baseline_ids)
    observations = {**diagnostics("hit" if added else "miss", added), **observations}
    if admission is not None:
        observations["source_admission"] = {
            **admission.diagnostics(),
            "graph_selected": len(rows),
            "graph_novel_selected": added,
        }
    selected_pks = tuple(
        dict.fromkeys(row.pk for row in rows if row.pk in {r.pk for r in final})
    )
    return (final, observations), final, selected_pks
