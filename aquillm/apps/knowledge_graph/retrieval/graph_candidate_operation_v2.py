"""Native-only V2 pool: full fresh authority at both ends, no intermediate positives."""

from copy import copy, deepcopy
from dataclasses import fields, is_dataclass

from apps.documents.models import TextChunk
from apps.documents.services.chunk_search_scope import (
    current_scope,
    validate_requested_authorization,
)
from apps.documents.services.chunk_search_source_operation import (
    _closed_header,
)
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.source_deadline import (
    _DEADLINE,
    source_preparation_scope,
    source_remaining_ms,
)
from apps.documents.services.source_loading import bounded_source_database
from apps.documents.services.graph_admission_adapter import (
    captured_admission, deferred_baseline, order_graph_pool, selector_body_reader,
    observe_published_graph,
)
from apps.documents.services.graph_admission_seeds import validate_final_native_seeds

from .graph_candidate_stage_v2 import _GraphSourceStage
from .production_runtime import _NATIVE_GRAPH_RUN_V2, ProductionHybridBranchRuntime

def _settings_values(settings):
    return {f.name: getattr(settings, f.name) for f in fields(settings)} if is_dataclass(settings) else vars(settings)


def _same_references(captured, current):
    return all(a is b for a, b in zip(captured, current, strict=True))

def _try_closed_graph_pool(
    snapshot, query, authorization, dependencies, *, diagnostics, fallback
):
    scope, graph, settings = (
        current_scope(),
        dependencies.runtime,
        dependencies.settings,
    )
    if (
        scope is None
        or type(graph) is not ProductionHybridBranchRuntime
        or graph.settings is not settings
        or getattr(graph.run_v2, "__func__", None) is not _NATIVE_GRAPH_RUN_V2
        or getattr(settings, "graph_remediation_mode", None) != "v2"
    ):
        return None
    base = TextChunk.objects.using(scope.using).all()
    header = _closed_header(base, scope)
    if header is None:
        return None
    stage = _GraphSourceStage(scope, base, header)
    admission = getattr(snapshot, "source_admission", None)
    dependencies_header = (
        dependencies.runtime,
        dependencies.settings,
        dependencies.materialize,
    )
    graph_header = (graph.authorization, graph.settings, graph.codec, graph.clock)
    settings_before = deepcopy(_settings_values(settings))
    from apps.documents.services.chunk_search_rank_policy import controls

    ranking_before = controls()
    from lib.retrieval.operation import _OPERATION, capture_operation_bound

    operation = _OPERATION.get()
    parent_deadline = _DEADLINE.get()

    def check():
        if parent_deadline is not None and (
            parent_deadline[1]() >= parent_deadline[0]
            or (parent_deadline[2] is not None and not parent_deadline[2].can_publish())
        ):
            raise SourceScopeUnavailable("graph_original_scope_unavailable")
        if operation is not None and not operation.active():
            raise SourceScopeUnavailable("graph_original_scope_unavailable")
        stage.check()
        if getattr(snapshot, "source_admission", None) is not admission:
            raise SourceScopeUnavailable("graph_original_scope_unavailable")
        captured_admission(snapshot, scope, query=query)
        if (
            not _same_references(
                dependencies_header,
                (dependencies.runtime, dependencies.settings, dependencies.materialize),
            )
            or getattr(graph.run_v2, "__func__", None) is not _NATIVE_GRAPH_RUN_V2
            or not _same_references(
                graph_header,
                (graph.authorization, graph.settings, graph.codec, graph.clock),
            )
            or _settings_values(settings) != settings_before
            or controls() != ranking_before
            or _OPERATION.get() is not operation
        ):
            raise SourceScopeUnavailable("graph_original_scope_unavailable")

    deadline = None
    try:
        bound = capture_operation_bound(stage.budget)
        timeout = settings.graph_overall_timeout_ms
        if type(timeout) is not int or not 1 <= timeout <= 5000:
            raise ValueError("graph timeout is invalid")
        clock = graph.clock
        now = (
            clock()
        )  # Translate the original source allowance before installing a child.
        deadline = min(
            now + timeout / 1000,
            now + source_remaining_ms(stage.budget.remaining_ms()) / 1000,
        )
        if bound is not None:
            from apps.documents.services.hybrid_graph_deadline import (
                graph_rerank_reserve_ms,
            )

            deadline = min(
                deadline,
                bound.deadline_for(clock, reserve_ms=graph_rerank_reserve_ms()),
            )

        def phase_check():
            if clock() >= deadline:
                raise TimeoutError("graph phase deadline")
            check()

        preparation_clock = parent_deadline[1] if parent_deadline is not None else clock
        preparation_now = preparation_clock()
        preparation_deadline = preparation_now + max(0, deadline - clock())
        phase_check()
        with header[0][5]:
            phase_check()
            with source_preparation_scope(
                preparation_deadline, preparation_clock, stage.budget
            ):
                phase_check()
                with bounded_source_database(stage.runtime, scope.using):
                    phase_check()
                    result, final_rows, selected_pks = _run_closed_graph(
                        stage,
                        snapshot,
                        query,
                        authorization,
                        graph,
                        settings,
                        deadline,
                        phase_check,
                        diagnostics,
                        admission,
                    )
            # Both preparation and DB cleanup precede publication under the same lock.
            phase_check()
            stage.publish(final_rows, phase_check)
            observe_published_graph(selected_pks)
            return result
    except Exception:
        # Mid-operation identity/lifetime loss cannot switch to a fresh authority.
        try:
            check()
        except Exception:
            return (), diagnostics("error")
        stage.withdraw()
        return fallback(admit_deferred=deadline is None or clock() < deadline)
    finally:
        stage.withdraw()


def _run_closed_graph(
    stage, snapshot, query, authorization, graph, settings, deadline, check, diagnostics, admission
):
    from dataclasses import asdict

    from apps.documents.services.chunk_search_scoped_rows import _retain_dense
    from apps.documents.services.hybrid_graph_v2 import bounded_pool_v2
    from apps.knowledge_graph.resolution.vocabulary_authority import validate_ready

    from .graph_evidence_repository_v2 import _load_selection_inputs_v2
    from .graph_hydration_operation_v2 import _hydrate_selection_rows
    from .graph_runtime_contracts_v2 import GraphRuntimeEnvelopeV2
    from .graph_runtime_execution_v2 import _execute_captured_graph_v2
    from .graph_runtime_v2 import _run_graph_v2
    from .ready_materialization_v2 import _materialize_rows, validate_captured_selection
    from .ready_scope_repository_v2 import select_ready_scope_v2
    from .ready_scope_v2 import graph_selection_digest_v2

    scope, budget = stage.scope, stage.budget
    validate_requested_authorization(stage.runtime, authorization, snapshot.documents)
    stage.initial_keys = frozenset(scope.validate())
    baseline, _ = stage.current(tuple(snapshot.baseline_candidates), stage.initial_keys)
    baseline = tuple(
        row for row in baseline if row.doc_id in scope.requested_document_ids
    )
    selected = select_ready_scope_v2(
        authorization=scope.authorization,
        settings=settings,
        representation_snapshot=scope.representation,
        requested_document_ids=scope.requested_document_ids,
    )
    check()
    runtime = copy(graph)
    runtime.authorization = scope.authorization
    seed_validation = stage.seed_validation(snapshot, check, runtime=runtime) if admission is not None else None

    def hydrate(*, selected, pairs, runtime, document_codec, deadline):
        check()
        for group in selected.groups:
            validate_ready(group.scope, scope.authorization)
        rows, fingerprints = _hydrate_selection_rows(
            selected=selected,
            pairs=pairs,
            runtime=runtime,
            scope=scope,
            codec=graph.codec,
            document_codec=document_codec,
            deadline=deadline,
            body_reader=selector_body_reader(admission, stage),
        )
        stage.fingerprints(fingerprints)
        check()
        return rows

    def load(**kwargs):
        return _load_selection_inputs_v2(**kwargs, hydrate=hydrate)

    def execute(**kwargs):
        return _execute_captured_graph_v2(**kwargs, selected=selected, hydrate=load,
                                        seed_validation=seed_validation)

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
    )
    if type(envelope) is not GraphRuntimeEnvelopeV2 or envelope.selection is not selected:
        raise ValueError("V2 replaced captured selection")
    envelope.__post_init__()
    rows = _materialize_rows(
        selected=selected,
        refs=envelope.handoff.candidates,
        authorization=scope.authorization,
        codec=graph.codec,
        body_reader=stage.read,
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
    combined = baseline + tuple(row for row in rows if row.pk not in {b.pk for b in baseline})
    ordered = order_graph_pool(
        snapshot, combined,
        query=query,
        scope=stage.staged,
        graph_authority=graph_selection_digest_v2(selected),
        graph_policy=envelope.handoff.algorithm_signature,
        preserved_ranks={
            "graph_v2": tuple((row.pk, rank) for rank, row in enumerate(rows, 1))
        },
    )
    ordered = bounded_pool_v2(ordered, baseline)
    validate_captured_selection(
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
    observations = {**diagnostics("hit" if added else "miss", added),
                    "graph_coverage_v2": asdict(envelope.handoff.coverage)}
    observations.update(frontier_budget=asdict(envelope.frontier_budget),
                        frontier_invocations=[asdict(envelope.frontier_invocation)])
    if admission is not None:
        observations["source_admission"] = {
            **admission.diagnostics(), "graph_selected": len(rows), "graph_novel_selected": added,
        }
    selected_pks = tuple(dict.fromkeys(row.pk for row in rows if row.pk in {r.pk for r in final}))
    return (final, observations), final, selected_pks
