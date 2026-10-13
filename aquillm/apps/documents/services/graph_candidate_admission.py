"""Production-only native V2 metadata collector; all other collectors stay eager."""

from copy import deepcopy
from contextlib import contextmanager
from dataclasses import replace
from threading import get_ident

from .chunk_search_graph_seeds import build_graph_seeds
from .chunk_search_rank_policy import controls
from .chunk_search_scope import FrozenRetrievalScope, validate_requested_authorization
from .chunk_search_source_operation import _closed_header, _same_header
from .graph_candidate_metadata import capture_metadata
from .graph_source_admission import NativeSourceAdmission, dependency_header, native_dependencies
from .graph_source_admission_policy import CHANNELS, allocation, charged_sources, fair_prefix
from .representation_contracts import SourceScopeUnavailable


@contextmanager
def _metadata_database(scope, header, check):
    check()
    with header[0][5]:
        check()
        with scope.database():
            check()
            yield
        check()


def _query_bindings(scope, query):
    from lib.embeddings.document_contracts import digest
    from lib.embeddings.document_provider import prepare_query
    from lib.embeddings.utils import EmbeddingContractError

    bindings = set()
    if scope.query_compatibility is None:
        bindings.add(digest({"version": "unprepared-query-v1", "query": query}))
    if scope.space is not None and scope.query_compatibility in (None, scope.space.space_key):
        try:
            bindings.add(prepare_query(query, space=scope.space)[1])
        except EmbeddingContractError:
            pass
    return frozenset(bindings)


def try_native_snapshot(model, query, documents, scope, observation, config, limits,
                        initial_error, authorization, dependencies):
    from apps.documents.models import TextChunk
    from apps.knowledge_graph.retrieval import GraphExpansionConfig

    if (model is not TextChunk or type(scope) is not FrozenRetrievalScope
            or type(config) is not GraphExpansionConfig
            or dependencies is None or not native_dependencies(dependencies, scope)):
        return None
    base = model.objects.using(scope.using).all()
    header = _closed_header(base, scope)
    if header is None:
        return None
    from apps.knowledge_graph.retrieval.graph_candidate_operation_v2 import _settings_values
    from lib.retrieval.operation import _OPERATION
    from .source_deadline import _DEADLINE, check_source_deadline
    from .graph_source_runtime import graph_source_runtime
    from .chunk_search_scope import current_scope

    operation, ranking = _OPERATION.get(), tuple(controls().items())
    parent_deadline = _DEADLINE.get()
    settings_before = deepcopy(_settings_values(dependencies.settings))
    dependencies_before = dependency_header(dependencies)
    dense_container = scope.dense_rows
    dense_before = dense_container.copy()
    query_bindings = _query_bindings(scope, query)

    def check():
        check_source_deadline()
        if (current_scope() is not scope or graph_source_runtime() is not scope.runtime
                or (get_ident() == parent_thread
                    and connections[scope.using] is not parent_connection)
                or scope.query_digest not in query_bindings
                or not _same_header(header, _closed_header(base, scope))
                or not scope.runtime.budget.can_publish()
                or _OPERATION.get() is not operation
                or _DEADLINE.get() is not parent_deadline
                or (operation is not None and not operation.active())
                or not native_dependencies(dependencies, scope)
                or any(a is not b for a, b in zip(dependencies_before,
                       dependency_header(dependencies), strict=True))
                or tuple(controls().items()) != ranking
                or _settings_values(dependencies.settings) != settings_before
                or scope.dense_rows is not dense_container or dense_container != dense_before):
            raise SourceScopeUnavailable("source_admission_original_scope_unavailable")

    from django.db import connections
    parent_connection = connections[scope.using]
    parent_thread = get_ident()
    allow_lexical_overlap = (
        parent_connection.vendor == "postgresql"
        and not parent_connection.in_atomic_block
        and parent_connection.get_autocommit()
        and not parent_connection.execute_wrappers
    )
    check()
    with _metadata_database(scope, header, check):
        check()
        validate_requested_authorization(scope.runtime, authorization, documents)
        channels, dense, terms, timings, coverage, error = capture_metadata(
            model, query, scope, limits, observation, check,
            allow_lexical_overlap=allow_lexical_overlap,
        )
        from apps.knowledge_graph.retrieval.graph_runtime_v2 import _caps

        cap = min(20, config.max_candidates, _caps(dependencies.settings).max_results)
        budget = scope.runtime.budget
        charged = charged_sources(budget)
        active = sum(bool(channels[name]) for name in CHANNELS)
        baseline_allowance, graph_ceiling = allocation(
            max(0, budget.limits.unique_sources - len(charged)), cap, active
        )
        ids = {name: tuple(r.pk for r in refs) for name, refs in channels.items()}
        all_refs = {r.pk: r for refs in channels.values() for r in refs}
        reused = tuple(pk for pk in fair_prefix(ids, 45) if all_refs[pk].identity in charged)
        novel = {name: tuple(pk for pk in lane if all_refs[pk].identity not in charged)
                 for name, lane in ids.items()}
        protected = ()
        if dict(ranking)["protected_exact"] and terms and any(
            term.case_sensitive or term.kind == "phrase" for term in terms
        ):
            protected = tuple(r.pk for r in channels["exact"]
                              if r.exact_coverage == len(terms))[:1]
        selected = set(reused + fair_prefix(novel, baseline_allowance, protected=protected))
        baseline_order = tuple(dict.fromkeys(r.pk for name in
                              ("dense", "trigram", "exact", "full_text") for r in channels[name]))
        initial_ids = tuple(pk for pk in baseline_order if pk in selected)[:45]
        check()
    plan = NativeSourceAdmission(
        scope, header, operation, parent_deadline, dependencies, dependencies_before, settings_before, ranking,
        tuple((name, channels[name]) for name in CHANNELS), tuple(dense.items()),
        query, graph_ceiling, initial_ids, frozenset(),
    )
    initial = plan.initial()
    plan = replace(plan, graph_start=charged_sources(budget))
    plan.counts["baseline_initial_hydrated"] = len(initial)
    by_id = {r.pk: r for r in initial}
    baseline = initial
    from .chunk_search_candidates import HybridCandidateSnapshot

    return HybridCandidateSnapshot(
        documents=tuple(documents),
        vector_results=tuple(by_id[pk] for pk in ids["dense"] if pk in by_id),
        trigram_results=tuple(by_id[pk] for pk in ids["trigram"] if pk in by_id),
        exact_results=tuple(by_id[pk] for pk in ids["exact"] if pk in by_id),
        vector_chunk_ids=ids["dense"], trigram_chunk_ids=ids["trigram"],
        exact_chunk_ids=ids["exact"], baseline_candidates=baseline,
        graph_seeds=build_graph_seeds(channels["dense"], channels["trigram"], channels["exact"], config),
        exact_terms=tuple(t.text for t in terms), vector_error=error or initial_error,
        vector_ms=timings["vector"], trigram_ms=timings["trigram"], exact_ms=timings["exact"],
        pre_dedupe_count=sum(len(rs) for rs in channels.values()), graph_seed_error=False,
        effective_limits=limits, full_text_results=tuple(by_id[pk] for pk in ids["full_text"] if pk in by_id),
        full_text_chunk_ids=ids["full_text"], full_text_coverage=coverage,
        fusion_v2=scope.fusion_v2, source_admission=plan,
    )
