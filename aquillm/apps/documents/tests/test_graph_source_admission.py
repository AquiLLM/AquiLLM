"""Native V2 leaves body capacity for novel graph evidence on the same ledger."""

from dataclasses import replace
from inspect import signature
from types import SimpleNamespace

import pytest

from apps.documents.models import TextChunk
from apps.documents.services.chunk_search_candidates import collect_hybrid_candidate_snapshot
from apps.documents.services.hybrid_graph_v2 import hybrid_graph_candidate_pool_v2
from apps.knowledge_graph.retrieval.expansion import get_graph_expansion_config
from apps.knowledge_graph.tests.test_graph_candidate_operation_work import graph_frame

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.fixture(autouse=True)
def full_candidate_limit(monkeypatch):
    from django.conf import settings

    monkeypatch.setattr(settings, "RAG_CANDIDATE_MULTIPLIER", 3.0, raising=False)
    monkeypatch.setattr(settings, "RAG_SHORT_QUERY_CANDIDATE_SCALE", 1.0, raising=False)


def pressure_rows(f):
    first = f.chunks["A"]
    text = f.documents["A"].full_text
    ends = tuple(end for end in range(5, 50) if end != first.end_position)
    assert len(ends) == 44
    TextChunk.objects.bulk_create(
        [
            TextChunk(
                doc_id=first.doc_id, content=text[:end], chunk_number=100 + number,
                start_position=0, end_position=end,
            )
            for number, end in enumerate(ends)
        ]
    )


def test_pressure_rows_never_requests_embeddings(monkeypatch):
    attempts = []

    def unexpected_embedding(chunk, callback=None):
        attempts.append(chunk.content)
        raise AssertionError("lexical pressure must not request an embedding")

    monkeypatch.setattr(TextChunk, "get_chunk_embedding", unexpected_embedding)
    with graph_frame(monkeypatch, prepare_sources=pressure_rows, dataset=True) as active:
        doc_id = active[0].chunks["A"].doc_id
        rows = list(
            TextChunk.objects.filter(doc_id=doc_id, chunk_number__gte=100)
            .order_by("chunk_number")
            .values_list("content", "embedding")
        )
        assert len(rows) == 44
        assert all(content and embedding is None for content, embedding in rows)
        assert attempts == []


def capture(active, *, native=True):
    from lib.embeddings.document_contracts import digest

    f, _, _, scope, snapshot, dependencies, _, _ = active
    scope.query_digest = digest({"version": "unprepared-query-v1", "query": "Atlas"})
    dependencies.settings.memgraph_traversal_enabled = True
    return collect_hybrid_candidate_snapshot(
        TextChunk, "Atlas", 20, snapshot.documents,
        query_embedding=None, graph_config=get_graph_expansion_config(),
        app_config_getter=lambda _: SimpleNamespace(vector_top_k=45, trigram_top_k=45),
        authorization_context=f.authorization,
        **({"_graph_dependencies": dependencies} if native and
           "_graph_dependencies" in signature(collect_hybrid_candidate_snapshot).parameters else {}),
    )


def test_real_45_source_pressure_admits_novel_graph_without_raising_cap(monkeypatch):
    with graph_frame(monkeypatch, prepare_sources=pressure_rows, dataset=True) as active:
        f, _, source, scope, _, dependencies, _, _ = active
        snapshot = capture(active)
        assert len(snapshot.trigram_chunk_ids) == 45
        assert source.budget.sources_used == 25
        assert len(snapshot.baseline_candidates) == 25
        rows, diagnostics = hybrid_graph_candidate_pool_v2(
            snapshot, "Atlas", f.authorization, dependencies
        )
        assert diagnostics["graph_status"] == "hit"
        assert f.chunks["B"].pk in {row.pk for row in rows}
        assert source.budget.sources_used <= 45
        assert set(scope.bodies) >= {row.pk for row in rows}
        assert diagnostics["source_admission"]["policy"] == "native-v2-channel-admission-v1"


def test_graph_off_collector_keeps_existing_full_body_prefix(monkeypatch):
    with graph_frame(monkeypatch, prepare_sources=pressure_rows, dataset=True) as active:
        snapshot = capture(active, native=False)
        source = active[2]
        assert len(snapshot.baseline_candidates) == 45
        assert source.budget.sources_used == 45
        assert getattr(snapshot, "source_admission", None) is None


@pytest.mark.parametrize("kind", ["wrapped", "legacy", "graph_off"])
def test_custom_and_non_v2_collectors_keep_existing_admission(monkeypatch, kind):
    with graph_frame(monkeypatch, prepare_sources=pressure_rows, dataset=True) as active:
        _, graph, source, scope, _, dependencies, _, _ = active
        if kind == "wrapped":
            original = graph.run_v2
            graph.run_v2 = lambda **kwargs: original(**kwargs)
        elif kind == "legacy":
            dependencies.settings.graph_remediation_mode = "legacy"
        else:
            scope.graph_enabled = False
        snapshot = capture(active)
        assert snapshot.source_admission is None
        assert len(snapshot.baseline_candidates) == source.budget.sources_used == 45


@pytest.mark.parametrize("mutation", ["permission", "source", "operation"])
def test_metadata_loss_cannot_publish_body_or_positive_cache(monkeypatch, mutation):
    from apps.documents.services import graph_candidate_metadata as metadata
    from apps.documents.services.representation_contracts import SourceScopeUnavailable

    with graph_frame(monkeypatch, prepare_sources=pressure_rows, dataset=True) as active:
        f, _, source, scope, _, _, operation, _ = active
        original = metadata.rank_refs

        def observe(*args, **kwargs):
            refs = original(*args, **kwargs)
            assert source.budget.sources_used == 0
            assert all(not hasattr(ref, "content") for ref in refs)
            if mutation == "permission":
                f.policy.rows = ()
            elif mutation == "source":
                doc = f.documents["A"]
                doc.full_text = doc.full_text.replace("Atlas", "Cedar")
                doc.save()
            else:
                operation.live = False
            return refs

        monkeypatch.setattr(metadata, "rank_refs", observe)
        with pytest.raises((SourceScopeUnavailable, TimeoutError)):
            capture(active)
        assert source.cache == {} and scope.bodies == {}
        assert source.budget.sources_used == 0


def test_deferred_top_up_does_not_reacquire_rank_queries(monkeypatch):
    from apps.documents.services import graph_candidate_admission as metadata

    with graph_frame(monkeypatch, prepare_sources=pressure_rows, dataset=True) as active:
        snapshot = capture(active)
        plan, source = snapshot.source_admission, active[2]
        assert source.budget.sources_used == 25
        monkeypatch.setattr(metadata, "capture_metadata", lambda *a, **kw: pytest.fail("rank reacquisition"))
        # The captured plan itself issues fresh exact metadata/body/authority reads.
        top_up = plan.top_up()
        assert len(top_up) == 20
        assert source.budget.sources_used == 45
        assert set(r.pk for r in top_up).isdisjoint(plan.initial_ids)


@pytest.mark.parametrize("mutation", ["permission", "operation", "context"])
def test_late_top_up_requires_original_live_authority(monkeypatch, mutation):
    from apps.documents.services.representation_contracts import SourceScopeUnavailable
    from apps.documents.services.chunk_search_scope import bind_search_scope, reset_search_scope

    with graph_frame(monkeypatch, prepare_sources=pressure_rows, dataset=True) as active:
        f, _, source, scope, _, _, operation, _ = active
        snapshot = capture(active)
        before = source.cache.copy()
        token = None
        if mutation == "permission":
            f.policy.rows = ()
        elif mutation == "operation":
            operation.live = False
        else:
            token = bind_search_scope(replace(scope), "What is A?", graph_enabled=True)
        try:
            with pytest.raises((SourceScopeUnavailable, TimeoutError)):
                snapshot.source_admission.top_up()
            assert source.cache == before
            assert source.budget.sources_used == 25
        finally:
            if token is not None:
                reset_search_scope(token)


def test_graph_failure_after_real_novel_reads_never_refunds_or_leaks_graph_cache(monkeypatch):
    from apps.knowledge_graph.retrieval import ready_materialization_v2 as materializer
    from apps.documents.services.graph_source_admission_policy import charged_sources

    with graph_frame(monkeypatch, prepare_sources=pressure_rows, dataset=True) as active:
        f, _, source, scope, _, dependencies, _, _ = active
        snapshot = capture(active)
        before, charged_at_failure, spent = source.budget.sources_used, [], []

        def fail(**kwargs):
            charged_at_failure.append(source.budget.sources_used)
            assert source.budget.sources_used > before
            spent.append(charged_sources(source.budget) - snapshot.source_admission.graph_start)
            raise ValueError("synthetic failure after actual selector hydration")

        monkeypatch.setattr(materializer, "_materialize_rows", fail)
        rows, diagnostics = hybrid_graph_candidate_pool_v2(snapshot, "Atlas", f.authorization, dependencies)
        assert diagnostics["graph_status"] == "error"
        assert charged_at_failure and source.budget.sources_used == 45
        assert f.chunks["B"].pk not in {r.pk for r in rows}
        assert f.chunks["B"].pk not in {r.pk for r in source.cache.values()}
        # Reused captured baseline identities are eligible, while every graph
        # input outside captured baseline ranks remains spent without a refund.
        captured = {r.identity for r in snapshot.source_admission.references.values()}
        outside = spent[0] - captured
        assert any(pk == f.chunks["B"].pk for pk, _ in outside)
        assert len(rows) == 45 - len(outside)
        assert set(scope.bodies) == {r.pk for r in rows}
