"""Real pending authority and bounded multi-output hub."""
from concurrent.futures import ThreadPoolExecutor

import pytest
from django.db import connections

from apps.knowledge_graph.models import CollectionEntity, CollectionRelation, RelationMention
from tests.integration.persisted_graph_acquisition_support import acquire_graph
from tests.integration.persisted_graph_corpus_support import HUB, graph_corpus as graph_corpus
from tests.integration.persisted_graph_gateway_support import owned_gateway
from tests.integration.persisted_graph_projection_support import graph_configuration as graph_configuration
from tests.integration.persisted_graph_role_support import projection_roles as projection_roles
from tests.integration.persisted_graph_scope_observation import (
    assert_runtime_bounds, graph_observation, identity_labels, keys_for,
)
from tests.integration.persisted_graph_source_support import upload as upload

pytestmark = pytest.mark.django_db(
    transaction=True, databases=["default", "projection_source", "projection_state"]
)
OTHER = "doi:10.1000/p1-pending"
LATE = "doi:10.1000/p1-late"
QUERY = f"Which papers does {HUB} cite?"


def pending_corpus(corpus, neighbors):
    assert neighbors in (2, 201)
    # Separate late owner prevents another admitted relation's chunk exposing it.
    for first in range(0, neighbors - 1, 20):
        lines = [f"{HUB} cites doi:10.1000/p1-neighbor-{index:03d}."
                 for index in range(first, min(first + 20, neighbors - 1))]
        corpus.add(f"hub-{first}", collection="A", text="\n".join(lines))
    corpus.add("late", collection="A", text=f"{HUB} cites {LATE}. Late answer sentinel.")
    corpus.add("pending", collection="B",
        text=f"{OTHER} cites doi:10.1000/p1-pending-answer. Pending answer sentinel.")
    artifacts = corpus.prepare()
    artifact = next(a for a in artifacts if a.collection_scope_id == corpus.collections["A"].pk)
    assert CollectionEntity.objects.filter(artifact=artifact, status="active").count() == neighbors + 1
    assert CollectionRelation.objects.filter(artifact=artifact, status="active", relation_type="cites").count() == neighbors
    owners = tuple(d for d in corpus.documents() if d.collection_id == corpus.collections["A"].pk)
    assert RelationMention.objects.filter(document_id__in=[d.id for d in owners],
        artifact__status="active").count() == neighbors
    for document in owners:
        for chunk in document.chunks.all():
            assert chunk.representation_generation_id is not None
            assert document.full_text[chunk.start_position:chunk.end_position] == chunk.content
            if document.pk != corpus.sources["late"].document.pk:
                assert "Late answer sentinel" not in chunk.content and LATE not in chunk.content
    late = corpus.sources["late"].document.chunks.get()
    assert late.content == corpus.sources["late"].document.full_text
    assert LATE in late.content and "Late answer sentinel" in late.content
    corpus.project("A")
    assert corpus.pending("B").state == "pending"
    return corpus.sources["hub-0"]


@pytest.mark.parametrize("neighbors", [2, 201])
def test_actual_ready_subset_and_over_capacity_hub_keep_partiality(
    graph_corpus, projection_roles, monkeypatch, settings, tmp_path, neighbors,
):
    corpus = graph_corpus
    source = pending_corpus(corpus, neighbors)
    labels = identity_labels(corpus)
    assert len(keys_for(labels, HUB)) == 1 and len(keys_for(labels, LATE)) == 1
    with owned_gateway(tmp_path) as gateway, graph_observation(monkeypatch) as graph:
        result, observed = acquire_graph(source, gateway, monkeypatch, settings,
            documents=corpus.documents(), query=QUERY, query_entity=("paper", HUB))
    assert_runtime_bounds(result, observed, query=QUERY, documents=corpus.documents(),
                          allow_frontier=neighbors == 201)
    assert result.selection.coverage.selected_documents == len(corpus.documents())
    assert result.selection.coverage.ready_documents == len(corpus.documents()) - 1
    assert result.selection.coverage.reasons == ("not_ready",)
    assert result.handoff.coverage.status == "partial"
    assert "not_ready" in result.handoff.coverage.reasons
    assert observed.inputs and result.handoff.candidates
    assert graph.seeds == keys_for(labels, HUB)
    assert all("Pending answer sentinel" not in row.text for row in observed.inputs)
    if neighbors == 201:
        assert "frontier_budget" in result.handoff.coverage.reasons
        assert not keys_for(labels, LATE).intersection(graph.identities)
        assert all("Late answer sentinel" not in row.text for row in observed.inputs)
        assert 1 < len(graph.identities) <= 200
        groups = [g for arguments, _ in graph.ranks for g in arguments["snapshot"].relation_groups]
        assert groups and all(g.raw_weight > 0 for g in groups)
    else:
        assert keys_for(labels, LATE) <= graph.identities
        assert any("Late answer sentinel" in row.text for row in observed.inputs)


def test_pending_becomes_ready_without_expanding_original_selection(
    graph_corpus, projection_roles, monkeypatch, settings, tmp_path,
):
    corpus = graph_corpus
    source = pending_corpus(corpus, 2)
    published = []

    def worker():
        try:
            return corpus.project("B")
        finally:
            connections.close_all()

    def barrier():
        with ThreadPoolExecutor(max_workers=1) as pool:
            published.extend(pool.submit(worker).result(timeout=3))

    with owned_gateway(tmp_path) as gateway:
        stale, original = acquire_graph(source, gateway, monkeypatch, settings,
            documents=corpus.documents(), query=QUERY, query_entity=("paper", HUB), barrier=barrier)
        assert len(published) == 1 and published[0].state == "ready"
        assert original.references and original.inputs == () and stale.handoff.candidates == ()
        assert original.after_swap_remaining_ms >= 1000
        assert {a.collection_id for group in stale.selection.groups for a in group.scope.projections} == {
            corpus.collections["A"].pk,
        }
        assert_runtime_bounds(stale, original, query=QUERY, documents=corpus.documents())
        query = f"Which paper does {OTHER} cite?"
        current, acquired = acquire_graph(source, gateway, monkeypatch, settings,
            documents=corpus.documents(), query=query, query_entity=("paper", OTHER))
    assert_runtime_bounds(current, acquired, query=query, documents=corpus.documents())
    assert current.selection.coverage.status == "complete"
    assert current.selection.coverage.ready_documents == len(corpus.documents())
    assert {a.collection_id for group in current.selection.groups for a in group.scope.projections} == {
        c.pk for c in corpus.collections.values()
    }
    assert any("Pending answer sentinel" in row.text for row in acquired.inputs)
    assert acquired.representation is not original.representation
    assert acquired.budget is not original.budget
