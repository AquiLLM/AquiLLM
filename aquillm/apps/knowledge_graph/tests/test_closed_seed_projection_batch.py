"""Exact cross-projection seed SQL keeps each original manifest predicate."""

import pytest

from apps.knowledge_graph.retrieval.extended_seed_repository import ExtendedSeedRepository
from apps.knowledge_graph.retrieval.ready_scope_repository_v2 import select_ready_scope_v2
from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import FrontierBudgetV2
from apps.knowledge_graph.tests.test_graph_candidate_operation_work import graph_frame

pytestmark = pytest.mark.django_db(transaction=True)


def test_projection_seed_batch_uses_four_physical_statements_across_groups(monkeypatch):
    with graph_frame(monkeypatch) as active:
        f, graph, source, scope, _, dependencies, _, _ = active
        selected = select_ready_scope_v2(authorization=scope.authorization,
            settings=dependencies.settings, representation_snapshot=scope.representation,
            requested_document_ids=scope.requested_document_ids)
        projections = tuple(p for g in selected.groups for p in g.scope.projections)
        requests = tuple((p, tuple((c.pk, c.doc_id) for c in f.chunks.values()
                                 if c.doc_id in dict(p.documents))) for p in projections)
        assert len(requests) == 2
        budget = FrontierBudgetV2(turn_budget=source.budget, deadline=graph.clock() + 5,
            clock=graph.clock, max_identities=200, max_edges=1000,
            max_queries=64, max_source_rows=5000, max_wire_bytes=1048576)
        repository = ExtendedSeedRepository(using="default")
        kwargs = dict(authorization=scope.authorization, codec=graph.codec,
                      max_rows=4999, read_budget=budget)
        if hasattr(repository, "_load_seed_identities_batch"):
            result = repository._load_seed_identities_batch(requests=requests, **kwargs)
        else:
            result = {p.projection_id: repository.load_seed_identities(authority=p, chunks=chunks, **kwargs)
                      for p, chunks in requests}
        assert len(result) == 2 and all(any(mapping.values()) for mapping in result.values())
        print(f"projection batch SQL: queries={budget.queries}, rows={budget.source_rows}")
        assert budget.queries == 4


@pytest.mark.parametrize("change", ["projection", "permission"])
def test_batch_rechecks_authority_after_actual_identity_rows(monkeypatch, change):
    from apps.knowledge_graph.models import CollectionEntityDocumentLink, CollectionGraphProjection
    from apps.knowledge_graph.retrieval import closed_seed_projection_batch as repository_module

    with graph_frame(monkeypatch) as active:
        f, graph, _, scope, _, dependencies, _, _ = active
        selected = select_ready_scope_v2(authorization=scope.authorization,
            settings=dependencies.settings, representation_snapshot=scope.representation,
            requested_document_ids=scope.requested_document_ids)
        requests = tuple((p, tuple((c.pk, c.doc_id) for c in f.chunks.values()
                                 if c.doc_id in dict(p.documents)))
                         for g in selected.groups for p in g.scope.projections)
        original, observed = repository_module.seed_query_rows, []

        def read(query, **kwargs):
            rows = original(query, **kwargs)
            if query.model is CollectionEntityDocumentLink:
                assert rows
                observed.append(True)
                if change == "projection":
                    changed = requests[-1][0]
                    CollectionGraphProjection.objects.filter(pk=changed.projection_id).update(membership_epoch=changed.membership_epoch + 1)
                else:
                    f.policy.rows = ()
            return rows

        monkeypatch.setattr(repository_module, "seed_query_rows", read)
        with pytest.raises(ValueError):
            ExtendedSeedRepository(using="default")._load_seed_identities_batch(
                requests=requests, authorization=scope.authorization, codec=graph.codec, max_rows=4999)
        assert observed == [True]


def test_actual_postgres_seed_cancellation_poisoning_cannot_retry(monkeypatch):
    from django.db import DatabaseError
    from apps.knowledge_graph.retrieval.direct_seed_sql import DirectSeedReadTimeout
    from apps.knowledge_graph.retrieval import closed_seed_projection_batch as repository_module

    with graph_frame(monkeypatch) as active:
        f, graph, source, scope, _, dependencies, _, _ = active
        selected = select_ready_scope_v2(authorization=scope.authorization,
            settings=dependencies.settings, representation_snapshot=scope.representation,
            requested_document_ids=scope.requested_document_ids)
        requests = tuple((p, tuple((c.pk, c.doc_id) for c in f.chunks.values()
                                 if c.doc_id in dict(p.documents)))
                         for g in selected.groups for p in g.scope.projections)
        original = repository_module._seed_queries

        def delayed(**kwargs):
            return tuple(q.extra(where=["pg_sleep(4) IS NOT NULL"]) for q in original(**kwargs))

        monkeypatch.setattr(repository_module, "_seed_queries", delayed)
        budget = FrontierBudgetV2(turn_budget=source.budget, deadline=graph.clock() + 2,
            clock=graph.clock, max_identities=200, max_edges=1000,
            max_queries=64, max_source_rows=5000, max_wire_bytes=1048576)
        repository = ExtendedSeedRepository(using="default")
        kwargs = dict(requests=requests, authorization=scope.authorization,
                      codec=graph.codec, max_rows=4999, read_budget=budget)
        with pytest.raises(DirectSeedReadTimeout) as caught:
            repository._load_seed_identities_batch(**kwargs)
        assert isinstance(caught.value.__cause__, DatabaseError)
        assert "statement timeout" in str(caught.value.__cause__)
        assert budget.queries == 3 and budget.source_rows == 4 and budget._uncertain
        reserved = budget._reserved_rows
        assert reserved > 0
        with pytest.raises(TimeoutError):
            repository._load_seed_identities_batch(**kwargs)
        assert (budget.queries, budget.source_rows, budget._reserved_rows) == (3, 4, reserved)
