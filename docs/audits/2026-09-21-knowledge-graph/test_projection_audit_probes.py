"""Read-only audit reproductions. These assert the observed defective behavior.

External stores and the broker are fakes; the production orchestration and
validation functions are used unchanged. No database or service is contacted.
"""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from apps.knowledge_graph.projection import generation_audit, reconciler, tasks
from apps.knowledge_graph.projection.django_projection_source import (
    DjangoProjectionRowSource,
)
from apps.knowledge_graph.projection.postgres_repository import (
    PostgresProjectionRepository,
)


def test_reconciliation_leaves_new_outbox_work_unpublished(monkeypatch):
    pending, sent = [], []

    def publish(**kwargs):
        count = len(pending)
        sent.extend(pending)
        pending.clear()
        return SimpleNamespace(published_count=count, failed_count=0)

    def reconcile(**kwargs):
        pending.append("new-projection")
        return SimpleNamespace(examined_count=1, enqueued_count=1)

    monkeypatch.setattr(tasks, "publish_projection_outbox", publish)
    monkeypatch.setattr(tasks, "reconcile_graph_projections", reconcile)
    result = tasks.reconcile_knowledge_graph_projections.run(10, False, 7)
    assert result["enqueued_count"] == 1
    assert result["published_count"] == 0
    assert pending == ["new-projection"] and sent == []


def test_reconciliation_aborts_instead_of_replaying_old_projection_version():
    identifier = uuid4()
    snapshot = {
        "projection": {
            "id": identifier,
            "state": "ready",
            "schema_version": "schema-v1",
            "projection_version": "projection-v1",
            "identifier_key_version": "key-v1",
        },
        **{
            family: ()
            for family in (
                "artifacts", "entities", "memberships", "documents", "chunks",
                "relations", "evidence", "entity_mentions",
            )
        },
    }
    source = DjangoProjectionRowSource(
        "default",
        loader=SimpleNamespace(load=lambda **kwargs: snapshot),
        identifier_key=b"audit-only",
        identifier_key_version="key-v1",
        schema_version="schema-v2",
        projection_version="projection-v1",
    )
    postgres = PostgresProjectionRepository(source=source)
    with pytest.raises(ValueError, match="Django projection version is stale"):
        generation_audit.audit_projection_generation(
            row=SimpleNamespace(id=identifier, state="ready"),
            postgres=postgres,
            graph=object(),
            settings=SimpleNamespace(projection_batch_size=10),
        )


def test_bulk_prune_revisits_deleted_page_and_never_reaches_later_rows(monkeypatch):
    # Mirrors the immutable terminal authority rows returned by the production
    # query: graph deletion does not delete or mark these PostgreSQL rows.
    authorities = (SimpleNamespace(id=1), SimpleNamespace(id=2))
    graph_generations = {1, 2}
    visited = []

    def delete(*, row, graph, settings):
        visited.append(row.id)
        if row.id not in graph_generations:
            return False
        graph_generations.remove(row.id)
        return True

    monkeypatch.setattr(
        reconciler, "_prune_candidates",
        lambda **kwargs: authorities[:kwargs["page_size"]],
    )
    monkeypatch.setattr(reconciler, "_projection_settings", lambda: object())
    monkeypatch.setattr(reconciler, "_memgraph_repository", lambda: object())
    monkeypatch.setattr(reconciler, "_delete_projection_generation", delete)
    first = reconciler.prune_graph_projection_generations(
        page_size=1, retain=1, dry_run=False,
    )
    second = reconciler.prune_graph_projection_generations(
        page_size=1, retain=1, dry_run=False,
    )
    assert (first.deleted_count, second.deleted_count) == (1, 0)
    assert visited == [1, 1]
    assert graph_generations == {2}
