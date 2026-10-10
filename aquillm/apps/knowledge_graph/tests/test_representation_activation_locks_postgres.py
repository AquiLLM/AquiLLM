"""The graph-first activation seam keeps locks through the caller's leaf update."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from time import monotonic, sleep
from uuid import uuid4

import pytest
from django.db import connection, connections, transaction

from apps.documents.models import RawTextDocument
from apps.knowledge_graph.graph.invalidation import (
    DocumentLifecycleRef,
    locked_document_lifecycle_row,
)
from apps.knowledge_graph.models import GraphArtifact, GraphRebuildRequest
from apps.knowledge_graph.projection.representation_lifecycle import (
    prepare_document_representation_activation,
)
from apps.knowledge_graph.tests.ready_subset_support import ScopeFixture

pytestmark = pytest.mark.django_db(transaction=True)


def test_graph_lock_precedes_leaf_and_caller_rollback_releases_all(monkeypatch):
    fixture = ScopeFixture(monkeypatch)
    projection = fixture.persist_ready("A")
    reference = fixture.snapshot.selections[0].document
    lifecycle = DocumentLifecycleRef(
        reference.model_label, reference.pkid, reference.id
    )
    with pytest.raises(RuntimeError, match="caller atomic"):
        prepare_document_representation_activation(
            reference,
            (7,),
            expected_source_hash=reference.source_hash,
            old_representation_key=None,
            new_representation_key=uuid4(),
            using="default",
        )
    held, release_graph, prepared, release_activation = (Event() for _ in range(4))
    pid = []

    def hold_graph():
        try:
            with transaction.atomic():
                GraphArtifact.objects.select_for_update().get(pk=projection.artifact_id)
                held.set()
                assert release_graph.wait(5)
        finally:
            connections.close_all()

    def activate():
        try:
            with transaction.atomic():
                with connection.cursor() as cursor:
                    cursor.execute("SELECT pg_backend_pid()")
                    pid.append(cursor.fetchone()[0])
                prepare_document_representation_activation(
                    reference,
                    (7,),
                    expected_source_hash=reference.source_hash,
                    old_representation_key=None,
                    new_representation_key=uuid4(),
                    using="default",
                )
                # This is the approved I5 call sequence: graph seam, then leaf.
                with locked_document_lifecycle_row(lifecycle, (7,), using="default"):
                    prepared.set()
                    assert release_activation.wait(5)
                    raise RuntimeError("rollback activation")
        except RuntimeError as error:
            assert str(error) == "rollback activation"
        finally:
            connections.close_all()

    before = GraphRebuildRequest.objects.count()
    with ThreadPoolExecutor(max_workers=2) as pool:
        graph = pool.submit(hold_graph)
        assert held.wait(2)
        activation = pool.submit(activate)
        try:
            stop = monotonic() + 2
            blocked = False
            while monotonic() < stop:
                if pid:
                    with connection.cursor() as cursor:
                        cursor.execute(
                            "SELECT cardinality(pg_blocking_pids(%s))", [pid[0]]
                        )
                        blocked = cursor.fetchone()[0] > 0
                    if blocked:
                        break
                sleep(0.01)
            assert blocked and not prepared.is_set()
            # Waiting for the graph row must not already hold the leaf source row.
            with transaction.atomic():
                RawTextDocument.objects.select_for_update(nowait=True).get(
                    pk=reference.pkid
                )
            release_graph.set()
            assert prepared.wait(2)
            # Preparation's inner savepoint must not release the caller's row lock.
            from django.db import OperationalError

            with pytest.raises(OperationalError), transaction.atomic():
                RawTextDocument.objects.select_for_update(nowait=True).get(
                    pk=reference.pkid
                )
        finally:
            release_graph.set()
            release_activation.set()
        graph.result(timeout=2)
        activation.result(timeout=2)
    assert GraphRebuildRequest.objects.count() == before
    projection.refresh_from_db()
    assert projection.state == "ready"
    assert GraphArtifact.objects.get(pk=projection.artifact_id).status == "active"
