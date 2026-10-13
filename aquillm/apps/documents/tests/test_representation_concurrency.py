"""Separate connections race a captured activation with real owner transitions."""
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from threading import Event, current_thread

import pytest
from django.db import close_old_connections, connection

from apps.documents.models import DocumentRepresentationState, RawTextDocument, TextChunk
from apps.documents.services import representation_activation as activation
from apps.documents.services.representation_snapshot_build import build_canonical_snapshot_representation
from .representation_support import retained as retained, upload as upload, reference
from .test_representation_edits import manual
from .test_representation_lifecycle import destination, staged

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("mutation", ["edit", "move", "delete", "pointer"])
def test_competing_connection_invalidates_captured_activation(retained, monkeypatch, mutation):
    doc = manual(retained)
    old_pks = tuple(TextChunk.objects.filter(doc_id=doc.id).values_list("pk", flat=True))
    lease, predecessor = staged(doc)
    captured, resume = Event(), Event()
    transition = activation.locked_document_representation_transition
    parent_thread = current_thread()
    worker_pids = []

    @contextmanager
    def barrier(*args, **kwargs):
        if current_thread() is not parent_thread:
            with connection.cursor() as cursor:
                cursor.execute("SELECT pg_backend_pid()")
                worker_pids.append(cursor.fetchone()[0])
            captured.set()
            if not resume.wait(20):
                raise RuntimeError("activation test barrier timed out")
        with transition(*args, **kwargs) as value:
            yield value

    def run():
        close_old_connections()
        try:
            return activation.activate_representation(lease, expected_active_key=predecessor, using="default")
        finally:
            connection.close()

    monkeypatch.setattr(activation, "locked_document_representation_transition", barrier)
    with connection.cursor() as cursor:
        cursor.execute("SELECT pg_backend_pid()")
        parent_pid = cursor.fetchone()[0]
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run)
        try:
            assert captured.wait(20)
            assert worker_pids and worker_pids[0] != parent_pid
            if mutation == "edit":
                doc.full_text = "edited on the competing connection"
                doc.save(dont_rechunk=True)
            elif mutation == "move":
                doc.move_to(destination(doc), actor=doc.ingested_by)
            elif mutation == "delete":
                doc.delete()
            else:
                assert build_canonical_snapshot_representation(reference(doc), using="default") == "activated"
        finally:
            resume.set()
        assert future.result(timeout=20) == "stale"
    if mutation == "delete":
        assert not DocumentRepresentationState.objects.filter(document_id=doc.id).exists()
        assert not TextChunk.objects.filter(doc_id=doc.id).exists()
    else:
        state = DocumentRepresentationState.objects.get(document_id=doc.id)
        assert state.active_generation_id != lease.generation_key
        assert TextChunk.objects.filter(pk__in=old_pks).count() == len(old_pks)
        if mutation == "edit":
            assert RawTextDocument.objects.get(pk=doc.pk).full_text == doc.full_text
