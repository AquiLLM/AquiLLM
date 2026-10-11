"""Real PostgreSQL schema parity and commit-time retained owner enforcement."""
import pytest
from django.core.management import call_command
from django.db import DatabaseError, connection, transaction
from django.contrib.contenttypes.models import ContentType

from apps.documents.models import DocumentRepresentationState, TextChunk
from apps.documents.services.representation_history import capture_legacy, locked_state
from .representation_support import retained as retained, upload as upload

pytestmark = pytest.mark.django_db(transaction=True)


def capture(doc):
    from apps.knowledge_graph.graph.invalidation import DocumentLifecycleRef, locked_document_lifecycle_row
    with locked_document_lifecycle_row(DocumentLifecycleRef(doc._meta.label_lower, doc.pkid, doc.id),
                                        (doc.collection_id,), using="default") as (current, _):
        state = locked_state(current, using="default")
        snapshot = capture_legacy(current, state, using="default")
    return state, snapshot


def test_model_migration_parity():
    call_command("makemigrations", "apps_documents", dry_run=True, check=True, verbosity=1)
    call_command("check", verbosity=0)


def test_captured_history_delete_refused_at_commit_and_exact_document_delete_allowed(retained):
    doc, _, chunk = retained
    state, snapshot = capture(doc)
    for row in (chunk, snapshot, state):
        with pytest.raises(DatabaseError):
            with transaction.atomic():
                type(row).objects.filter(pk=row.pk).delete()
    assert TextChunk.objects.filter(pk=chunk.pk).exists()
    doc.delete()
    assert not DocumentRepresentationState.objects.filter(pk=state.pk).exists()
    assert not TextChunk.objects.filter(pk=chunk.pk).exists()


def test_wrong_contenttype_owner_and_identity_repoint_refused(retained):
    doc, _, _ = retained
    wrong = ContentType.objects.get_for_model(TextChunk)
    with pytest.raises(DatabaseError):
        with transaction.atomic():
            DocumentRepresentationState.objects.create(model_label=doc._meta.label_lower,
                document_pkid=doc.pkid, document_id=doc.id, collection_id=doc.collection_id,
                owner_content_type=wrong)
    state, _ = capture(doc)
    with pytest.raises(DatabaseError):
        with transaction.atomic():
            DocumentRepresentationState.objects.filter(pk=state.pk).update(document_pkid=doc.pkid + 10)


def test_frozen_json_digest_matches_unicode():
    from lib.parsers.canonical_projection import canonical_json
    import json
    value = {"é": ["😀\n中文", 1, None, True], "z": "\\\""}
    with connection.cursor() as cursor:
        cursor.execute("SELECT representation_json_v1(%s::jsonb)", [json.dumps(value)])
        assert cursor.fetchone()[0].encode() == canonical_json(value)


def test_origin_output_and_batch_retained_until_exact_collection_delete(retained):
    from django.db.models.deletion import RestrictedError
    doc, output, chunk = retained
    capture(doc)
    for row in (output, output.item, output.item.batch):
        with pytest.raises(RestrictedError):
            type(row).objects.filter(pk=row.pk).delete()
    doc.collection.delete()
    assert not type(output).objects.filter(pk=output.pk).exists()
    assert not TextChunk.objects.filter(pk=chunk.pk).exists()


def test_raw_owner_only_and_partial_history_delete_fail_at_outer_commit(retained):
    doc, _, chunk = retained
    state, _ = capture(doc)
    for sql, params in (
        (f'DELETE FROM "{doc._meta.db_table}" WHERE pkid=%s', [doc.pkid]),
        ('DELETE FROM aquillm_textchunk WHERE id=%s', [chunk.pk]),
    ):
        with pytest.raises(DatabaseError):
            with transaction.atomic(), connection.cursor() as cursor:
                cursor.execute(sql, params)
    assert DocumentRepresentationState.objects.filter(pk=state.pk).exists()


def test_bounded_collector_counts_history_and_refuses_overflow_before_delete(retained, monkeypatch):
    from apps.knowledge_graph.graph import invalidation
    doc, _, chunk = retained
    state, snapshot = capture(doc)
    collector = invalidation._BoundedDeleteCollector(using="default", origin=doc)
    collector.collect([doc])
    for row in (state, snapshot, chunk):
        assert row in collector.data[type(row)]
    assert not collector.fast_deletes
    counted = collector._bounded_source_count
    collector.collect([doc])
    assert collector._bounded_source_count == counted
    monkeypatch.setattr(invalidation, "_MAX_LIFECYCLE_ROWS", counted - 1)
    with pytest.raises(RuntimeError, match="bounded"):
        invalidation._BoundedDeleteCollector(using="default", origin=doc).collect([doc])
    assert TextChunk.objects.filter(pk=chunk.pk).exists()
    monkeypatch.setattr(invalidation, "_MAX_LIFECYCLE_ROWS", counted)
    monkeypatch.setattr(invalidation._bounded_values, "__kwdefaults__", {"maximum": counted})
    doc.delete()
    assert not TextChunk.objects.filter(pk=chunk.pk).exists()
