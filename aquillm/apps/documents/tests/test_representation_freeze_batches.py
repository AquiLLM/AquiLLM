"""Representation freeze batches metadata without relaxing immutable identity."""

from dataclasses import replace
from hashlib import sha256
from uuid import uuid4

import pytest
from django.contrib.contenttypes.models import ContentType
from django.db import connection
from django.test.utils import CaptureQueriesContext

from apps.documents.models import (
    DocumentRepresentationGeneration,
    DocumentRepresentationState,
    PDFDocument,
    RawTextDocument,
)
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.representation_store import (
    freeze_active_representations,
    validate_representation_snapshot,
)

from .representation_support import activate, reference
from .representation_support import retained as retained
from .representation_support import upload as upload
from .test_representation_validation_batches import raw_body_selected

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.fixture
def owners(retained, monkeypatch):
    # Generation lifecycle stays real; suppress only external broker dispatch.
    monkeypatch.setattr(
        "apps.knowledge_graph.services.builds.resume_rebuild_request", lambda *a: None
    )
    active, output, _ = retained
    assert activate(active, output) == "activated"
    active.refresh_from_db()
    documents = [active]
    for index in range(7):
        model = PDFDocument if index % 2 == 0 else RawTextDocument
        text = f"Canonical owner {index}."
        document = model(
            title=f"Owner {index}",
            full_text=text,
            full_text_hash=sha256(text.encode()).hexdigest(),
            collection=active.collection,
            ingested_by=active.ingested_by,
        )
        model.objects.bulk_create([document])
        if index % 2:
            DocumentRepresentationState.objects.create(
                model_label=model._meta.label_lower,
                document_pkid=document.pk,
                document_id=document.id,
                collection_id=document.collection_id,
                owner_content_type=ContentType.objects.get_for_model(model),
                legacy_source_hash=document.full_text_hash,
            )
        documents.append(document)
    return tuple(reversed(documents)), active


def test_freeze_state_reads_do_not_grow_per_owner(owners):
    documents, _ = owners
    with CaptureQueriesContext(connection) as single:
        one = freeze_active_representations((reference(documents[0]),), using="default")
    with CaptureQueriesContext(connection) as many:
        snapshot = freeze_active_representations(
            tuple(reference(d) for d in documents), using="default"
        )
    assert len(single) == len(many) == 3
    assert len(one.selections) == 1
    assert [s.document.id for s in snapshot.selections] == [d.id for d in documents]
    for query in many:
        sql = query["sql"].lower()
        assert "limit " in sql
        for branch in sql.split(" union all "):
            projection = branch.lstrip("(").split(" from ", 1)[0]
            assert not raw_body_selected(projection)
            assert all(
                field not in projection
                for field in (
                    '"canonical_text"',
                    '"manifest"',
                    '"source_binding"',
                    '"media_binding"',
                )
            )


def test_freeze_matches_standalone_selection_and_preserves_active_revision(owners):
    from apps.documents.services.representation_scope import selection_for_document

    documents, active = owners
    refs = tuple(reference(d) for d in documents)
    snapshot = freeze_active_representations(refs, using="default")
    assert snapshot.selections == tuple(
        selection_for_document(ref, using="default") for ref in refs
    )
    selected = next(s for s in snapshot.selections if s.document.id == active.id)
    generation = DocumentRepresentationGeneration.objects.get(
        pk=selected.generation_key
    )
    assert selected.canonical_sha256 == active.full_text_hash
    assert selected.source_revision_sha256 == generation.source_revision_sha256
    assert sum(s.generation_key is None for s in snapshot.selections) == 7


def test_freeze_empty_input_reads_nothing():
    with CaptureQueriesContext(connection) as queries:
        snapshot = freeze_active_representations((), using="default")
    assert snapshot.selections == () and len(queries) == 0


def test_freeze_rejects_duplicate_document_identity(owners):
    documents, _ = owners
    ref = reference(documents[0])
    with pytest.raises(SourceScopeUnavailable):
        freeze_active_representations((ref, ref), using="default")


@pytest.mark.parametrize("change", ["closed", "hash", "null_hash"])
def test_freeze_legacy_state_never_falls_back_on_unavailable_owner(owners, change):
    documents, _ = owners
    document = documents[0]  # A legacy owner with no existing state.
    values = {
        "closed": {"legacy_closed": True},
        "hash": {"legacy_source_hash": "f" * 64},
        "null_hash": {"legacy_source_hash": None},
    }[change]
    DocumentRepresentationState.objects.create(
        model_label=document._meta.label_lower,
        document_pkid=document.pk,
        document_id=document.id,
        collection_id=document.collection_id,
        owner_content_type=ContentType.objects.get_for_model(type(document)),
        **({"legacy_source_hash": document.full_text_hash} | values),
    )
    with pytest.raises(SourceScopeUnavailable):
        freeze_active_representations((reference(document),), using="default")


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_label", "apps_documents.pdfdocument"),
        ("pkid", 2147483647),
        ("collection_id", 2147483647),
        ("source_hash", "f" * 64),
    ],
)
def test_freeze_rejects_crossed_owner_components(owners, field, value):
    _, active = owners
    ref = replace(reference(active), **{field: value})
    with pytest.raises(SourceScopeUnavailable):
        freeze_active_representations((ref,), using="default")


def test_freeze_checks_actual_body_even_if_stored_source_hash_did_not_change(owners):
    documents, _ = owners
    document = documents[0]
    type(document).objects.filter(pk=document.pk).update(
        full_text="Untracked replacement"
    )
    with pytest.raises(SourceScopeUnavailable):
        freeze_active_representations((reference(document),), using="default")


def test_freeze_rejects_cross_model_uuid_collision(owners):
    _, active = owners
    PDFDocument.objects.bulk_create(
        [
            PDFDocument(
                id=active.id,
                title="Ambiguous owner",
                full_text=active.full_text,
                full_text_hash=active.full_text_hash,
                collection=active.collection,
                ingested_by=active.ingested_by,
            )
        ]
    )
    with pytest.raises(SourceScopeUnavailable):
        freeze_active_representations((reference(active),), using="default")


@pytest.mark.parametrize("change", ["revision", "generation"])
def test_freeze_rejects_forged_captured_generation_metadata(owners, change):
    _, active = owners
    snapshot = freeze_active_representations((reference(active),), using="default")
    selected = snapshot.selections[0]
    forged = replace(
        selected,
        **(
            {"source_revision_sha256": "f" * 64}
            if change == "revision"
            else {"generation_key": uuid4()}
        ),
    )
    with pytest.raises(SourceScopeUnavailable):
        validate_representation_snapshot(
            replace(snapshot, selections=(forged,)), using="default"
        )


def test_freeze_stays_immutable_after_real_generation_replacement(owners, retained):
    _, active = owners
    snapshot = freeze_active_representations((reference(active),), using="default")
    selected = snapshot.selections[0]
    assert (
        activate(
            active, retained[1], text=active.full_text, parser="replacement-parser"
        )
        == "activated"
    )
    active.refresh_from_db()
    fresh = freeze_active_representations((reference(active),), using="default")
    assert snapshot.selections[0] == selected
    assert fresh.selections[0].generation_key != selected.generation_key
    assert (
        DocumentRepresentationGeneration.objects.get(pk=selected.generation_key).status
        == "retired"
    )
    with pytest.raises(SourceScopeUnavailable):
        validate_representation_snapshot(snapshot, using="default")
