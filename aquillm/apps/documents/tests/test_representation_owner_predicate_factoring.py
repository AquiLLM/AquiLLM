"""Bound owner SQL per concrete model without weakening lazy current-source guards."""
from dataclasses import replace
from uuid import uuid4

import pytest
from django.db.models import Exists

from apps.documents.models import (
    DESCENDED_FROM_DOCUMENT, DocumentRepresentationState, PDFDocument, RawTextDocument, TextChunk,
)
from apps.documents.services.representation_contracts import RepresentationSnapshotV1, SourceScopeUnavailable
from apps.documents.services.representation_store import active_chunk_scope, freeze_active_representations
from lib.embeddings.tests.test_document_provider import transport as transport
from .dense_sweep_support import corpus
from .representation_owner_predicate_support import duplicate_owner, lazy_query, legacy_case, owner_shape
from .representation_support import retained as retained, upload as upload, activate, reference
from .test_representation_tool_scope import pair as pair

pytestmark = pytest.mark.django_db(transaction=True, databases=["default"])


@pytest.mark.parametrize("count", [1, 32])
def test_canonical_owner_sql_is_bounded_per_model_not_per_selected_source(count, transport, monkeypatch, settings):
    case = corpus(count, transport[0], monkeypatch, settings)
    query, sql, _ = lazy_query(case.snapshot)
    expected = list(TextChunk.objects.filter(doc_id__in=[doc.id for doc in case.docs],
        representation_generation_id__in=[s.generation_key for s in case.snapshot.selections])
        .order_by("pk").values_list("pk", flat=True))
    assert expected and list(query.values_list("pk", flat=True)) == expected
    represented = {s.document.model_label for s in case.snapshot.selections}
    expected_shape = {model._meta.label_lower: 2 if model._meta.label_lower in represented else 1
                      for model in DESCENDED_FROM_DOCUMENT}
    observed = {"owner_subqueries": owner_shape(sql), "canonical_sha": sql.lower().count("sha256(")}
    assert all(0 < total <= expected_shape[label] for label, total in observed["owner_subqueries"].items()), observed
    assert observed["canonical_sha"] == len(represented), observed
    assert sum(observed["owner_subqueries"].values()) <= len(DESCENDED_FROM_DOCUMENT) + len(represented)


def test_different_concrete_models_with_equal_primary_keys_both_survive():
    case = legacy_case(mixed=True)
    raw, pdf, sibling = case.docs
    assert raw.pkid == pdf.pkid and raw.id != pdf.id
    query, _, _ = lazy_query(case.snapshot)
    assert list(query.values_list("pk", flat=True)) == [chunk.pk for chunk in case.chunks]
    assert set(query.values_list("doc_id", flat=True)) == {raw.id, pdf.id, sibling.id}


@pytest.mark.parametrize("model", [RawTextDocument, PDFDocument])
@pytest.mark.parametrize("mixed", [False, True])
def test_late_same_or_other_concrete_model_collision_removes_only_affected_owner(model, mixed):
    case = legacy_case(mixed=mixed)
    victim = case.docs[0]
    query, _, _ = lazy_query(case.snapshot)
    collision = duplicate_owner(victim, model=model, collection=case.other)
    assert collision.id == victim.id and (type(collision), collision.pkid) != (type(victim), victim.pkid)
    assert list(query.values_list("pk", flat=True)) == [c.pk for c in case.chunks if c.doc_id != victim.id]


def test_late_selected_owner_uuid_shift_cannot_reuse_a_globally_allowed_primary_key():
    case = legacy_case(middle_has_chunk=False)
    victim, shifted, sibling = case.docs
    assert not DocumentRepresentationState.objects.filter(document_id=shifted.id).exists()
    assert not TextChunk.objects.filter(doc_id=shifted.id).exists()
    assert any(s.document.pkid == shifted.pkid and s.document.id == shifted.id for s in case.snapshot.selections)
    query, _, _ = lazy_query(case.snapshot)
    # The real DB allows an imported pre-history owner's identity to change;
    # no state/chunk guard is disabled and the selected PK remains unchanged.
    assert RawTextDocument.objects.filter(pk=shifted.pkid, id=shifted.id).update(id=victim.id) == 1
    assert RawTextDocument.objects.filter(id=victim.id).count() == 2
    assert list(query.values_list("pk", flat=True)) == [c.pk for c in case.chunks if c.doc_id == sibling.id]


@pytest.mark.parametrize("field", ["pkid", "collection_id", "source_hash"])
def test_complete_owner_tuple_cannot_be_replaced_by_cross_product_membership(field):
    case = legacy_case()
    first, second, _ = case.docs
    selections = list(case.snapshot.selections)
    left = next(i for i, s in enumerate(selections) if s.document.id == first.id)
    right = next(i for i, s in enumerate(selections) if s.document.id == second.id)
    if field == "collection_id":
        # Different collection IDs really occur in the selected tuple set.
        second.collection = case.other
        second.save(dont_rechunk=True, update_fields=["collection"])
        snapshot = freeze_active_representations(tuple(reference(d) for d in case.docs), using="default")
        selections = list(snapshot.selections)
    a, b = selections[left], selections[right]
    ref_a = replace(a.document, **{field: getattr(b.document, field)})
    ref_b = replace(b.document, **{field: getattr(a.document, field)})
    selections[left] = replace(a, document=ref_a, canonical_sha256=ref_a.source_hash)
    selections[right] = replace(b, document=ref_b, canonical_sha256=ref_b.source_hash)
    altered = RepresentationSnapshotV1(tuple(selections))
    query, _, _ = lazy_query(altered)
    assert list(query.values_list("pk", flat=True)) == [case.chunks[2].pk]
    with pytest.raises(SourceScopeUnavailable):
        active_chunk_scope(TextChunk.objects.all(), snapshot=altered)


def test_late_canonical_corruption_with_unchanged_stored_hash_is_not_admitted(transport, monkeypatch, settings):
    case = corpus(2, transport[0], monkeypatch, settings)
    victim, sibling = case.docs
    query, _, _ = lazy_query(case.snapshot)
    old_hash = victim.full_text_hash
    assert not TextChunk.objects.filter(doc_id=victim.id, representation_generation__isnull=True,
                                       legacy_source_snapshot__isnull=True).exists()
    assert RawTextDocument.objects.filter(pk=victim.pkid).update(full_text="Changed bytes; same stored digest") == 1
    victim.refresh_from_db()
    assert victim.full_text_hash == old_hash and victim.hash_fn(victim.full_text) != old_hash
    expected = list(TextChunk.objects.filter(doc_id=sibling.id).order_by("pk").values_list("pk", flat=True))
    assert expected and list(query.values_list("pk", flat=True)) == expected


def test_late_collection_move_does_not_remove_unaffected_rows():
    case = legacy_case()
    victim = case.docs[0]
    query, _, _ = lazy_query(case.snapshot)
    victim.collection = case.other
    victim.save(dont_rechunk=True, update_fields=["collection"])
    assert list(query.values_list("pk", flat=True)) == [c.pk for c in case.chunks if c.doc_id != victim.id]


@pytest.mark.parametrize("captured_active", [False, True])
def test_late_state_generation_transition_never_adopts_replacement_rows(pair, captured_active):
    a, b, output, _, b_chunk, _ = pair
    if captured_active:
        assert activate(a, output, text=a.full_text, parser="owner-predicate-first") == "activated"
        a.refresh_from_db()
    snapshot = freeze_active_representations((reference(a), reference(b)), using="default")
    selected = next(s for s in snapshot.selections if s.document.id == a.id)
    assert (selected.generation_key is not None) == captured_active
    query, _, _ = lazy_query(snapshot)
    assert activate(a, output, text=a.full_text, parser="owner-predicate-second") == "activated"
    state = DocumentRepresentationState.objects.get(model_label=a._meta.label_lower,
                                                    document_pkid=a.pkid, document_id=a.id)
    assert state.legacy_closed and state.active_generation_id != selected.generation_key
    assert list(query.values_list("pk", flat=True)) == [b_chunk.pk]
    assert TextChunk.objects.filter(doc_id=a.id, representation_generation_id=state.active_generation_id).exists()


def test_empty_snapshot_and_exact_document_hint_remain_narrow():
    case = legacy_case()
    empty = active_chunk_scope(TextChunk.objects.all(), snapshot=RepresentationSnapshotV1(()))
    assert list(empty) == []
    chosen = case.docs[1]
    query, _, params = lazy_query(case.snapshot, document_id=chosen.id)
    assert list(query.values_list("pk", flat=True)) == [case.chunks[1].pk]
    parameter_text = {str(value) for value in params}
    assert str(chosen.id) in parameter_text
    assert all(str(doc.id) not in parameter_text for doc in (case.docs[0], case.docs[2]))
    for wrong in (uuid4(), str(chosen.id)):
        with pytest.raises(SourceScopeUnavailable):
            lazy_query(case.snapshot, document_id=wrong)


def test_public_document_hint_does_not_skip_other_selected_owner_validation():
    case = legacy_case()
    victim, chosen, _ = case.docs
    victim.full_text = "A supported change to a non-target owner"
    victim.save(dont_rechunk=True)
    with pytest.raises(SourceScopeUnavailable):
        active_chunk_scope(TextChunk.objects.all(), snapshot=case.snapshot, document_id=chosen.id)


def test_nested_chunk_query_keeps_owner_correlation_at_the_inner_chunk():
    case = legacy_case()
    victim, _, sibling = case.docs
    inner, _, _ = lazy_query(case.snapshot)
    outer = TextChunk.objects.filter(pk=case.chunks[2].pk).annotate(
        captured=Exists(inner.filter(pk=case.chunks[0].pk)))
    assert list(outer.values_list("doc_id", "captured")) == [(sibling.id, True)]
    duplicate_owner(victim, model=PDFDocument, collection=case.other)
    assert list(outer.values_list("doc_id", "captured")) == [(sibling.id, False)]
