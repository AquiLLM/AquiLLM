"""Keep current-state SQL bounded while every captured owner tuple stays exact."""
from dataclasses import replace

import pytest
from django.db import connection
from django.db.models import Exists

from apps.documents.models import DocumentRepresentationState, TextChunk
from apps.documents.services.representation_contracts import RepresentationSnapshotV1
from apps.documents.services.representation_store import freeze_active_representations
from lib.embeddings.tests.test_document_provider import transport as transport
from .dense_sweep_support import corpus
from .representation_owner_predicate_support import lazy_query, legacy_case
from .representation_support import retained as retained, upload as upload, activate, reference
from .test_representation_tool_scope import pair as pair

pytestmark = pytest.mark.django_db(transaction=True, databases=["default"])


def identities(query):
    return list(query.values_list("pk", "doc_id", "representation_generation_id"))


@pytest.mark.parametrize("count", [1, 4])
def test_current_state_query_cost_is_bounded_after_real_rows_are_returned(count, transport, monkeypatch, settings):
    case = corpus(count, transport[0], monkeypatch, settings)
    query, sql, _ = lazy_query(case.snapshot)
    expected = identities(TextChunk.objects.filter(doc_id__in=[d.id for d in case.docs]).order_by("pk"))
    assert len(expected) == count and identities(query) == expected
    # Count actual compiled state scans only after the real query returned all rows.
    # Construction itself performed no SQL (the shared lazy_query assertion).
    state_scans = sql.count("FROM " + connection.ops.quote_name(DocumentRepresentationState._meta.db_table))
    assert 0 < state_scans <= 1, {"selected": count, "state_scans": state_scans}


@pytest.mark.parametrize("hint", [None, "current", "legacy"])
def test_current_and_missing_legacy_states_keep_exact_requested_document_hint(hint, transport, monkeypatch, settings):
    current = corpus(2, transport[0], monkeypatch, settings)
    legacy = legacy_case(mixed=True)
    assert all(s.generation_key is None for s in legacy.snapshot.selections)
    assert not DocumentRepresentationState.objects.filter(document_id__in=[d.id for d in legacy.docs]).exists()
    snapshot = RepresentationSnapshotV1(current.snapshot.selections + legacy.snapshot.selections)
    chosen = current.docs[0] if hint == "current" else legacy.docs[1] if hint == "legacy" else None
    query, _, _ = lazy_query(snapshot, document_id=chosen.id if chosen is not None else None)
    ids = [chosen.id] if chosen is not None else [d.id for d in (*current.docs, *legacy.docs)]
    expected = identities(TextChunk.objects.filter(doc_id__in=ids).order_by("pk"))
    assert expected and identities(query) == expected
    assert all(row[1] in ids for row in identities(query))


@pytest.mark.parametrize("field", ["generation_key", "source_revision_sha256"])
def test_other_selected_generation_or_revision_cannot_complete_a_crossed_tuple(field, transport, monkeypatch, settings):
    case = corpus(3, transport[0], monkeypatch, settings)
    first, second, survivor = case.snapshot.selections
    assert getattr(first, field) != getattr(second, field)
    selections = (replace(first, **{field: getattr(second, field)}),
                  replace(second, **{field: getattr(first, field)}), survivor)
    query, _, _ = lazy_query(RepresentationSnapshotV1(selections))
    expected = identities(TextChunk.objects.filter(doc_id=survivor.document.id).order_by("pk"))
    assert len(expected) == 1 and identities(query) == expected


def test_nested_current_state_correlation_remains_lazy_after_generation_replacement(pair):
    a, b, output, _, b_chunk, _ = pair
    assert activate(a, output, text=a.full_text, parser="state-factor-first") == "activated"
    a.refresh_from_db()
    snapshot = freeze_active_representations((reference(a), reference(b)), using="default")
    selected = next(s for s in snapshot.selections if s.document.id == a.id)
    assert selected.generation_key is not None
    assert next(s for s in snapshot.selections if s.document.id == b.id).generation_key is None
    inner, _, _ = lazy_query(snapshot)
    outer = TextChunk.objects.filter(pk=b_chunk.pk).annotate(captured=Exists(inner.filter(doc_id=a.id)))
    assert list(outer.values_list("doc_id", "captured")) == [(b.id, True)]
    # Ordered committed mutation between construction and evaluation, not a concurrent-race claim.
    assert activate(a, output, text=a.full_text, parser="state-factor-second") == "activated"
    state = DocumentRepresentationState.objects.get(model_label=a._meta.label_lower,
        document_pkid=a.pkid, document_id=a.id)
    assert state.active_generation_id != selected.generation_key and state.legacy_closed
    assert list(outer.values_list("doc_id", "captured")) == [(b.id, False)]
    assert identities(inner) == [(b_chunk.pk, b.id, None)]
