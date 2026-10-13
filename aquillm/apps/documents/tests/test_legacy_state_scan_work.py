"""Bound legacy state scans while retaining lazy exact-owner invalidation."""

import pytest
from django.contrib.contenttypes.models import ContentType
from django.db import connection, transaction
from django.db.models import Exists

from apps.documents.models import DocumentRepresentationState, TextChunk

from .representation_owner_predicate_support import lazy_query, legacy_case

pytestmark = pytest.mark.django_db(transaction=True, databases=["default"])


def state_for(doc, **changes):
    values = dict(
        model_label=doc._meta.label_lower,
        document_pkid=doc.pkid,
        document_id=doc.id,
        collection_id=doc.collection_id,
        legacy_source_hash=doc.full_text_hash,
        legacy_closed=False,
        owner_content_type=ContentType.objects.get_for_model(doc),
    )
    values.update(changes)
    return DocumentRepresentationState.objects.create(**values)


def test_legacy_state_scan_is_single_after_all_real_rows_return():
    case = legacy_case(mixed=True)
    for doc in case.docs:
        state_for(doc)
    query, sql, _ = lazy_query(case.snapshot)
    assert list(query.values_list("pk", flat=True)) == [c.pk for c in case.chunks]
    scans = sql.count(
        "FROM " + connection.ops.quote_name(DocumentRepresentationState._meta.db_table)
    )
    assert scans == 1


@pytest.mark.parametrize(
    "change", ["legacy_closed", "legacy_source_hash", "null_hash", "unavailable"]
)
def test_lazy_legacy_state_mutation_excludes_only_exact_owner(change):
    case = legacy_case(mixed=True)
    a, b, c = case.docs
    query, _, _ = lazy_query(case.snapshot)
    if change == "unavailable":
        from apps.documents.services.representation_history import capture_legacy

        TextChunk.objects.filter(doc_id=b.id).update(end_position=99999)
        with transaction.atomic():
            capture_legacy(b, state_for(b), using="default")
    elif change == "null_hash":
        state_for(b, legacy_source_hash=None)
    else:
        state_for(
            b, **{change: True if change == "legacy_closed" else a.full_text_hash}
        )
    assert list(query.values_list("doc_id", flat=True)) == [a.id, c.id]


def test_same_pk_other_concrete_owner_state_cannot_close_selected_legacy_owner():
    case = legacy_case(mixed=True)
    a, b, _ = case.docs
    assert a.pkid == b.pkid
    state_for(b, legacy_closed=True)
    query, _, _ = lazy_query(case.snapshot)
    assert list(query.values_list("doc_id", flat=True)) == [a.id, case.docs[2].id]


def test_nested_legacy_scope_is_lazy_after_state_creation():
    case = legacy_case(mixed=True)
    a, b, _ = case.docs
    inner, _, _ = lazy_query(case.snapshot)
    outer = TextChunk.objects.filter(doc_id=a.id).annotate(
        present=Exists(inner.filter(doc_id=b.id))
    )
    assert list(outer.values_list("present", flat=True)) == [True]
    state_for(b, legacy_closed=True)
    assert list(outer.values_list("present", flat=True)) == [False]
