"""Actual database behavior of the private non-null tuple predicate."""
from uuid import UUID, uuid4

import pytest
from django.db import connection
from django.test.utils import CaptureQueriesContext

from apps.documents.models import RawTextDocument, TextChunk
from apps.documents.services.representation_tuple_predicate import ExactTupleMembership
from .representation_owner_predicate_support import legacy_case

pytestmark = pytest.mark.django_db(transaction=True, databases=["default"])


def test_uuid_integer_foreign_key_and_quoted_text_remain_bound_parameters():
    case = legacy_case()
    chosen = case.docs[0]
    title = "O'Reilly %s / quoted source"
    assert RawTextDocument.objects.filter(pk=chosen.pkid).update(title=title) == 1
    predicate = ExactTupleMembership(
        ("id", "pkid", "collection", "full_text_hash", "title"),
        ((chosen.id, chosen.pkid, case.collection.pk, chosen.full_text_hash, title),),
    )
    query = RawTextDocument.objects.filter(predicate).order_by("pk")
    sql, params = query.query.sql_with_params()
    assert list(query.values_list("pkid", "id", "title")) == [(chosen.pkid, chosen.id, title)]
    assert title not in sql and title in params
    assert any(type(value) is UUID and value == chosen.id for value in params)
    assert any(type(value) is int and value == chosen.pkid for value in params)
    assert any(type(value) is int and value == case.collection.pk for value in params)
    assert chosen.full_text_hash in params


def test_empty_membership_is_false_and_its_negation_keeps_real_rows():
    case = legacy_case()
    empty = ExactTupleMembership(("id", "pkid"), ())
    assert list(RawTextDocument.objects.filter(empty)) == []
    actual = list(RawTextDocument.objects.filter(~empty).order_by("pk").values_list("id", flat=True))
    assert actual == [doc.id for doc in case.docs]


@pytest.mark.parametrize("crossed", [False, True])
def test_negated_uuid_pk_pairs_never_exempt_an_independent_cross_product(crossed):
    case = legacy_case()
    first, second, _ = case.docs
    rows = ((first.id, second.pkid), (second.id, first.pkid)) if crossed else ((first.id, first.pkid),)
    predicate = ExactTupleMembership(("id", "pkid"), rows)
    positive = list(RawTextDocument.objects.filter(predicate).order_by("pk").values_list("id", flat=True))
    negative = list(RawTextDocument.objects.filter(~predicate).order_by("pk").values_list("id", flat=True))
    assert positive == ([] if crossed else [first.id])
    assert negative == [doc.id for doc in (case.docs if crossed else case.docs[1:])]


def test_current_null_generation_is_not_admitted_by_nonnull_captured_tuple():
    case = legacy_case()
    chosen = case.docs[0]
    assert TextChunk.objects.filter(doc_id=chosen.id, representation_generation__isnull=True).exists()
    predicate = ExactTupleMembership(("doc_id", "representation_generation_id"), ((chosen.id, uuid4()),))
    assert list(TextChunk.objects.filter(predicate)) == []


@pytest.mark.parametrize(("columns", "rows", "error"), [
    ([], (), ValueError),
    ((), (), ValueError),
    (("id",), [], ValueError),
    (("id",), ([uuid4()],), ValueError),
    (("id", "pkid"), ((uuid4(),),), ValueError),
    (("id",), ((None,),), ValueError),
    (("id",), ((True,),), TypeError),
    (("id",), ((1.5,),), TypeError),
])
def test_malformed_or_nullable_rhs_is_rejected_before_query(columns, rows, error):
    with CaptureQueriesContext(connection) as statements:
        with pytest.raises(error):
            ExactTupleMembership(columns, rows)
    assert not statements


@pytest.mark.parametrize(("column", "value"), [
    ("id", str(uuid4())), ("pkid", "1"), ("title", 42),
])
def test_resolved_column_rejects_wrong_scalar_type_before_sql(column, value):
    with CaptureQueriesContext(connection) as statements:
        with pytest.raises(TypeError):
            query = RawTextDocument.objects.filter(ExactTupleMembership((column,), ((value,),)))
            query.query.sql_with_params()
    assert not statements
