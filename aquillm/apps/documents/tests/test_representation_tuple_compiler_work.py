"""Real lazy rows precede bounds on repeated ownership-column compilation."""
from collections import Counter

import pytest
from django.db.models import Exists
from django.db.models.expressions import Col
from django.db.models.sql.compiler import SQLCompiler

from apps.documents.models import RawTextDocument, TextChunk
from apps.documents.services.representation_contracts import RepresentationSnapshotV1
from apps.documents.services.representation_store import _active_chunk_predicates
from lib.embeddings.tests.test_document_provider import transport as transport
from .dense_sweep_support import corpus
from .representation_owner_predicate_support import lazy_query

pytestmark = pytest.mark.django_db(transaction=True, databases=["default"])


def identities(query):
    return list(query.values_list("pk", "doc_id", "representation_generation_id"))


def measured_rows(snapshot, monkeypatch):
    query = _active_chunk_predicates(TextChunk.objects.order_by("pk"), snapshot=snapshot)
    counts = Counter()
    original = SQLCompiler.compile

    def compile_once(compiler, node):
        counts["nodes"] += 1
        if isinstance(node, Col):
            counts["columns"] += 1
        return original(compiler, node)

    # Observe the compiler used by actual SQL execution; never replace a result,
    # pre-evaluate/cache the queryset, or count fixture/setup queries.
    with monkeypatch.context() as observation:
        observation.setattr(SQLCompiler, "compile", compile_once)
        rows = identities(query)
    return rows, counts


def test_current_owner_compiler_work_does_not_scale_with_selected_tuple_count(transport, monkeypatch, settings):
    case = corpus(32, transport[0], monkeypatch, settings)
    expected = identities(TextChunk.objects.filter(doc_id__in=[d.id for d in case.docs]).order_by("pk"))
    assert len(expected) == 32 and len({row[1] for row in expected}) == 32
    assert all(row[2] is not None for row in expected)
    first = case.snapshot.selections[0]
    singleton = RepresentationSnapshotV1((first,))
    narrow, narrow_work = measured_rows(singleton, monkeypatch)
    broad, broad_work = measured_rows(case.snapshot, monkeypatch)
    assert narrow == [row for row in expected if row[1] == first.document.id]
    assert len(narrow) == 1 and broad == expected
    # A 32-fold increase in captured tuples must not cause comparable repeated
    # column compilation. Four-fold headroom allows fixed query/alias overhead;
    # RHS parameters may still scale with sources. This is not a timing claim.
    assert 0 < broad_work["columns"] <= 4 * narrow_work["columns"], {
        "singleton": dict(narrow_work), "selected_32": dict(broad_work),
    }


def test_reused_nested_current_query_clones_stay_exact_after_lazy_owner_change(transport, monkeypatch, settings):
    case = corpus(3, transport[0], monkeypatch, settings)
    victim, second, outer_doc = case.docs
    inner, _, _ = lazy_query(case.snapshot)
    expected = identities(TextChunk.objects.filter(doc_id__in=[d.id for d in case.docs]).order_by("pk"))
    assert len(expected) == 3 and identities(inner) == expected
    original_sql = inner.query.sql_with_params()
    outer = TextChunk.objects.filter(doc_id=outer_doc.id).annotate(
        first=Exists(inner.filter(doc_id=victim.id)),
        second=Exists(inner.filter(doc_id=second.id)),
    ).order_by("pk")
    sibling_clone = outer.all()
    nested_sql = outer.query.sql_with_params()
    assert sibling_clone.query.sql_with_params() == nested_sql
    assert outer.query.sql_with_params() == nested_sql
    assert inner.query.sql_with_params() == original_sql
    assert list(outer.values_list("doc_id", "first", "second")) == [(outer_doc.id, True, True)]
    assert list(sibling_clone.values_list("doc_id", "first", "second")) == [(outer_doc.id, True, True)]

    # An ordered real DB mutation after construction, not a concurrent-race claim.
    # Leave the stored hash unchanged so the lazy canonical owner fence matters.
    assert RawTextDocument.objects.filter(pk=victim.pkid).update(full_text="Changed current bytes") == 1
    assert list(outer.values_list("doc_id", "first", "second")) == [(outer_doc.id, False, True)]
    assert list(sibling_clone.values_list("doc_id", "first", "second")) == [(outer_doc.id, False, True)]
    assert identities(inner) == [row for row in expected if row[1] != victim.id]
    assert inner.query.sql_with_params() == original_sql
