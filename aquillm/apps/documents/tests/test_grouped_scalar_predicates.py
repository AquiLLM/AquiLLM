"""Fresh grouped scalar predicates preserve complete owner identity tuples."""
from contextlib import contextmanager
from dataclasses import replace

import pytest
from django.db import connection
from django.db.models.expressions import Col
from django.db.models.sql.compiler import SQLCompiler
from django.test.utils import CaptureQueriesContext

from apps.documents.models import DocumentEmbeddingState
from apps.documents.services import embedding_dense_rows as dense
from apps.documents.services.embedding_index_store import persist_embedding_space
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.representation_validation_rows import validate_state_rows
from apps.documents.tests.dense_sweep_support import corpus
from apps.documents.tests.representation_owner_predicate_support import legacy_case
from apps.documents.tests.representation_support import activate
from apps.documents.tests.representation_support import retained as retained
from apps.documents.tests.representation_support import upload as upload
from apps.documents.tests.test_representation_validation_batches import mixed as mixed
from lib.embeddings.tests.test_document_provider import transport as transport

pytestmark = pytest.mark.django_db(transaction=True)


def insert_dense(selections, space):
    identities = tuple(dense._identity(s, space.space_key) for s in selections)
    for identity in identities:
        DocumentEmbeddingState.objects.get_or_create(**dict(zip(dense._IDENTITY, identity, strict=True)))
    return identities


def read_dense(identities):
    with CaptureQueriesContext(connection) as statements:
        rows = dense._batch_rows(identities, using="default")
    assert len(statements) == 1
    assert len(rows) <= len(identities)
    assert statements[0]["sql"].rstrip().endswith(f"LIMIT {len(identities) + 1}")
    return rows, statements[0]["sql"].split(" WHERE ", 1)[1]


def identities_of(rows):
    return {tuple(row[field] for field in dense._IDENTITY) for row in rows}


@contextmanager
def compiler_work(monkeypatch):
    counts = {"columns": 0}
    original = SQLCompiler.compile

    def compile_node(compiler, node):
        if isinstance(node, Col):
            counts["columns"] += 1
        return original(compiler, node)

    with monkeypatch.context() as observation:
        observation.setattr(SQLCompiler, "compile", compile_node)
        yield counts


@pytest.mark.parametrize("count", [1, 64])
def test_dense_common_fields_compile_once_per_group_after_real_rows(count, transport, monkeypatch, settings):
    case = corpus(count, transport[0], monkeypatch, settings)
    identities = insert_dense(case.snapshot.selections, case.space)
    with compiler_work(monkeypatch) as work:
        rows, _ = read_dense(identities)
    assert identities_of(rows) == set(identities)
    assert all(row["active_generation_id"] is None for row in rows)
    print(f"DENSE_OWNER_COMPILER_WORK owners={count} columns={work['columns']}")
    # Hydrate real exact rows first, then bound repeated column resolution.
    # Parameters may grow with owners; ORM expression work must stay bounded.
    assert 0 < work["columns"] <= 40, work


@pytest.mark.parametrize("count", [1, 64])
def test_state_model_label_compiles_once_after_complete_current_validation(count, transport, monkeypatch, settings):
    case = corpus(count, transport[0], monkeypatch, settings)
    with (
        compiler_work(monkeypatch) as work,
        CaptureQueriesContext(connection) as statements,
    ):
        assert validate_state_rows(case.snapshot.selections, using="default") is None
    assert len(statements) == 1
    sql = statements[0]["sql"]
    assert sql.rstrip().endswith(f"LIMIT {count + 1}")
    print("REPRESENTATION_STATE_COMPILER_WORK "
          f"owners={count} columns={work['columns']}")
    assert 0 < work["columns"] <= 30, work


@pytest.mark.parametrize("index", range(6))
def test_dense_crossed_identity_fields_do_not_admit_real_other_rows(index, transport, monkeypatch, settings):
    case = corpus(2, transport[0], monkeypatch, settings)
    identities = insert_dense(case.snapshot.selections, case.space)
    first, other = identities
    changed = list(first)
    values = ("apps_documents.pdfdocument", other[1], other[2],
              case.collection.pk + 100000, other[4], "f" * 64)
    changed[index] = values[index]
    rows, _ = read_dense((tuple(changed), other))
    assert identities_of(rows) == {other}
    assert DocumentEmbeddingState.objects.filter(document_id=first[2]).exists()


def test_mixed_groups_keep_equal_concrete_pks_and_null_representation(transport, monkeypatch, settings):
    case = corpus(1, transport[0], monkeypatch, settings)
    legacy = legacy_case(mixed=True)
    assert legacy.docs[0].pkid == legacy.docs[1].pkid
    persist_embedding_space(case.space, using="default")
    selections = case.snapshot.selections + legacy.snapshot.selections
    identities = insert_dense(selections, case.space)
    with compiler_work(monkeypatch) as dense_work:
        rows, _ = read_dense(identities)
    assert identities_of(rows) == set(identities)
    assert sum(row["representation_key"] is None for row in rows) == 3
    assert 0 < dense_work["columns"] <= 40, dense_work
    with CaptureQueriesContext(connection) as statements:
        assert validate_state_rows(selections, using="default") is None
    assert len(statements) == 1


@pytest.mark.parametrize("field", ["pkid", "id"])
def test_state_crossed_pairs_remain_unavailable(mixed, field):
    active, legacy, _, snapshot = mixed
    original = next(s for s in snapshot.selections if s.document.id == active.id)
    changed = replace(original, document=replace(original.document, **{field: getattr(legacy, field)}))
    selected = tuple(changed if s is original else s for s in snapshot.selections)
    with pytest.raises(SourceScopeUnavailable, match="^source_scope_unavailable$"):
        validate_state_rows(selected, using="default")
    assert validate_state_rows(snapshot.selections, using="default") is None


def test_state_grouping_does_not_cache_current_active_generation(mixed):
    active, _, output, snapshot = mixed
    assert validate_state_rows(snapshot.selections, using="default") is None
    assert activate(active, output, text=active.full_text, parser="grouped-state-replacement") == "activated"
    with pytest.raises(SourceScopeUnavailable, match="^source_scope_unavailable$"):
        validate_state_rows(snapshot.selections, using="default")


def test_empty_state_selection_still_issues_no_query():
    with CaptureQueriesContext(connection) as statements:
        assert validate_state_rows((), using="default") is None
    assert not statements


def test_dense_null_and_current_representation_keys_do_not_cross_match(
    transport, monkeypatch, settings,
):
    case = corpus(1, transport[0], monkeypatch, settings)
    current, = insert_dense(case.snapshot.selections, case.space)
    legacy = (*current[:4], None, current[5])
    DocumentEmbeddingState.objects.create(
        **dict(zip(dense._IDENTITY, legacy, strict=True)))
    for requested in ((current,), (legacy,), (legacy, current)):
        rows, _ = read_dense(requested)
        assert identities_of(rows) == set(requested)
