"""Fresh source validation coalesces only an exact projection/artifact pair."""
from copy import deepcopy
from hashlib import sha256

import pytest
from django.core.exceptions import ValidationError
from django.db import connection
from django.test.utils import CaptureQueriesContext

from apps.documents.models import DocumentRepresentationState, RawTextDocument
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.source_deadline import source_preparation_scope
from apps.documents.services.source_loading import SourcePreparationLimited
from apps.knowledge_graph.models import CollectionGraphProjection, GraphArtifact
from apps.knowledge_graph.retrieval import ready_scope_repository_v2 as repository
from apps.knowledge_graph.retrieval.ready_materialization_v2 import (
    _validate_closed_captured_selection,
)
from apps.knowledge_graph.tests.ready_subset_support import ScopeFixture

pytestmark = pytest.mark.django_db(transaction=True)


def configured(monkeypatch):
    fixture = ScopeFixture(monkeypatch)
    for label in ("A", "B"):
        fixture.persist_ready(label, captured=True)
    return fixture


def authorities(fixture):
    return repository.load_authorities_v2(
        authorization=fixture.authorization, settings=fixture.settings,
        representation_snapshot=fixture.snapshot, codec=fixture.codec,
    )


def source_reads(queries):
    statements = [q["sql"] for q in queries.captured_queries]
    document = [sql for sql in statements if 'AS "_canonical"' in sql]
    state = [sql for sql in statements if
             'FROM "' + DocumentRepresentationState._meta.db_table + '"' in sql]
    return document, state


def test_equal_pair_keeps_both_record_checks_but_reads_live_representation_once(monkeypatch):
    fixture = configured(monkeypatch)
    original, checked = repository.validate_record_binding, []

    def binding(record):
        result = original(record)
        checked.append((type(record), record.pk, result))
        return result

    monkeypatch.setattr(repository, "validate_record_binding", binding)
    with CaptureQueriesContext(connection) as queries:
        result = authorities(fixture)
    assert len(result) == 2
    assert [kind for kind, _, _ in checked] == [
        CollectionGraphProjection, GraphArtifact,
        CollectionGraphProjection, GraphArtifact,
    ]
    # Separate payloads decode to distinct exact immutable values, not one object.
    assert checked[0][2] == checked[1][2] and checked[0][2] is not checked[1][2]
    assert checked[2][2] == checked[3][2] and checked[2][2] is not checked[3][2]
    assert checked[0][2] != checked[2][2]
    document, state = source_reads(queries)
    assert len(document) == 2, document
    assert len(state) == 2, state


def test_second_record_payload_is_still_parsed_despite_equal_stored_digest(monkeypatch):
    fixture = configured(monkeypatch)
    original = repository.validate_record_binding
    checked = []

    def invalid_second_record(record):
        checked.append(type(record))
        if type(record) is GraphArtifact:
            record.representation_snapshot = deepcopy(record.representation_snapshot)
            record.representation_snapshot["selections"][0]["document"]["source_hash"] = "0" * 64
        return original(record)

    monkeypatch.setattr(repository, "validate_record_binding", invalid_second_record)
    with pytest.raises(ValidationError):
        authorities(fixture)
    assert checked == [CollectionGraphProjection, GraphArtifact]


def test_source_change_during_second_binding_precedes_fresh_database_validation(monkeypatch):
    fixture = configured(monkeypatch)
    original_binding = repository.validate_record_binding
    original_validation = repository.validate_representation_snapshot
    validated, changed, returned = [], [], []

    def validation(snapshot, *, using):
        validated.append(snapshot)
        assert changed, "the fresh read must follow both record payload checks"
        return original_validation(snapshot, using=using)

    def binding(record):
        snapshot = original_binding(record)
        if type(record) is GraphArtifact and not changed:
            assert not validated
            text = "changed before the fresh representation read"
            RawTextDocument.objects.filter(pk=fixture.documents["A"].pk).update(
                full_text=text, full_text_hash=sha256(text.encode()).hexdigest(),
            )
            changed.append(True)
        return snapshot

    monkeypatch.setattr(repository, "validate_record_binding", binding)
    monkeypatch.setattr(repository, "validate_representation_snapshot", validation)
    with pytest.raises(SourceScopeUnavailable):
        returned.append(authorities(fixture))
    assert changed and len(validated) == 1 and returned == []


@pytest.mark.parametrize("mutation", ("source", "generation", "permission"))
def test_final_authority_load_remains_fresh_after_initial_selection(monkeypatch, mutation):
    fixture = configured(monkeypatch)
    selected = repository._select_closed_ready_scope_v2(
        authorization=fixture.authorization, settings=fixture.settings,
        representation_snapshot=fixture.snapshot,
    )
    assert selected.coverage.ready_documents == 2
    if mutation == "source":
        text = "source changed between independent authority loads"
        RawTextDocument.objects.filter(pk=fixture.documents["A"].pk).update(
            full_text=text, full_text_hash=sha256(text.encode()).hexdigest(),
        )
    elif mutation == "generation":
        from apps.documents.services.representation_history import reference
        from apps.documents.services.representation_snapshot_build import (
            build_canonical_snapshot_representation,
        )

        assert build_canonical_snapshot_representation(
            reference(fixture.documents["A"]), using="default",
        ) == "activated"
    else:
        fixture.policy.rows = ()
    with pytest.raises((SourceScopeUnavailable, ValueError)):
        _validate_closed_captured_selection(
            selected=selected, authorization=fixture.authorization,
            settings=fixture.settings, codec=fixture.codec,
        )


def test_equal_pair_cannot_skip_original_deadline_after_live_read(monkeypatch):
    fixture = configured(monkeypatch)
    original = repository.validate_representation_snapshot
    now, validated, returned = [0.0], [], []

    def expire(snapshot, *, using):
        result = original(snapshot, using=using)
        validated.append(snapshot)
        now[0] = 2.0
        return result

    monkeypatch.setattr(repository, "validate_representation_snapshot", expire)
    with pytest.raises((TimeoutError, SourcePreparationLimited)):
        with source_preparation_scope(1.0, lambda: now[0]):
            returned.append(authorities(fixture))
    assert len(validated) == 1 and returned == []


def test_each_independent_loader_call_repeats_the_live_source_reads(monkeypatch):
    fixture = configured(monkeypatch)
    captures = []
    for _ in range(2):
        with CaptureQueriesContext(connection) as queries:
            assert len(authorities(fixture)) == 2
        captures.append(source_reads(queries))
    assert [(len(document), len(state)) for document, state in captures] == [(2, 2), (2, 2)]
