"""Malformed scalar results cannot hide in a bounded dense owner batch."""
from uuid import uuid4

import pytest

from apps.documents.services import embedding_dense_rows as rows
from apps.documents.services import embedding_index_query as query, embedding_index_store as store
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.representation_store import freeze_active_representations
from lib.embeddings.tests.test_document_provider import transport as transport
from .dense_sweep_support import corpus, freeze, build, dense_reads, assert_scalar_reads
from .representation_support import reference
from .test_embedding_generation import legacy as legacy, stage

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("malformation", ["duplicate_below_cap", "overflow", "unexpected_owner", "duplicate_pk", "missing_field"])
def test_bad_scalar_batch_is_rejected_before_mapping(malformation, transport, monkeypatch, settings):
    case = corpus(2, transport[0], monkeypatch, settings)
    build(case, 0)
    build(case, 1)
    original = rows._batch_rows
    observations = []

    def malformed(identities, *, using):
        actual = original(identities, using=using)
        observations.append((identities, actual))
        assert len(actual) == len(identities) == 2
        changed = [dict(row) for row in actual]
        if malformation == "duplicate_below_cap":
            changed[1] = dict(changed[0])
        elif malformation == "overflow":
            changed.append(dict(changed[0]))
        elif malformation == "unexpected_owner":
            changed[0]["document_id"] = uuid4()
        elif malformation == "duplicate_pk":
            changed[1]["pk"] = changed[0]["pk"]
        else:
            changed[0].pop("active_generation__key")
        return tuple(changed)

    monkeypatch.setattr(rows, "_batch_rows", malformed)
    with dense_reads() as statements:
        with pytest.raises(SourceScopeUnavailable, match="^source_scope_unavailable$"):
            freeze(case)
    assert len(observations) == 1
    assert_scalar_reads(statements, 1)
    assert "LIMIT 3" in statements[0]


def test_same_batch_retains_genuine_null_and_nonnull_owners_without_expanding_old_capture(
    legacy, transport, monkeypatch, settings,
):
    case = corpus(1, transport[0], monkeypatch, settings)
    current = build(case)
    legacy_doc, legacy_ref = legacy
    snapshot = freeze_active_representations(
        (reference(case.docs[0]), legacy_ref), using="default")
    selected = {s.document.id: s for s in snapshot.selections}
    assert selected[legacy_doc.id].generation_key is None
    assert selected[case.docs[0].id].generation_key == current.representation_key
    original = query.freeze_dense_generations(snapshot, space_key=case.space.space_key, using="default")
    assert original.selections == (current,)
    _, lease, _, _ = stage(legacy, transport)
    assert store.activate_embedding_generation(lease, using="default") == "activated"
    with dense_reads() as statements:
        captured = query.freeze_dense_generations(snapshot, space_key=case.space.space_key, using="default")
    assert_scalar_reads(statements, 1)
    by_document = {s.document_id: s for s in captured.selections}
    assert set(by_document) == {legacy_doc.id, case.docs[0].id}
    assert by_document[legacy_doc.id].representation_key is None
    assert by_document[legacy_doc.id].generation_key == lease.generation_key
    assert by_document[case.docs[0].id] == current
    with dense_reads() as statements:
        assert query.validate_dense_snapshot(snapshot, original, using="default") == (current.generation_key,)
    assert_scalar_reads(statements, 1)
    assert query.validate_dense_snapshot(snapshot, captured, using="default") == tuple(
        s.generation_key for s in captured.selections)
