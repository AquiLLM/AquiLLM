"""Scalar payload bytes and rejection rules do not require recursive UUID copies."""
from dataclasses import asdict, replace
from uuid import UUID

import pytest

from apps.documents.services.representation_contracts import (
    DocumentRefV1, RepresentationSelectionV1, RepresentationSnapshotV1,
    SourceScopeUnavailable,
)
from apps.documents.services.representation_store import document_payload, representation_snapshot_digest


def reference():
    return DocumentRefV1("apps_documents.rawtextdocument", 1,
        UUID("11111111-1111-4111-8111-111111111111"), 2, "a" * 64)


def test_payload_preserves_exact_values_order_and_independent_container():
    ref = reference()
    expected = {**asdict(ref), "id": str(ref.id)}
    payload = document_payload(ref)
    assert payload == expected
    assert tuple(payload) == ("model_label", "pkid", "id", "collection_id", "source_hash")
    assert type(payload["id"]) is str
    payload["pkid"] = 999
    assert ref.pkid == 1 and document_payload(ref) == expected


def test_legacy_null_snapshot_retains_existing_golden_digest():
    ref = reference()
    snapshot = RepresentationSnapshotV1((RepresentationSelectionV1(ref, None, "a" * 64, None),))
    assert representation_snapshot_digest(snapshot) == "1fb03a912b9dae3e7783d36a778d7268d5682ebc2fbea00300e654bee5ff8795"


def test_nonnull_snapshot_digest_preserves_reference_serialization_and_order():
    from lib.embeddings.document_contracts import digest
    first = RepresentationSelectionV1(reference(), UUID("22222222-2222-4222-8222-222222222222"), "a" * 64, "b" * 64)
    second = RepresentationSelectionV1(replace(reference(), pkid=3,
        id=UUID("33333333-3333-4333-8333-333333333333")), None, "a" * 64, None)
    expected = digest({"version": "representation-snapshot-v1", "selections": [
        {"document": {**asdict(row.document), "id": str(row.document.id)},
         "generation_key": str(row.generation_key) if row.generation_key else None,
         "canonical_sha256": row.canonical_sha256, "source_revision_sha256": row.source_revision_sha256}
        for row in (first, second)]})
    assert representation_snapshot_digest(RepresentationSnapshotV1((first, second))) == expected
    assert representation_snapshot_digest(RepresentationSnapshotV1((second, first))) == expected


class DerivedReference(DocumentRefV1):
    pass


class DerivedString(str):
    pass


@pytest.mark.parametrize("field,value", [
    ("model_label", "apps_documents.RawTextDocument"), ("model_label", "rawtextdocument"),
    ("model_label", DerivedString("apps_documents.rawtextdocument")),
    ("pkid", True), ("pkid", 0), ("pkid", -1), ("pkid", 1.0),
    ("collection_id", True), ("collection_id", 0), ("collection_id", -1),
    ("id", "11111111-1111-4111-8111-111111111111"), ("id", None),
    ("source_hash", "A" * 64), ("source_hash", "a" * 63),
    ("source_hash", b"a" * 64), ("source_hash", DerivedString("a" * 64)),
])
def test_malformed_scalar_reference_remains_unavailable(field, value):
    with pytest.raises(SourceScopeUnavailable, match="^source_scope_unavailable$"):
        document_payload(replace(reference(), **{field: value}))


@pytest.mark.parametrize("ref", [None, {}, DerivedReference(
    "apps_documents.rawtextdocument", 1, UUID("11111111-1111-4111-8111-111111111111"), 2, "a" * 64)])
def test_only_exact_reference_type_is_accepted(ref):
    with pytest.raises(SourceScopeUnavailable, match="^source_scope_unavailable$"):
        document_payload(ref)


def test_real_payloads_avoid_recursive_uuid_copy_after_value_equivalence(monkeypatch):
    import copy
    refs = tuple(replace(reference(), pkid=index + 1, id=UUID(int=index + 1)) for index in range(32))
    expected = tuple({**asdict(ref), "id": str(ref.id)} for ref in refs)
    original, visits = copy.deepcopy, []
    def observed(value, memo=None):
        if type(value) is UUID:
            visits.append(value)
        return original(value, memo)
    monkeypatch.setattr(copy, "deepcopy", observed)
    actual = tuple(document_payload(ref) for ref in refs)
    assert actual == expected
    assert visits == []
