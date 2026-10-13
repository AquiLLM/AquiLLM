"""Exact current-owner validation with bounded scalar database projections."""
from dataclasses import replace
from hashlib import sha256
from unittest.mock import patch
from uuid import uuid4

import pytest
from django.db import connection
from django.test.utils import CaptureQueriesContext

from apps.documents.models import DESCENDED_FROM_DOCUMENT, PDFDocument, RawTextDocument
from apps.documents.services.representation_contracts import (
    RepresentationSnapshotV1, SourceScopeUnavailable,
)
from apps.documents.services.representation_store import (
    freeze_active_representations, validate_document_reference, validate_representation_snapshot,
)
from .representation_support import retained as retained, upload as upload, activate, reference

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.fixture
def mixed(retained):
    active, output, _ = retained
    assert activate(active, output) == "activated"
    active.refresh_from_db()
    with patch("apps.documents.tasks.create_chunks.delay"):
        legacy = RawTextDocument.objects.create(title="Legacy metadata", full_text="legacy source",
            collection=active.collection, ingested_by=active.ingested_by)
    snapshot = freeze_active_representations((reference(active), reference(legacy)), using="default")
    return active, legacy, output, snapshot


def selected(snapshot, document):
    return next(s for s in snapshot.selections if s.document.id == document.id)


def with_selection(snapshot, changed):
    return replace(snapshot, selections=tuple(changed if s.document.id == changed.document.id else s
                                              for s in snapshot.selections))


def raw_body_selected(projection):
    """Inspect top-level SELECT items, not a hash function's body operand."""
    import re
    items, start, depth, quote = [], 0, 0, None
    for index, character in enumerate(projection):
        if quote:
            if character == quote:
                quote = None
        elif character in {"'", '"'}:
            quote = character
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif character == "," and depth == 0:
            items.append(projection[start:index])
            start = index + 1
    items.append(projection[start:])
    return any(re.fullmatch(r'(?:"[^"]+"\.)?"full_text"(?:\s+as\s+"[^"]+")?',
                           item.strip().removeprefix("select ")) for item in items)


def test_body_projection_probe_distinguishes_hash_operand_from_raw_column():
    hashed = "encode(sha256(convert_to(\"doc\".\"full_text\", 'utf8')), 'hex') as \"_canonical\""
    assert not raw_body_selected('select "doc"."pkid", ' + hashed)
    assert raw_body_selected('select "doc"."pkid", "doc"."full_text", ' + hashed)
    assert raw_body_selected('select "doc"."full_text" as "saved_body", ' + hashed)


def test_batch_queries_are_bounded_scalar_and_do_not_grow_per_owner(mixed):
    active, legacy, _, snapshot = mixed
    single = RepresentationSnapshotV1((selected(snapshot, active),))
    with CaptureQueriesContext(connection) as one:
        assert validate_representation_snapshot(single, using="default") is single
    with CaptureQueriesContext(connection) as two:
        assert validate_representation_snapshot(snapshot, using="default") is snapshot
    assert len(one) == len(two) == 2
    for query in two.captured_queries:
        sql = query["sql"].lower()
        assert "limit " in sql
        branches = sql.split(" union all ")
        for branch in branches:
            projection = branch.lstrip("(").split(" from ", 1)[0]
            assert all(name not in projection for name in (
                '"canonical_text"', '"manifest"', '"source_binding"', '"media_binding"',
                '"embedding"', '"metadata"', '"input_binding"',
            ))
            if "aquillm_representationstate" not in sql:
                assert "sha256(convert_to(" in projection
                assert '"full_text_hash"' in projection
                assert not raw_body_selected(projection)
        if " union all " in sql:
            assert len(branches) == len(DESCENDED_FROM_DOCUMENT)
            assert sql.rstrip().endswith("limit 3")
            for model in DESCENDED_FROM_DOCUMENT:
                assert '"' + model._meta.db_table + '"' in sql
                assert model._meta.label_lower in sql
    assert selected(snapshot, legacy).generation_key is None
    assert selected(snapshot, active).generation_key is not None


def test_empty_snapshot_reads_nothing_and_single_owner_api_remains(mixed):
    active, _, _, _ = mixed
    empty = RepresentationSnapshotV1(())
    with CaptureQueriesContext(connection) as queries:
        assert validate_representation_snapshot(empty, using="default") is empty
    assert not queries
    from apps.documents.services.representation_validation_rows import validate_document_rows
    with CaptureQueriesContext(connection) as direct:
        validate_document_rows((), using="default")
    assert not direct
    ref = reference(active)
    assert validate_document_reference(ref, using="default") is ref
    with pytest.raises(SourceScopeUnavailable):
        validate_document_reference(replace(ref, pkid=ref.pkid + 100000), using="default")


@pytest.mark.parametrize("field,value", [
    ("model_label", "apps_documents.pdfdocument"), ("model_label", "auth.user"),
    ("pkid", 2147483647), ("collection_id", 2147483647),
    ("source_hash", "f" * 64), ("id", None),
])
def test_exact_owner_mismatch_rejected(mixed, field, value):
    active, _, _, snapshot = mixed
    original = selected(snapshot, active)
    ref = replace(original.document, **{field: uuid4() if field == "id" else value})
    changed = replace(original, document=ref,
        canonical_sha256=ref.source_hash if field == "source_hash" else original.canonical_sha256)
    forged = RepresentationSnapshotV1((changed, selected(snapshot, mixed[1])))
    with pytest.raises(SourceScopeUnavailable):
        validate_representation_snapshot(forged, using="default")


@pytest.mark.parametrize("change", ["generation", "revision", "legacy", "missing_state"])
def test_exact_state_and_generation_mismatch_rejected(mixed, change):
    active, legacy, _, snapshot = mixed
    original = selected(snapshot, legacy if change == "missing_state" else active)
    if change in {"generation", "missing_state"}:
        changed = replace(original, generation_key=uuid4(), source_revision_sha256="f" * 64)
    elif change == "revision":
        changed = replace(original, source_revision_sha256="f" * 64)
    else:
        changed = replace(original, generation_key=None, source_revision_sha256=None)
    with pytest.raises(SourceScopeUnavailable):
        validate_representation_snapshot(with_selection(snapshot, changed), using="default")


def test_duplicate_snapshot_and_cross_model_collision_are_not_count_equivalence(mixed):
    active, legacy, _, snapshot = mixed
    duplicate = RepresentationSnapshotV1((selected(snapshot, active), selected(snapshot, active)))
    with CaptureQueriesContext(connection) as queries, pytest.raises(SourceScopeUnavailable):
        validate_representation_snapshot(duplicate, using="default")
    assert not queries
    # No state/history exists for this legacy owner. A duplicate concrete owner
    # must still reject, even if another requested UUID is missing.
    PDFDocument.objects.bulk_create([PDFDocument(id=legacy.id, collection=legacy.collection,
        ingested_by=legacy.ingested_by, title="Ambiguous metadata", full_text=legacy.full_text,
        full_text_hash=sha256(legacy.full_text.encode()).hexdigest())])
    missing = replace(selected(snapshot, active), document=replace(reference(active), id=uuid4()))
    ambiguous = RepresentationSnapshotV1((selected(snapshot, legacy), missing))
    with pytest.raises(SourceScopeUnavailable):
        validate_representation_snapshot(ambiguous, using="default")
    with pytest.raises(SourceScopeUnavailable):
        validate_representation_snapshot(snapshot, using="default")


def test_actual_canonical_bytes_and_later_generation_are_rechecked(mixed):
    active, legacy, output, snapshot = mixed
    RawTextDocument.objects.filter(pk=legacy.pk).update(full_text="same length??")
    with pytest.raises(SourceScopeUnavailable):
        validate_representation_snapshot(snapshot, using="default")
    RawTextDocument.objects.filter(pk=legacy.pk).update(full_text=legacy.full_text)
    assert validate_representation_snapshot(snapshot, using="default") is snapshot
    assert activate(active, output, text=active.full_text, parser="batch-replacement") == "activated"
    with pytest.raises(SourceScopeUnavailable):
        validate_representation_snapshot(snapshot, using="default")


def test_state_rows_do_not_accept_cross_product_owner_fields(mixed):
    active, legacy, _, snapshot = mixed
    from apps.documents.services.representation_validation_rows import validate_state_rows
    original = selected(snapshot, active)
    crossed = replace(original, document=replace(original.document, pkid=legacy.pkid))
    with pytest.raises(SourceScopeUnavailable):
        validate_state_rows((crossed, selected(snapshot, legacy)), using="default")
