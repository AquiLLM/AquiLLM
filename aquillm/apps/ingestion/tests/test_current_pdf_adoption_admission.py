"""Reject invalid current-adoption requests before attached-source I/O."""
from dataclasses import replace
from hashlib import sha256
import pytest
from django.db import connection
from apps.collections.models import Collection, CollectionPermission
from apps.documents.models import PDFDocument
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.ingestion.models import SourceArtifact, SourceArtifactOccurrence, ExtractionRevision
from apps.ingestion.services.source_artifacts import SourceArtifactIntegrityError
from .current_pdf_adoption_support import (
    ledger_counts, owner_snapshot, reference, service, existing_representation, synthetic_pdf,
    pdf_adoption_case as pdf_adoption_case, s3 as s3,
)

pytestmark = pytest.mark.django_db(transaction=True)
REJECTED = (SourceScopeUnavailable, SourceArtifactIntegrityError)


def adopt(case, **kwargs):
    return service().adopt_current_pdf(case.ref, actor_id=case.actor.pk, using="default", **kwargs)


@pytest.mark.parametrize("change", ["stored_hash", "codepoints", "utf8_bytes"])
def test_invalid_canonical_is_rejected_before_attached_source_io(pdf_adoption_case, change):
    case = pdf_adoption_case
    retained = case.document
    # Malformed legacy admission starts with a distinct, initially chunkless
    # owner. Never delete retained citation rows or bypass their lifecycle guard.
    if change == "stored_hash":
        text, digest = "Malformed legacy canonical", "0" * 64
    else:
        text = "x" * 1_000_001 if change == "codepoints" else "\U0001f642" * (2 * 1024 * 1024 + 1)
        digest = sha256(text.encode()).hexdigest()
    case.document = PDFDocument.objects.bulk_create([PDFDocument(title="Malformed legacy owner",
        full_text=text, full_text_hash=digest, pdf_file="attached/current.pdf",
        collection=case.collection, ingested_by=case.actor)])[0]
    case.ref = reference(case.document)
    before = owner_snapshot(retained)
    def forbid_oversize_body_transfer(execute, sql, params, many, context):
        result = execute(sql, params, many, context)
        description = context["cursor"].description
        if change != "stored_hash" and description:
            assert "full_text" not in {column[0] for column in description}, (
                "oversize canonical must fail SQL size checks before body materialization")
        return result

    with connection.execute_wrapper(forbid_oversize_body_transfer), pytest.raises(REJECTED):
        adopt(case)
    assert case.transport.calls == [] and ledger_counts() == (0, 0, 0)
    assert owner_snapshot(retained) == before


@pytest.mark.parametrize("field,value", [("pkid", True), ("id", "not-a-uuid"),
    ("collection_id", True), ("source_hash", "A" * 64), ("model_label", "apps_documents.rawtextdocument")])
def test_malformed_exact_reference_is_not_a_permission_or_owner_lookup(pdf_adoption_case, field, value):
    case = pdf_adoption_case
    case.ref = replace(case.ref, **{field: value})
    with pytest.raises(REJECTED):
        adopt(case)
    assert case.transport.calls == [] and ledger_counts() == (0, 0, 0)


@pytest.mark.parametrize("kwargs", [{"actor_id": True}, {"actor_id": 0}, {"using": None},
    {"deadline": True}, {"deadline": float("nan")}, {"deadline": float("inf")}, {"deadline": -1.0}])
def test_invalid_or_expired_operation_budget_cannot_start_capture(pdf_adoption_case, kwargs):
    case = pdf_adoption_case
    options = {"actor_id": case.actor.pk, "using": "default", **kwargs}
    with pytest.raises(REJECTED):
        service().adopt_current_pdf(case.ref, **options)
    assert case.transport.calls == [] and ledger_counts() == (0, 0, 0)


def test_manual_transaction_is_rejected_before_source_io(pdf_adoption_case):
    case = pdf_adoption_case
    coordinator = service().adopt_current_pdf
    before = owner_snapshot(case.document)
    source_before = dict(case.transport.blobs)
    previous_autocommit = connection.get_autocommit()
    assert previous_autocommit and not connection.in_atomic_block
    connection.set_autocommit(False)
    try:
        # A manually controlled transaction is invisible to in_atomic_block.
        # The coordinator must not perform capture or return an uncommitted bind.
        assert not connection.in_atomic_block
        with pytest.raises(SourceScopeUnavailable, match="source_scope_unavailable"):
            coordinator(case.ref, actor_id=case.actor.pk, using="default")
    finally:
        try:
            connection.rollback()
        finally:
            connection.set_autocommit(previous_autocommit)
    assert case.transport.calls == [] and case.transport.blobs == source_before
    assert ledger_counts() == (0, 0, 0)
    assert not SourceArtifact.objects.exists()
    assert not SourceArtifactOccurrence.objects.exists()
    assert not ExtractionRevision.objects.exists()
    assert owner_snapshot(case.document) == before


@pytest.mark.parametrize("permission", ["EDIT", "MANAGE", "VIEW"])
def test_current_inherited_edit_permission_is_required(pdf_adoption_case, permission):
    case = pdf_adoption_case
    parent = Collection.objects.create(name="Inherited authorization")
    case.collection.parent = parent
    case.collection.save(update_fields=["parent"])
    case.grant.delete()
    CollectionPermission.objects.create(user=case.actor, collection=parent, permission=permission)
    if permission == "VIEW":
        with pytest.raises(REJECTED):
            adopt(case)
        assert ledger_counts() == (0, 0, 0) and case.transport.calls == []
    else:
        assert adopt(case).document_ref == case.ref


@pytest.mark.parametrize("active", [False, True])
def test_existing_building_or_active_representation_prevents_adoption(pdf_adoption_case, active):
    case = pdf_adoption_case
    existing_representation(case, active=active)
    before = owner_snapshot(case.document)
    with pytest.raises(REJECTED):
        adopt(case)
    assert ledger_counts() == (0, 0, 0) and case.transport.calls == []
    assert owner_snapshot(case.document) == before


@pytest.mark.parametrize("source", ["mismatch", "empty_pdf", "zero_bytes", "malformed", "oversize"])
def test_nonmatching_or_unbounded_bytes_never_bind_owner(pdf_adoption_case, source):
    case = pdf_adoption_case
    before = owner_snapshot(case.document)
    data = {"mismatch": synthetic_pdf("same words but different text"),
            "empty_pdf": synthetic_pdf(""), "zero_bytes": b"", "malformed": b"not a PDF"}.get(source)
    if source == "oversize":
        case.transport.head_override = {"ContentLength": 32 * 1024 * 1024 + 1}
    else:
        case.transport.blobs["private/attached/current.pdf"] = data
    with pytest.raises(REJECTED):
        adopt(case)
    assert ledger_counts() == (0, 0, 0)
    assert not SourceArtifact.objects.exists()
    assert owner_snapshot(case.document) == before
