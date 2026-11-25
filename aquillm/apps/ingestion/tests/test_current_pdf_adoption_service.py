"""Present source correspondence binds an existing owner without republishing it."""
from dataclasses import asdict, replace
from datetime import datetime
from hashlib import sha256
import os
import time
from uuid import uuid4

import pytest
from django.contrib.auth.models import User
from django.db import connection

from apps.collections.models import Collection, CollectionPermission
from apps.documents.models import (
    Document, DocumentRepresentationState, PDFDocument, RawTextDocument,
)
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.ingestion.models import (
    ExtractionRevision, IngestionBatch, IngestionBatchItem, IngestionOutput,
    SourceArtifact, SourceArtifactOccurrence,
)
from apps.ingestion.services.output_ledger import OutputResultV1
from apps.ingestion.services.source_artifacts import SourceArtifactIntegrityError
from lib.parsers import pdf_current_validation
from .current_pdf_adoption_support import (
    CANONICAL, existing_representation, ledger_counts, on_first_artifact_write, on_first_source_read,
    owner_snapshot, pdf_adoption_case as pdf_adoption_case, reference, s3 as s3,
    service, synthetic_pdf,
)


pytestmark = pytest.mark.django_db(transaction=True)
REJECTED = (SourceScopeUnavailable, SourceArtifactIntegrityError)


def adopt(case, **kwargs):
    return service().adopt_current_pdf(case.ref, actor_id=case.actor.pk, using="default", **kwargs)


def test_adoption_retains_exact_owner_chunks_and_pointers(pdf_adoption_case, monkeypatch):
    case = pdf_adoption_case
    before = owner_snapshot(case.document)
    policy = pdf_current_validation.freeze_policy()

    def forbidden(*args, **kwargs):
        pytest.fail("adoption attempted document publication, historical capture, or task dispatch")

    # This service must never create a new document or enter any publication path.
    monkeypatch.setattr(Document, "save", forbidden)
    monkeypatch.setattr("apps.ingestion.services.output_ledger.publish_output", forbidden)
    monkeypatch.setattr("apps.ingestion.services.output_materialization.build_document", forbidden)
    monkeypatch.setattr("apps.documents.services.representation_history.capture_legacy", forbidden)
    monkeypatch.setattr("celery.app.task.Task.apply_async", forbidden)
    with connection.cursor() as cursor:
        cursor.execute("SELECT clock_timestamp()")
        started = cursor.fetchone()[0]
    result = adopt(case)
    with connection.cursor() as cursor:
        cursor.execute("SELECT clock_timestamp()")
        ended = cursor.fetchone()[0]

    assert type(result) is OutputResultV1
    assert (result.status, result.document_ref, result.error_code) == ("success", case.ref, None)
    output = IngestionOutput.objects.get(output_key=result.output_key)
    assert (output.provenance_kind, output.output_kind, output.element_key, output.attempt) == (
        "current_adoption", "text", "body", 0)
    assert output.parent_output_id is None
    assert owner_snapshot(case.document) == before
    assert not DocumentRepresentationState.objects.exists()
    assert ledger_counts() == (1, 1, 1)
    item = output.item
    assert (item.status, item.attempts, item.manifest_complete) == ("success", 0, True)
    assert item.document_ids == [str(case.document.id)] and item.finished_at is not None
    assert item.parser_metadata["outputs"][0]["document_id"] == str(case.document.id)
    assert item.parser_metadata["outputs"][0]["issue_codes"] == ["structure_unavailable"]
    assert item.parser_metadata["outputs"][0]["provenance_kind"] == "current_adoption"

    occurrence, revision = output.source_occurrence, output.extraction_revision
    artifact = occurrence.artifact
    assert case.transport.blobs["private/" + artifact.immutable_storage_key] == case.raw
    assert (artifact.sha256, artifact.size_bytes) == (sha256(case.raw).hexdigest(), len(case.raw))
    assert (occurrence.member_path, occurrence.container_id, occurrence.archive_lineage) == (None, None, [])
    assert (revision.parser_key, revision.canonical_text, revision.quality, revision.issue_codes) == (
        "current-validation:legacy-upload-PDF-current-validation-v1", CANONICAL,
        "partial", ["structure_unavailable"])
    assert revision.ordered_elements == [{"element_key": "body", "kind": "paragraph", "start": 0,
        "end": len(CANONICAL), "page": None, "sheet": None, "slide": None,
        "parent_key": None, "cells": [], "bbox": None}]
    evidence = output.provenance_evidence
    assert evidence["policy"] == policy
    assert evidence["document"] == {**asdict(case.ref), "id": str(case.ref.id)}
    assert evidence["attached_reference"] == {"field": "pdf_file", "name": "attached/current.pdf"}
    assert evidence["actor_id"] == case.actor.pk
    assert started <= datetime.fromisoformat(evidence["validated_at"]) <= ended


@pytest.mark.skipif(os.name != "posix", reason="actual worker isolation needs the target POSIX runtime")
def test_posix_adoption_runs_actual_isolated_parser_and_native_persistence(pdf_adoption_case):
    case = pdf_adoption_case
    assert pdf_current_validation.validate is case.native_validate
    result = adopt(case)
    assert result.document_ref == case.ref
    output = IngestionOutput.objects.get(output_key=result.output_key)
    assert output.extraction_revision.canonical_text == CANONICAL
    assert case.transport.blobs["private/" + output.source_occurrence.artifact.immutable_storage_key] == case.raw


def test_exact_repeat_revalidates_and_reuses_one_immutable_output(pdf_adoption_case):
    case = pdf_adoption_case
    first = adopt(case)
    output_before = IngestionOutput.objects.values().get()
    counts = ledger_counts()
    transport_reads = len(case.transport.bodies)
    second = adopt(case)
    assert second == first and ledger_counts() == counts
    assert IngestionOutput.objects.values().get() == output_before
    assert len(case.transport.bodies) > transport_reads


def test_other_fresh_editor_can_reuse_existing_evidence_without_relabeling_actor(pdf_adoption_case):
    case = pdf_adoption_case
    first = adopt(case)
    evidence = IngestionOutput.objects.get().provenance_evidence
    editor = User.objects.create_user(username="second-current-operator")
    CollectionPermission.objects.create(user=editor, collection=case.collection, permission="MANAGE")
    second = service().adopt_current_pdf(case.ref, actor_id=editor.pk, using="default")
    assert second == first and ledger_counts() == (1, 1, 1)
    assert IngestionOutput.objects.get().provenance_evidence == evidence


@pytest.mark.parametrize("change", ["grant", "inactive", "attached_bytes"])
def test_retry_does_not_treat_existing_evidence_as_authorization(pdf_adoption_case, change):
    case = pdf_adoption_case
    first = adopt(case)
    before = IngestionOutput.objects.values().get()
    if change == "grant":
        case.grant.delete()
    elif change == "inactive":
        User.objects.filter(pk=case.actor.pk).update(is_active=False)
    else:
        case.transport.blobs["private/attached/current.pdf"] = synthetic_pdf("changed")
    with pytest.raises(REJECTED):
        adopt(case)
    assert IngestionOutput.objects.values().get() == before
    assert IngestionOutput.objects.get().output_key == first.output_key
    assert ledger_counts() == (1, 1, 1)


@pytest.mark.parametrize("change", ["text", "move", "attachment", "grant", "inactive", "backend", "duplicate_uuid"])
@pytest.mark.parametrize("phase", ["capture", "persistence"])
def test_changed_authority_or_owner_never_partially_binds(pdf_adoption_case, change, phase):
    case = pdf_adoption_case
    other = Collection.objects.create(name="Other collection")
    after_mutation = []

    def mutate():
        assert not connection.in_atomic_block, "source work held row locks across a race"
        if change == "text":
            # Exercise the real edit lifecycle, including its legitimate legacy
            # snapshot. Adoption must preserve those edit-owned changes.
            case.document.full_text = "owner changed"
            case.document.save(dont_rechunk=True, update_fields=["full_text"])
        elif change == "move":
            PDFDocument.objects.filter(pk=case.document.pk).update(collection=other)
        elif change == "attachment":
            PDFDocument.objects.filter(pk=case.document.pk).update(pdf_file="attached/replaced.pdf")
        elif change == "grant":
            case.grant.delete()
        elif change == "inactive":
            User.objects.filter(pk=case.actor.pk).update(is_active=False)
        elif change == "backend":
            case.storage.bucket_name = "changed-bucket"
        else:
            RawTextDocument.objects.bulk_create([RawTextDocument(id=case.document.id, title="ambiguous owner",
                full_text="other", full_text_hash=sha256(b"other").hexdigest(),
                collection=other, ingested_by=case.actor)])
        after_mutation.append(owner_snapshot(case.document))

    (on_first_source_read if phase == "capture" else on_first_artifact_write)(case, mutate)
    with pytest.raises(REJECTED):
        adopt(case)
    assert ledger_counts() == (0, 0, 0)
    assert len(after_mutation) == 1 and owner_snapshot(case.document) == after_mutation[0]


@pytest.mark.parametrize("field,value", [("raw", b"caller PDF"), ("path", "private.pdf"),
    ("storage", object()), ("parser", lambda: True), ("receipt", {"status": "match"}), ("parity", True)])
def test_caller_cannot_inject_source_or_validation_authority(pdf_adoption_case, field, value):
    with pytest.raises(TypeError):
        adopt(pdf_adoption_case, **{field: value})
    assert ledger_counts() == (0, 0, 0)


def test_policy_is_frozen_before_source_capture(pdf_adoption_case, monkeypatch):
    case = pdf_adoption_case
    freeze, validate = pdf_current_validation.freeze_policy, pdf_current_validation.validate
    expected, events = freeze(), []

    def observed_freeze():
        events.append("freeze")
        return freeze()

    def observed_validate(policy, *args):
        assert events == ["freeze", "capture"]
        assert policy == expected
        return validate(policy, *args)

    monkeypatch.setattr(pdf_current_validation, "freeze_policy", observed_freeze)
    monkeypatch.setattr(pdf_current_validation, "validate", observed_validate)
    on_first_source_read(case, lambda: events.append("capture"))
    adopt(case)
    assert IngestionOutput.objects.get().provenance_evidence["policy"] == expected


def test_existing_genuine_ingestion_origin_is_a_conflict(pdf_adoption_case):
    from apps.ingestion.services.source_artifacts import persist_artifact, persist_occurrence
    from apps.ingestion.services.extraction_revisions import persist_extraction_revision
    from apps.ingestion.services.output_status import output_record
    from lib.parsers.source_contracts import ExtractionV1, SourceElementV1

    case = pdf_adoption_case
    deadline = time.monotonic() + 30
    artifact = persist_artifact(case.raw, member_path=None, container_sha256=None,
                                using="default", deadline=deadline)
    occurrence = persist_occurrence(artifact, original_filename="current.pdf", content_type="application/pdf",
                                    archive_lineage=(), using="default", deadline=deadline)
    revision = persist_extraction_revision(occurrence, ExtractionV1("native-synthetic-v1", artifact,
        CANONICAL, (SourceElementV1("body", "paragraph", 0, len(CANONICAL), None, None, None, None),),
        "partial", ("structure_unavailable",)), using="default", deadline=deadline)
    batch = IngestionBatch.objects.create(user=case.actor, collection=case.collection)
    item = IngestionBatchItem.objects.create(batch=batch, status="success", original_filename="current.pdf",
        source_file=artifact.storage_key, content_type="application/pdf", file_size_bytes=len(case.raw))
    payload = {"title": case.document.title, "full_text": CANONICAL, "normalized_type": "pdf",
        "modality": "text", "provider": "", "model": "", "issue_codes": ["structure_unavailable"],
        "quality": "partial", "source_artifact": asdict(artifact), "media_artifact": None,
        "source_content_type": "application/pdf", "source_format": "pdf", "figure_index": 0,
        "extracted_caption": "", "location_metadata": {}, "collection_id": case.collection.pk,
        "collection_parent_id": None}
    IngestionOutput.objects.create(item=item, output_key=uuid4().hex + uuid4().hex,
        source_occurrence_id=occurrence.occurrence_id, extraction_revision_id=revision,
        output_kind="text", element_key="body", status="success", payload=payload,
        document_model=case.ref.model_label, document_pkid=case.ref.pkid, document_id=case.ref.id,
        document_collection_id=case.ref.collection_id, document_source_hash=case.ref.source_hash)
    before = IngestionOutput.objects.values().get()
    assert output_record(IngestionOutput.objects.get(), using="default", validate_scope=False)["provenance_kind"] == "ingestion"
    with pytest.raises(REJECTED):
        adopt(case)
    assert IngestionOutput.objects.values().get() == before and ledger_counts() == (1, 1, 1)


def test_simultaneous_identical_adoptions_leave_one_terminal_binding(pdf_adoption_case):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, local
    from django.db import close_old_connections
    from .test_bounded_source_storage import Body

    case = pdf_adoption_case
    coordinator = service()
    ready = Barrier(2)
    thread_state = local()

    class ConcurrentBody(Body):
        def read(self, size=-1):
            assert not connection.in_atomic_block
            if not getattr(thread_state, "captured", False):
                thread_state.captured = True
                ready.wait(timeout=20)
            return super().read(size)

    case.transport.body_factory = ConcurrentBody

    def run(_):
        close_old_connections()
        try:
            try:
                return coordinator.adopt_current_pdf(case.ref, actor_id=case.actor.pk, using="default")
            except SourceScopeUnavailable:
                return None  # A bounded conflicting retry may fail closed.
        finally:
            connection.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, range(2)))
    successful = [value for value in results if value is not None]
    assert successful and all(value == successful[0] for value in successful)
    assert ledger_counts() == (1, 1, 1)
    assert IngestionOutput.objects.get().document_id == case.document.id
    assert not DocumentRepresentationState.objects.exists()
