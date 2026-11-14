"""Present validation cannot become historical ingestion or parser work."""
from copy import deepcopy
from uuid import uuid4

import pytest
from django.core.exceptions import ValidationError
from django.db import DatabaseError, connection, transaction

from apps.documents.models import PDFDocument
from apps.ingestion.models import IngestionOutput
from apps.ingestion.services.output_ledger import (
    OutputLeaseV1, claim_output, publish_output, reset_output,
)
from .current_adoption_support import adoption_candidate as adoption_candidate, upload as upload
from .current_adoption_support import insert_candidate

pytestmark = pytest.mark.django_db


@pytest.mark.parametrize("bulk", [False, True])
@pytest.mark.parametrize("change", ["pending", "non_text", "element", "parent", "owner", "attachment",
    "canonical", "revision", "artifact", "actor", "extra_evidence", "boolean_evidence", "unknown_kind",
    "extra_payload", "payload_text", "policy", "unknown_policy", "validation_time", "nonterminal_item"])
def test_invalid_adoption_is_rejected_before_registration(adoption_candidate, bulk, change):
    document, _, values, evidence = adoption_candidate
    values, evidence = dict(values), deepcopy(evidence)
    kind = "current_adoption"
    if change == "pending":
        values["status"] = "queued"
    elif change == "non_text":
        values["output_kind"] = "image"
    elif change == "element":
        values["element_key"] = "invented"
    elif change == "parent":
        # The preexisting parent-source guard already rejects nonexistent parents.
        values["parent_output_id"] = 999999
    elif change == "owner":
        values["document_id"] = uuid4()
        evidence["document"]["id"] = str(values["document_id"])
    elif change == "attachment":
        evidence["attached_reference"]["name"] = "different/current.pdf"
    elif change == "canonical":
        values["document_source_hash"] = "f" * 64
        evidence["document"]["source_hash"] = "f" * 64
    elif change == "revision":
        evidence["extraction_revision"]["manifest_sha256"] = "f" * 64
    elif change == "artifact":
        evidence["artifact"]["size_bytes"] += 1
    elif change == "actor":
        evidence["actor_id"] += 1
    elif change == "extra_evidence":
        evidence["parity"] = True
    elif change == "boolean_evidence":
        evidence = True
    elif change == "unknown_kind":
        kind = "historical_guess"
    elif change == "extra_payload":
        values["payload"] = {**values["payload"], "parity": True}
    elif change == "payload_text":
        values["payload"] = {**values["payload"], "full_text": "different canonical"}
    elif change == "policy":
        evidence["policy"]["implementation_sha256"] = "unversioned"
    elif change == "unknown_policy":
        evidence["policy"]["key"] = "unreviewed-policy"
    elif change == "validation_time":
        evidence["validated_at"] = "not-a-validation-timestamp"
    else:
        values["item"].status = "queued"
        values["item"].save(update_fields=["status"])
    with pytest.raises((ValidationError, DatabaseError)), transaction.atomic():
        insert_candidate(values, evidence, bulk=bulk, kind=kind)
    assert not IngestionOutput.objects.filter(document_id=values["document_id"]).exists()


@pytest.mark.parametrize("field,value", [("provenance_kind", "ingestion"), ("provenance_evidence", {})])
@pytest.mark.parametrize("method", ["save", "update", "bulk_update", "raw"])
def test_provenance_identity_is_immutable(adoption_candidate, field, value, method):
    _, _, values, evidence = adoption_candidate
    row = insert_candidate(values, evidence)
    with pytest.raises((ValidationError, DatabaseError)), transaction.atomic():
        if method == "save":
            setattr(row, field, value)
            row.save()
        elif method == "update":
            IngestionOutput.objects.filter(pk=row.pk).update(**{field: value})
        elif method == "bulk_update":
            setattr(row, field, value)
            IngestionOutput.objects.bulk_update([row], [field])
        else:
            import json
            encoded = json.dumps(value) if field == "provenance_evidence" else value
            with connection.cursor() as cursor:
                cursor.execute(f"UPDATE aquillm_ingestionoutput SET {field}=%s WHERE id=%s", [encoded, row.pk])


def test_stale_delivery_and_recovery_cannot_materialize_adoption(adoption_candidate):
    from apps.ingestion.services.output_dispatch import recover_due_outputs
    document, _, values, evidence = adoption_candidate
    row = insert_candidate(values, evidence)
    item = values["item"]
    item.manifest_complete = True
    item.save(update_fields=["manifest_complete"])
    assert claim_output(row.output_key, using="default") is None
    assert not reset_output(row.output_key, generation=row.generation,
                            extraction_revision_id=row.extraction_revision_id, using="default")
    assert recover_due_outputs(limit=1, using="default") == 0
    result = publish_output(OutputLeaseV1(row.pk, row.generation, 99), using="default")
    assert result.document_ref is None or result.document_ref.id == document.id
    assert PDFDocument.objects.count() == 1
    row.refresh_from_db()
    assert row.status == "success" and row.attempt == 0


def test_direct_materialization_cannot_create_owner_from_adoption(adoption_candidate):
    from apps.ingestion.services.output_materialization import build_document
    _, _, values, evidence = adoption_candidate
    row = insert_candidate(values, evidence)
    with pytest.raises(ValueError, match="source_scope_unavailable"):
        build_document(row, using="default")
    assert PDFDocument.objects.count() == 1


@pytest.mark.parametrize("changes", [{"status": "queued"}, {"source_file": "other.pdf"}])
def test_unfinalized_adoption_item_cannot_enter_parser_recovery(adoption_candidate, changes):
    _, _, values, evidence = adoption_candidate
    insert_candidate(values, evidence)
    item = values["item"]
    with pytest.raises(DatabaseError), transaction.atomic():
        type(item).objects.filter(pk=item.pk).update(**changes)


@pytest.mark.parametrize("competing_kind", ["current_adoption", "ingestion"])
@pytest.mark.parametrize("isolation", ["READ COMMITTED", "REPEATABLE READ"])
@pytest.mark.django_db(transaction=True)
def test_competing_successful_origins_are_rejected_across_connections(adoption_candidate, competing_kind, isolation):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from django.db import close_old_connections
    _, _, values, evidence = adoption_candidate
    ready = Barrier(2)

    def insert(kind):
        close_old_connections()
        try:
            with transaction.atomic(), connection.cursor() as cursor:
                cursor.execute(f"SET TRANSACTION ISOLATION LEVEL {isolation}")
                # Both transactions establish their snapshots before either insert.
                assert IngestionOutput.objects.count() == 0
                ready.wait(timeout=20)
                local = {**values, "output_key": uuid4().hex + uuid4().hex}
                insert_candidate(local, evidence if kind == "current_adoption" else {}, bulk=True, kind=kind)
            return "success"
        except DatabaseError:
            return "conflict"
        finally:
            connection.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(insert, ["current_adoption", competing_kind]))
    assert sorted(results) == ["conflict", "success"]
    assert IngestionOutput.objects.count() == 1


@pytest.mark.parametrize("recorded_at", ["2000-01-01T00:00:00+00:00", "2999-01-01T00:00:00+00:00"])
@pytest.mark.parametrize("bulk", [False, True])
def test_validation_timestamp_is_metadata_and_not_a_clock_based_authorization_token(adoption_candidate, recorded_at, bulk):
    _, _, values, evidence = adoption_candidate
    evidence = {**evidence, "validated_at": recorded_at}
    row = insert_candidate(values, evidence, bulk=bulk)
    row.refresh_from_db()
    assert row.provenance_evidence["validated_at"] == recorded_at


@pytest.mark.parametrize("recorded_at", ["2000-01-01T00:00:00", "2000-02-30T00:00:00+00:00", True, None])
@pytest.mark.parametrize("bulk", [False, True])
def test_validation_timestamp_requires_a_real_timezone_aware_iso_value(adoption_candidate, recorded_at, bulk):
    _, _, values, evidence = adoption_candidate
    with pytest.raises((ValidationError, DatabaseError)), transaction.atomic():
        insert_candidate(values, {**evidence, "validated_at": recorded_at}, bulk=bulk)


def test_later_same_occurrence_revision_can_differ_without_relabeling_adoption(adoption_candidate):
    from apps.documents.tests.representation_support import reference, revision
    from apps.documents.services.representation_inputs import extraction_input
    document, _, values, evidence = adoption_candidate
    output = insert_candidate(values, evidence)
    later = revision(output, "Native reparse yields a different canonical", "native-synthetic-v2")
    origin, selected, extraction = extraction_input(reference(document), later, using="default")
    assert origin.pk == output.pk and selected.pk == later
    assert extraction.canonical_text == "Native reparse yields a different canonical"
    output.refresh_from_db()
    assert output.extraction_revision_id == values["extraction_revision_id"]
    assert output.provenance_evidence == evidence
    assert output.payload["full_text"] == "Existing canonical table value 10"


@pytest.mark.parametrize("change", ["parser", "quality", "issues", "page", "heading", "multiple_elements"])
def test_first_policy_cannot_claim_native_parser_quality_or_structure(adoption_candidate, change):
    from apps.ingestion.models import ExtractionRevision, SourceArtifactOccurrence
    from apps.ingestion.services.source_artifacts import persist_occurrence
    from apps.ingestion.services.extraction_revisions import persist_extraction_revision
    from lib.parsers.source_contracts import ExtractionV1, SourceElementV1, SourceArtifactRefV1
    document, _, values, evidence = adoption_candidate
    source = SourceArtifactOccurrence.objects.get(pk=values["source_occurrence_id"])
    artifact = SourceArtifactRefV1(**values["payload"]["source_artifact"])
    occurrence = persist_occurrence(artifact, original_filename=source.original_filename,
        content_type=source.content_type, archive_lineage=(), using="default")
    end = len(document.full_text)
    elements = (SourceElementV1("body", "heading" if change == "heading" else "paragraph",
        0, end, 1 if change == "page" else None, None, None, None),)
    if change == "multiple_elements":
        elements = (SourceElementV1("first", "paragraph", 0, 4, None, None, None, None),
                    SourceElementV1("second", "paragraph", 4, end, None, None, None, None))
    parser = "native-pdf-ordered-v2" if change == "parser" else "current-validation:legacy-upload-PDF-current-validation-v1"
    quality = "complete" if change == "quality" else "partial"
    issues = ("layout_unavailable",) if change == "issues" else ("structure_unavailable",)
    revision_id = persist_extraction_revision(occurrence,
        ExtractionV1(parser, artifact, document.full_text, elements, quality, issues), using="default")
    revision = ExtractionRevision.objects.get(pk=revision_id)
    values = {**values, "extraction_revision_id": revision_id,
              "payload": {**values["payload"], "quality": quality, "issue_codes": list(issues)}}
    evidence = {**evidence, "extraction_revision": {"id": revision_id, "manifest_sha256": revision.manifest_sha256}}
    # Isolate revision policy from the existing cross-host timestamp defect.
    with connection.cursor() as cursor:
        cursor.execute("SELECT clock_timestamp()")
        evidence["validated_at"] = cursor.fetchone()[0].isoformat()
    with pytest.raises(DatabaseError, match="invalid_current_adoption_revision"), transaction.atomic():
        insert_candidate(values, evidence, bulk=True)


@pytest.mark.parametrize("first_kind,second_kind", [("ingestion", "ingestion"),
    ("ingestion", "current_adoption"), ("current_adoption", "ingestion"),
    ("current_adoption", "current_adoption")])
def test_successful_exact_owner_has_one_origin_for_every_provenance_kind(adoption_candidate, first_kind, second_kind):
    _, _, values, evidence = adoption_candidate
    insert_candidate(values, evidence if first_kind == "current_adoption" else {}, kind=first_kind)
    values = {**values, "output_key": uuid4().hex + uuid4().hex}
    with pytest.raises(DatabaseError), transaction.atomic():
        insert_candidate(values, evidence if second_kind == "current_adoption" else {}, bulk=True, kind=second_kind)
    assert IngestionOutput.objects.count() == 1


def test_older_ingestion_writer_can_omit_new_provenance_columns(adoption_candidate):
    import json
    from django.utils import timezone
    _, _, values, _ = adoption_candidate
    item = values["item"]
    item.status = "queued"
    item.save(update_fields=["status"])
    now = timezone.now()
    key = uuid4().hex + uuid4().hex
    # These are the columns written by the pre-provenance model. In particular,
    # neither new provenance column appears in this SQL or its parameters.
    with connection.cursor() as cursor:
        cursor.execute("""INSERT INTO aquillm_ingestionoutput (
            item_id, output_key, source_occurrence_id, extraction_revision_id,
            parent_output_id, output_kind, element_key, payload, status,
            generation, attempt, next_attempt_at, error_code, document_model,
            document_pkid, document_id, document_collection_id, document_source_hash,
            created_at, updated_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                  %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""", [
            item.pk, key, values["source_occurrence_id"], values["extraction_revision_id"],
            None, "text", "body", json.dumps(values["payload"]), "queued", uuid4(), 0,
            now, "", "", None, None, None, "", now, now,
        ])
        output_id = cursor.fetchone()[0]
    row = IngestionOutput.objects.get(pk=output_id)
    assert row.provenance_kind == "ingestion" and row.provenance_evidence == {}
    item.manifest_complete = True
    item.save(update_fields=["manifest_complete"])
    lease = claim_output(key, using="default")
    assert lease is not None and lease.output_id == output_id
    row.refresh_from_db()
    assert row.status == "processing"
