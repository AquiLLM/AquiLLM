"""Raw/ORM writes cannot invent immutable inputs or extend captured authority."""
from datetime import timedelta
from uuid import uuid4

import pytest
from django.db import DatabaseError, transaction
from django.utils import timezone

from apps.documents.models import DocumentRepresentationGeneration, LegacyDocumentSourceSnapshot, TextChunk
from apps.documents.services.representation_activation import _begin
from apps.documents.services.representation_inputs import (
    configured_chunker_key, content_hash, extraction_input, manifest_hash, representation_chunks,
)
from .representation_support import retained as retained, upload as upload, reference

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.fixture
def building(retained):
    doc, output, _ = retained
    origin, revision, extraction = extraction_input(reference(doc), output.extraction_revision_id, using="default")
    key = configured_chunker_key()
    chunks = representation_chunks(origin, extraction, key)
    lease = _begin(reference(doc), origin, revision, extraction, chunks, key, using="default")
    return doc, DocumentRepresentationGeneration.objects.get(pk=lease.generation_key)


@pytest.mark.parametrize("change", ["attempt", "expiry", "failure", "premature_time"])
def test_building_mutation_cannot_extend_or_relabel_captured_claim(building, change):
    _, generation = building
    values = {"attempt": {"attempt": 2},
        "expiry": {"lease_expires_at": timezone.now() + timedelta(days=1)},
        "failure": {"failure_code": "arbitrary message"},
        "premature_time": {"staged_at": timezone.now()}}[change]
    with pytest.raises(DatabaseError), transaction.atomic():
        DocumentRepresentationGeneration.objects.filter(pk=generation.pk).update(**values)


@pytest.mark.parametrize("change", ["collection", "predecessor", "expired", "occurrence", "extra_manifest"])
def test_generation_insert_requires_actual_predecessor_and_closed_input(building, change):
    doc, original = building
    values = {field.attname: getattr(original, field.attname) for field in original._meta.concrete_fields}
    values["key"] = uuid4()
    values["manifest"] = dict(original.manifest)
    if change == "collection":
        values["collection_id"] = doc.collection_id + 1
    elif change == "predecessor":
        values["predecessor_generation_key"] = uuid4()
    elif change == "expired":
        values["lease_expires_at"] = timezone.now() - timedelta(seconds=1)
    elif change == "occurrence":
        values["manifest"]["occurrence_key"] = "f" * 64
    else:
        values["manifest"]["unregistered_identity"] = "extra"
    values["manifest_sha256"] = manifest_hash(values["manifest"])
    with pytest.raises(DatabaseError), transaction.atomic():
        DocumentRepresentationGeneration.objects.create(**values)


@pytest.mark.parametrize("change", ["canonical", "manifest", "collection"])
def test_snapshot_insert_cannot_invent_capture_for_retained_owner(building, change):
    doc, generation = building
    original = generation.owner.source_snapshots.get()
    values = {field.attname: getattr(original, field.attname) for field in original._meta.concrete_fields}
    values["key"] = uuid4()
    if change == "canonical":
        values.update(canonical_text="invented history", canonical_sha256=content_hash("invented history"),
            captured_source_hash=content_hash("invented history"), source_artifact_id=None,
            source_occurrence_id=None, extraction_revision_id=None, origin_output_id=None,
            source_binding=None, media_binding=None, source_content_type="")
    elif change == "collection":
        values["collection_id_at_capture"] = doc.collection_id + 1
    else:
        values["manifest_sha256"] = "f" * 64
    from apps.documents.services.representation_citation_context import SNAPSHOT_FIELDS
    if change != "manifest":
        values["manifest_sha256"] = manifest_hash({"version": "legacy-canonical-snapshot-v1",
            **{name: values[name] for name in SNAPSHOT_FIELDS}})
    with pytest.raises(DatabaseError), transaction.atomic():
        LegacyDocumentSourceSnapshot.objects.create(**values)


def test_state_collection_requires_current_exact_owner_at_commit(building):
    doc, generation = building
    with pytest.raises(DatabaseError), transaction.atomic():
        type(generation.owner).objects.filter(pk=generation.owner_id).update(collection_id=doc.collection_id + 1)


@pytest.mark.parametrize("change", ["modality", "elements"])
def test_chunk_insert_requires_manifest_modality_and_element_binding(building, change):
    doc, generation = building
    entry = generation.manifest["chunks"][0]
    row = TextChunk(doc_id=doc.id, representation_generation=generation,
        content=entry["source_text"], start_position=entry["start"], end_position=entry["end"],
        chunk_number=entry["number"], modality="image" if change == "modality" else "text",
        metadata={"element_keys": ["invented"] if change == "elements" else entry["element_keys"]},
        legacy_context_status="not_applicable", legacy_artifact_status="not_applicable")
    with pytest.raises(DatabaseError), transaction.atomic():
        TextChunk.objects.bulk_create([row])
