"""Three persisted mentions; only the lower-confidence third answers the query."""

from hashlib import sha256

from apps.documents.models import TextChunk
from apps.documents.services.representation_contracts import DocumentRefV1
from apps.documents.services.representation_store import freeze_active_representations
from apps.knowledge_graph.models import (
    DocumentEntity,
    DocumentEntityMention,
    EntityMention,
    ProjectionChunkReference,
)
from apps.knowledge_graph.projection.identifiers import ProjectionIdentifierDomain

TEXTS = ("Atlas is a model", "Atlas is a model", "Atlas uses the Cedar dataset")


def prepare_dataset_document(fixture):
    document = fixture.documents["A"]
    document.full_text = "\n".join(TEXTS)
    document.full_text_hash = sha256(document.full_text.encode()).hexdigest()
    document.save(dont_rechunk=True)
    fixture.snapshot = freeze_active_representations(
        tuple(
            DocumentRefV1(
                d._meta.label_lower, d.pkid, d.id, d.collection_id, d.full_text_hash
            )
            for d in fixture.documents.values()
        ),
        using="default",
    )


def add_dataset_mentions(fixture, projection):
    chunk = fixture.chunks["A"]
    chunk.content, chunk.end_position = TEXTS[0], len(TEXTS[0])
    chunk.save(update_fields=["content", "end_position"])
    original = EntityMention.objects.get(chunk=chunk)
    entity = DocumentEntity.objects.get(artifact=original.artifact)
    start = len(TEXTS[0]) + 1
    for number, text, confidence in ((1, TEXTS[1], 0.98), (2, TEXTS[2], 0.80)):
        row = TextChunk.objects.create(
            doc_id=chunk.doc_id,
            content=text,
            chunk_number=number,
            start_position=start,
            end_position=start + len(text),
            embedding=[0.0] * 1024,
        )
        mention = EntityMention.objects.create(
            artifact=original.artifact,
            document_id=chunk.doc_id,
            chunk=row,
            start=start,
            end=start + 1,
            raw_text="A",
            normalized_text="a",
            entity_type="model",
            extraction_confidence=confidence,
            position_basis=EntityMention.PositionBasis.DOCUMENT_GLOBAL,
        )
        DocumentEntityMention.objects.create(
            document_entity=entity,
            mention=mention,
            method="root",
            resolver_version=original.artifact.resolver_version,
        )
        key = fixture.codec.encode(
            ProjectionIdentifierDomain.CHUNK,
            generation=projection.generation_key,
            source=row.pk,
        ).value
        ProjectionChunkReference.objects.create(
            projection=projection,
            projection_chunk_key=key,
            chunk=row,
            integer_chunk_pk=row.pk,
            document_uuid=row.doc_id,
            chunk_number=number,
        )
        start += len(text) + 1
        if number == 2:
            fixture.answer = row
