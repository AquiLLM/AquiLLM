"""Two-document persisted authority for provenance read integration tests."""

from types import SimpleNamespace
from uuid import uuid4

import pytest
from django.contrib.auth.models import User
from django.db import connection

from apps.collections.models import Collection
from apps.documents.models import RawTextDocument, TextChunk
from apps.knowledge_graph.models import (
    CollectionArtifactInput,
    CollectionEntity,
    CollectionEntityDocumentLink,
    DocumentEntity,
    DocumentEntityMention,
    EntityMention,
)
from apps.knowledge_graph.projection.identifiers import (
    HmacSha256ProjectionIdentifierCodec,
)
from apps.knowledge_graph.tests.test_models import _artifact


def _set(model, pk, **fields):
    table = connection.ops.quote_name(model._meta.db_table)
    assignments = ", ".join(
        f"{connection.ops.quote_name(name)} = %s" for name in fields
    )
    with connection.cursor() as cursor:
        cursor.execute(
            f"UPDATE {table} SET {assignments} WHERE id = %s", [*fields.values(), pk]
        )


@pytest.fixture
def case():
    collection = Collection.objects.create(name="provenance documents")
    actor = User.objects.create_user(username="provenance-documents")
    artifact = _artifact(
        scope_type="collection", scope_id=collection.pk, status="building"
    )
    artifact.save()
    entities, mentions, chunks, links, assignments, manifests, locals_ = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for index in range(2):
        document = RawTextDocument(
            collection=collection,
            ingested_by=actor,
            title=f"document {index}",
            full_text=f"Atlas Atlas {index}",
            full_text_hash=RawTextDocument.hash_fn(f"Atlas Atlas {index}"),
        )
        document.save(dont_rechunk=True)
        source = _artifact(scope_id=document.id, status="building")
        source.save()
        pair = tuple(
            TextChunk.objects.create(
                doc_id=document.id,
                content="Atlas",
                chunk_number=number,
                start_position=number * 6,
                end_position=number * 6 + 5,
                embedding=[0.0] * 1024,
            )
            for number in range(2)
        )
        mention = EntityMention.objects.create(
            artifact=source,
            document_id=document.id,
            chunk=pair[0],
            start=0,
            end=5,
            raw_text="Atlas",
            normalized_text="atlas",
            entity_type="model",
            extraction_confidence=0.9,
            position_basis="document_global",
            metadata={"observations": [{"chunk_id": pair[1].pk}]},
        )
        local = DocumentEntity.objects.create(
            artifact=source,
            document_id=document.id,
            cluster_key=f"{index + 1:064x}",
            label="Atlas",
            normalized_label="atlas",
            entity_type="model",
        )
        assignment = DocumentEntityMention.objects.create(
            document_entity=local,
            mention=mention,
            method="root",
            resolver_version=source.resolver_version,
        )
        if index == 1:
            inactive_mention = EntityMention.objects.create(
                artifact=source,
                document_id=document.id,
                chunk=pair[0],
                start=6,
                end=11,
                raw_text="Atlas",
                normalized_text="atlas",
                entity_type="model",
                extraction_confidence=0.9,
                position_basis="document_global",
            )
            inactive_assignment = DocumentEntityMention.objects.create(
                document_entity=local,
                mention=inactive_mention,
                method="root",
                resolver_version=source.resolver_version,
                status="superseded",
            )
        source.status = "active"
        source.save(update_fields=["status"])
        manifest = CollectionArtifactInput.objects.create(
            artifact=artifact,
            collection=collection,
            document_id=document.id,
            document_artifact=source,
            source_signature="0" * 64,
            build_signature="0" * 64,
        )
        entity = CollectionEntity.objects.create(
            artifact=artifact,
            collection=collection,
            cluster_key=f"{index + 1:064x}",
            label="Atlas",
            normalized_label="atlas",
            entity_type="model",
            extraction_confidence=0.9,
            resolution_confidence=0.9,
            retrieval_utility=0.9,
            promotion_confidence=0.9,
        )
        link = CollectionEntityDocumentLink.objects.create(
            artifact=artifact,
            manifest_input=manifest,
            document_entity=local,
            collection_entity=entity,
            score=0.9,
            method="exact",
            resolver_version=artifact.resolver_version,
            outcome="automatic",
            decision_checksum="6" * 64,
            reason="exact",
        )
        entities.append(entity)
        mentions.append(mention)
        chunks.append(pair)
        links.append(link)
        assignments.append(assignment)
        manifests.append(manifest)
        locals_.append(local)
    artifact.status = "active"
    artifact.save(update_fields=["status"])
    return SimpleNamespace(
        artifact=artifact,
        entities=entities,
        mentions=mentions,
        chunks=chunks,
        links=links,
        assignments=assignments,
        manifests=manifests,
        locals=locals_,
        inactive_assignment=inactive_assignment,
        authority=SimpleNamespace(
            artifact_id=artifact.pk, generation_id=uuid4(), documents=()
        ),
        codec=HmacSha256ProjectionIdentifierCodec(b"secret", key_version="key-v1"),
    )
