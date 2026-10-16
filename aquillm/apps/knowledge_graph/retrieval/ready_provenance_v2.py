"""Reconstruct captured evidence keys from current, bounded relational authority."""

import json
from hashlib import sha256

from apps.knowledge_graph.projection.identifiers import ProjectionIdentifierDomain


def provenance_for_chunk(*, authority, chunk, codec, using):
    from apps.knowledge_graph.models import CollectionEntity, CollectionRelationEvidence
    from apps.knowledge_graph.projection.django_projection_evidence import (
        load_projection_evidence,
    )
    from apps.knowledge_graph.projection.django_projection_topology import (
        load_entity_mentions,
    )
    from apps.knowledge_graph.projection.limits import MAX_DETAIL_ROWS, MAX_ENTITY_ROWS

    def key(source):
        return codec.encode(
            ProjectionIdentifierDomain.EVIDENCE,
            generation=authority.generation_id,
            source=source,
        ).value

    relation_ids = tuple(
        CollectionRelationEvidence.objects.using(using)
        .filter(
            artifact_id=authority.artifact_id,
            relation_mention__chunk_id=chunk.pk,
            relation_mention__document_id=chunk.doc_id,
            status="active",
        )
        .order_by("relation_id")
        .values_list("relation_id", flat=True)
        .distinct()[: MAX_DETAIL_ROWS + 1]
    )
    if len(relation_ids) > MAX_DETAIL_ROWS:
        raise ValueError("materialization provenance exceeds existing evidence cap")
    rows = load_projection_evidence(
        using=using,
        artifact_id=authority.artifact_id,
        relation_ids=relation_ids,
        document_ids=(chunk.doc_id,),
        document_artifact_ids=tuple(
            a for d, a in authority.documents if d == chunk.doc_id
        ),
        batch_size=1000,
    )
    result = {}
    for row in rows:
        if row["chunk_id"] != chunk.pk:
            continue
        semantic = sha256(
            json.dumps(
                row, default=str, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
        result[key(f"provenance:{semantic}")] = row["relation_type"]
    entity_ids = tuple(
        CollectionEntity.objects.using(using)
        .filter(
            artifact_id=authority.artifact_id,
            status="active",
        )
        .order_by("pk")
        .values_list("pk", flat=True)[: MAX_ENTITY_ROWS + 1]
    )
    if len(entity_ids) > MAX_ENTITY_ROWS:
        raise ValueError("materialization provenance exceeds existing entity cap")
    mentions = load_entity_mentions(
        using=using,
        artifact_id=authority.artifact_id,
        entity_ids=entity_ids,
        chunks={chunk.pk: (chunk.doc_id, chunk.chunk_number)},
        batch_size=1000,
        document_filtered=True,
    )
    for mention in mentions:
        result[key(f"entity-mention:{mention['mention_id']}:chunk:{chunk.pk}")] = None
    return result
