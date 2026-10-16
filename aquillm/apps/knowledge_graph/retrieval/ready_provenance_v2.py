"""Reconstruct captured evidence keys from current, bounded relational authority."""

import json
from hashlib import sha256

from apps.knowledge_graph.projection.identifiers import ProjectionIdentifierDomain


def provenance_for_chunk(*, authority, chunk, codec, using):
    return _provenance_for_chunk(
        authority=authority, chunk=chunk, codec=codec, using=using,
        mentions=_entity_mentions,
    )


_NATIVE_PROVENANCE_FOR_CHUNK = provenance_for_chunk


def _provenance_for_chunk(*, authority, chunk, codec, using, mentions):
    from apps.knowledge_graph.models import CollectionRelationEvidence
    from apps.knowledge_graph.projection.django_projection_evidence import (
        load_projection_evidence,
    )
    from apps.knowledge_graph.projection.limits import MAX_DETAIL_ROWS

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
    for mention in mentions(authority, chunk, using):
        result[key(f"entity-mention:{mention['mention_id']}:chunk:{chunk.pk}")] = None
    return result


def _entity_ids(authority, using):
    from apps.knowledge_graph.models import CollectionEntity
    from apps.knowledge_graph.projection.limits import MAX_ENTITY_ROWS

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
    return entity_ids


def _entity_mentions(authority, chunk, using):
    from apps.knowledge_graph.projection.django_projection_topology import (
        load_entity_mentions,
    )

    return load_entity_mentions(
        using=using,
        artifact_id=authority.artifact_id,
        entity_ids=_entity_ids(authority, using),
        chunks={chunk.pk: (chunk.doc_id, chunk.chunk_number)},
        batch_size=1000,
        document_filtered=True,
    )


def _supports_phase_provenance(reader):
    from apps.knowledge_graph.projection import django_projection_topology as topology

    return (reader is _NATIVE_PROVENANCE_FOR_CHUNK
            and topology.load_entity_mentions is topology._NATIVE_LOAD_ENTITY_MENTIONS)


def _phase_provenance_reader(pairs):
    """One closed phase's validated chunks; no authority or result survives it."""
    from apps.knowledge_graph.projection import django_projection_topology as topology

    captured, rows = {}, {}
    for authority, chunk in pairs:
        captured.setdefault(id(authority), (authority, {}))[1][chunk.pk] = (
            chunk.doc_id, chunk.chunk_number,
        )

    def mentions(authority, chunk, using):
        owner, chunks = captured[id(authority)]
        if (owner is not authority
                or chunks.get(chunk.pk) != (chunk.doc_id, chunk.chunk_number)):
            raise ValueError("phase provenance source identity changed")
        identity = id(authority), using
        if identity not in rows:
            raw = topology._load_entity_mention_rows(
                using=using, artifact_id=authority.artifact_id,
                entity_ids=_entity_ids(authority, using), chunks=chunks,
                batch_size=1000, document_filtered=True,
            )
            by_document = {}
            for row in raw:
                by_document.setdefault(row["document_id"], []).append(row)
            rows[identity] = by_document
        # Retain singleton fanout limits, rejection and confidence/mention ordering.
        return topology.expand_entity_mentions(
            rows[identity].get(chunk.doc_id, ()),
            {chunk.pk: (chunk.doc_id, chunk.chunk_number)},
        )

    def read(**kwargs):
        return _provenance_for_chunk(**kwargs, mentions=mentions)

    return read
