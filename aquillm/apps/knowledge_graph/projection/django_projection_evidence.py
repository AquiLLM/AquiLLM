from __future__ import annotations

from uuid import UUID

from django.db.models import F

from apps.knowledge_graph.models import CollectionRelationEvidence

from .limits import bounded_projection_rows

_FIELDS = (
    "id relation_id relation_mention_id relation_mention__chunk_id "
    "relation_mention__document_id relation_mention__chunk__chunk_number "
    "relation_mention__extraction_confidence relation_mention__artifact_id "
    "relation_mention__head_id relation_mention__tail_id head_mapping_id "
    "tail_mapping_id orientation relation__relation_type ontology_checksum "
    "assembly_config_checksum"
).split()
_ALIASES = {
    "relation_mention__chunk_id": "chunk_id",
    "relation_mention__document_id": "document_id",
    "relation_mention__chunk__chunk_number": "chunk_number",
    "relation_mention__extraction_confidence": "confidence",
    "relation_mention__artifact_id": "artifact_id",
    "relation_mention__head_id": "head_mention_id",
    "relation_mention__tail_id": "tail_mention_id",
    "relation__relation_type": "relation_type",
}


def load_projection_evidence(
    *,
    using: str,
    artifact_id: int,
    relation_ids: tuple[int, ...],
    document_ids: tuple[UUID, ...],
    document_artifact_ids: tuple[int, ...],
    batch_size: int,
) -> tuple[dict, ...]:
    query = CollectionRelationEvidence.objects.using(using).filter(
        artifact_id=artifact_id,
        status="active",
        relation_id__in=relation_ids,
        relation__status="active",
        relation__artifact_id=F("artifact_id"),
        relation_mention__artifact_id__in=document_artifact_ids,
        relation_mention__document_id__in=document_ids,
        relation_mention__chunk__doc_id=F("relation_mention__document_id"),
        relation_mention__relation_type=F("relation__relation_type"),
        ontology_checksum=F("artifact__ontology_checksum"),
        assembly_config_checksum=F("artifact__assembly_config_checksum"),
        head_mapping__collection_entity__status="active",
        tail_mapping__collection_entity__status="active",
    )
    rows = bounded_projection_rows(query, tuple(_FIELDS), batch_size)
    return tuple(
        {_ALIASES.get(key, key): value for key, value in row.items()} for row in rows
    )
