# Read-only validation of persisted document resolution outcomes.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apps.knowledge_graph.services.builds import CommitMarkerState


def _document_resolution_commit_state(
    artifact: object,
    run: object,
) -> CommitMarkerState:
    from apps.knowledge_graph.models import (
        DocumentEntity,
        DocumentEntityMention,
        EntityMention,
        RelationMention,
    )
    from apps.knowledge_graph.resolution.coreference import MAX_DOCUMENT_MENTIONS
    from apps.knowledge_graph.resolution.persistence import (
        _bounded_rows,
        resolution_commit_is_valid,
        resolution_rows_fingerprint,
        source_mention_fingerprint,
    )
    from apps.knowledge_graph.resolution.source_exclusions import (
        resolvable_mentions,
        source_exclusion_audit,
    )
    from apps.knowledge_graph.services.builds import (
        _HASH_PATTERN,
        CommitMarkerState,
        _commit_marker_state,
    )

    stats = getattr(run, "stats", None)
    marker = stats.get("resolution_commit") if type(stats) is dict else None
    entity_query = DocumentEntity.objects.filter(artifact=artifact)
    link_query = DocumentEntityMention.objects.select_related(
        "document_entity", "mention"
    ).filter(document_entity__artifact=artifact)
    entity_count = entity_query.count()
    membership_count = link_query.count()
    mention_query = EntityMention.objects.filter(artifact=artifact).order_by("pk")
    mention_count = mention_query.count()
    rows_present = bool(entity_count or membership_count)
    if mention_count > MAX_DOCUMENT_MENTIONS:
        return CommitMarkerState.CORRUPT
    if entity_count > MAX_DOCUMENT_MENTIONS:
        return CommitMarkerState.CORRUPT
    if membership_count > MAX_DOCUMENT_MENTIONS:
        return CommitMarkerState.CORRUPT
    if type(stats) is not dict or "resolution_commit" not in stats:
        return _commit_marker_state(
            stats,
            "resolution_commit",
            rows_present=rows_present,
            valid=False,
        )
    if type(marker) is not dict:
        return CommitMarkerState.CORRUPT
    try:
        mentions = _bounded_rows(
            mention_query,
            MAX_DOCUMENT_MENTIONS,
            "document mention",
        )
        entities = _bounded_rows(
            entity_query.order_by("pk"),
            MAX_DOCUMENT_MENTIONS,
            "document resolution entity",
        )
        links = _bounded_rows(
            link_query.order_by("mention_id"),
            MAX_DOCUMENT_MENTIONS,
            "document resolution membership",
        )
        source_fingerprint = source_mention_fingerprint(mentions)
        rows_fingerprint = resolution_rows_fingerprint(entities, links)
        accepted = resolvable_mentions(mentions)
        relations = ()
        if len(accepted) != len(mentions):
            from apps.knowledge_graph.extraction.pipeline import (
                DOCUMENT_EXTRACTION_V1_MAX_RELATIONS,
            )

            relations = _bounded_rows(
                RelationMention.objects.filter(artifact=artifact).order_by("pk"),
                DOCUMENT_EXTRACTION_V1_MAX_RELATIONS,
                "document relation",
            )
        exclusion_audit = source_exclusion_audit(mentions, relations)
    except (TypeError, ValueError):
        return CommitMarkerState.CORRUPT
    result_checksum = marker.get("result_checksum")
    valid_marker = resolution_commit_is_valid(
        marker,
        resolver_version=artifact.resolver_version,
        ontology_checksum=artifact.ontology_checksum,
        assembly_version=artifact.assembly_version,
        assembly_config_checksum=artifact.assembly_config_checksum,
        source_mention_count=len(mentions),
        source_mention_fingerprint=source_fingerprint,
        document_entity_count=entity_count,
        membership_count=membership_count,
        result_checksum=result_checksum,
        exclusion_audit=exclusion_audit,
    )
    entity_ids = {row.pk for row in entities}
    mention_ids = {row.pk for row in accepted}
    row_audits_valid = (
        len(entity_ids) == len(entities)
        and type(stats.get("resolution_rows_fingerprint")) is str
        and _HASH_PATTERN.fullmatch(stats["resolution_rows_fingerprint"]) is not None
        and stats["resolution_rows_fingerprint"] == rows_fingerprint
        and all(
            row.status == row.Status.ACTIVE
            and type(row.metadata) is dict
            and row.metadata.get("result_checksum") == result_checksum
            for row in entities
        )
        and {row.mention_id for row in links} == mention_ids
        and all(
            row.status == row.Status.ACTIVE
            and row.document_entity_id in entity_ids
            and row.mention.artifact_id == artifact.pk
            and row.resolver_version == artifact.resolver_version
            and type(row.metadata) is dict
            and row.metadata.get("result_checksum") == result_checksum
            for row in links
        )
    )
    return _commit_marker_state(
        stats,
        "resolution_commit",
        rows_present=rows_present,
        valid=bool(valid_marker and row_audits_valid),
    )
