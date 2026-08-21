"""Fixed bounded Cypher families for the shipping Memgraph topology loader."""

from __future__ import annotations

from .memgraph_pagination import cursor_predicate, page_return

_ROOT = (
    "MATCH (membership:AutomaticMembership {generation_key:$generation_key}) "
    "WHERE membership.automatic_membership_key IN split($seed_keys_csv, ',') "
    "OR membership.entity_key IN split($seed_keys_csv, ',') "
    "MATCH (seed:ProjectedEntity {generation_key:$generation_key})"
    "-[:ENTITY_MEMBERSHIP {generation_key:$generation_key}]->(membership) "
)
_NODE_PATH = (
    "MATCH path=(seed)-[:PROJECTED_RELATION*0..2]-"
    "(entity:ProjectedEntity {generation_key:$generation_key}) "
    "WHERE length(path) <= $max_depth "
    "AND all(edge IN relationships(path) "
    "WHERE edge.generation_key = $generation_key) "
)
_EDGE_PATH = (
    "MATCH path=(seed)-[:PROJECTED_RELATION*1..2]-"
    "(target:ProjectedEntity {generation_key:$generation_key}) "
    "WHERE length(path) <= $max_depth "
    "AND all(edge IN relationships(path) "
    "WHERE edge.generation_key = $generation_key) "
    "UNWIND relationships(path) AS physical "
)

def _return(label: str) -> str:
    return (
        "WITH DISTINCT n WHERE "
        + cursor_predicate(label)
        + " "
        + page_return(label)
    )

_QUERIES = {
    "ProjectedEntity": (
        _ROOT
        + _NODE_PATH
        + "MATCH (n:ProjectedEntity {generation_key:$generation_key}) "
        "WHERE n.opaque_key = entity.opaque_key "
        + _return("ProjectedEntity")
    ),
    "AutomaticMembership": (
        _ROOT
        + _NODE_PATH
        + "MATCH (entity)-[:ENTITY_MEMBERSHIP "
        "{generation_key:$generation_key}]->"
        "(n:AutomaticMembership {generation_key:$generation_key}) "
        + _return("AutomaticMembership")
    ),
    "ProjectedDocument": (
        "MATCH (n:ProjectedDocument {generation_key:$generation_key}) "
        "WHERE n.document_key IN split($authorized_document_keys_csv, ',') "
        + _return("ProjectedDocument")
    ),
    "ProjectedChunk": (
        "MATCH (document:ProjectedDocument {generation_key:$generation_key})"
        "-[:DOCUMENT_CHUNK {generation_key:$generation_key}]->"
        "(n:ProjectedChunk {generation_key:$generation_key}) "
        "WHERE document.document_key IN "
        "split($authorized_document_keys_csv, ',') "
        + _return("ProjectedChunk")
    ),
    "ProjectedRelationSemantics": (
        _ROOT
        + _EDGE_PATH
        + "MATCH (n:ProjectedRelationSemantics {generation_key:$generation_key}) "
        "WHERE n.artifact_key = physical.artifact_key "
        "AND n.relation_type = physical.relation_type "
        + _return("ProjectedRelationSemantics")
    ),
    "ProjectedRelation": (
        _ROOT
        + _EDGE_PATH
        + "MATCH (n:ProjectedRelation {generation_key:$generation_key}) "
        "WHERE n.opaque_key = physical.relation_key "
        + _return("ProjectedRelation")
    ),
    "ProjectedEvidence": (
        _ROOT
        + _EDGE_PATH
        + "MATCH (relation:ProjectedRelation {generation_key:$generation_key})"
        "-[evidence:RELATION_EVIDENCE {generation_key:$generation_key}]->"
        "(chunk:ProjectedChunk {generation_key:$generation_key}) "
        "WHERE relation.opaque_key = physical.relation_key "
        "AND chunk.document_key IN split($authorized_document_keys_csv, ',') "
        "MATCH (n:ProjectedEvidence {generation_key:$generation_key}) "
        "WHERE n.opaque_key = evidence.evidence_key "
        + _return("ProjectedEvidence")
    ),
    "ProjectedEntityMention": (
        _ROOT
        + _NODE_PATH
        + "MATCH (entity)-[mention:ENTITY_MENTION "
        "{generation_key:$generation_key}]->"
        "(chunk:ProjectedChunk {generation_key:$generation_key}) "
        "WHERE chunk.document_key IN split($authorized_document_keys_csv, ',') "
        "MATCH (n:ProjectedEntityMention {generation_key:$generation_key}) "
        "WHERE n.opaque_key = mention.mention_key "
        + _return("ProjectedEntityMention")
    ),
    "ArtifactProvenance": (
        "MATCH (n:ArtifactProvenance {generation_key:$generation_key}) "
        "WHERE (n.scope_type = 'collection' AND n.scope_key = $collection_key) "
        "OR (n.scope_type = 'document' AND n.scope_key IN "
        "split($authorized_document_keys_csv, ',')) "
        + _return("ArtifactProvenance")
    ),
}


def bounded_family_query(label: str) -> str:
    try:
        return _QUERIES[label]
    except KeyError:
        raise ValueError("unknown bounded Memgraph projection family") from None


__all__ = ["bounded_family_query"]
