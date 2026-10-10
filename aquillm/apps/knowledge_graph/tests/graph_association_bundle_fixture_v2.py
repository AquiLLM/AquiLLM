"""Close every synthetic source-page family into a real validated bundle."""

from dataclasses import replace


def bundle_from_pages(template, pages):
    specs = (
        ("nodes", "entities", "entity_count", "entity_key"),
        (
            "memberships",
            "automatic_memberships",
            "automatic_membership_count",
            "entity_key",
        ),
        ("documents", "documents", "document_count", "document_key"),
        ("chunks", "chunks", "chunk_count", "document_key chunk_number chunk_key"),
        ("relations", "relations", "relation_count", "relation_key"),
        (
            "relation_semantics",
            "relation_semantics",
            "relation_semantics_count",
            "artifact_key relation_type semantics_key",
        ),
        ("relation_evidence", "evidence", "evidence_count", "evidence_key"),
        (
            "mention_evidence",
            "entity_mentions",
            "entity_mention_count",
            "entity_key provenance_key mention_key",
        ),
        (
            "provenance",
            "artifact_provenance",
            "artifact_provenance_count",
            "scope_type scope_key artifact_key",
        ),
    )
    families, counts = {}, {}
    for page_name, bundle_name, count, keys in specs:
        fields = keys.split()
        rows = {
            tuple(getattr(row, k) for k in fields): row
            for page, _, _ in pages
            for row in getattr(page, page_name)
        }
        families[bundle_name] = tuple(rows[k] for k in sorted(rows))
        counts[count] = len(rows)
    return replace(template, **families, counts=replace(template.counts, **counts))
