"""Valid raw relation evidence outside the original selected documents."""

from dataclasses import replace

from apps.knowledge_graph.tests.frontier_fixture_support import bridge_bundles, key


def inaccessible_bridge_bundles():
    first, second = bridge_bundles()
    document, chunk, artifact = "f" * 64, "e" * 64, key("unselected-artifact")
    source = next(r for r in first.artifact_provenance if r.scope_type == "document")
    return replace(
        first,
        documents=(
            *first.documents,
            replace(first.documents[0], document_key=document),
        ),
        chunks=(
            *first.chunks,
            replace(first.chunks[0], chunk_key=chunk, document_key=document),
        ),
        artifact_provenance=(
            *first.artifact_provenance,
            replace(source, scope_key=document, artifact_key=artifact),
        ),
        evidence=tuple(
            replace(
                r,
                document_key=document,
                source_document_key=document,
                chunk_key=chunk,
                artifact_key=artifact,
            )
            for r in first.evidence
        ),
        counts=replace(
            first.counts, document_count=2, chunk_count=2, artifact_provenance_count=3
        ),
    ), second
