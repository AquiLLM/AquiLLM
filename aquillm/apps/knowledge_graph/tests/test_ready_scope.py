"""Selected-scope ready projection and materialization assembly contracts."""

from __future__ import annotations

from uuid import UUID

import pytest

from apps.documents.tests.hybrid_graph_test_support import Policy, authorization
from apps.knowledge_graph.projection.identifiers import (
    HmacSha256ProjectionIdentifierCodec,
)
from apps.knowledge_graph.retrieval.ready_scope import (
    ReadyProjectionAuthorityV1,
    assemble_selected_ready_scope,
)

_DOC_A = UUID("11111111-1111-4111-8111-111111111111")
_DOC_B = UUID("22222222-2222-4222-8222-222222222222")


def _authority(collection_id: int, document_id: UUID, marker: str):
    return ReadyProjectionAuthorityV1(
        projection_id=UUID(
            f"{marker * 8}-{marker * 4}-4{marker * 3}-8{marker * 3}-{marker * 12}"
        ),
        generation_id=UUID(
            f"{marker * 8}-{marker * 4}-4{marker * 3}-9{marker * 3}-{marker * 12}"
        ),
        collection_id=collection_id,
        artifact_id=collection_id + 100,
        schema_version="collection-graph-v1",
        projection_version="projection-v1",
        identifier_key_version="key-v1",
        membership_epoch=3,
        membership_checksum=marker * 64,
        graph_checksum=("a" if marker != "a" else "b") * 64,
        private_mapping_checksum=("c" if marker != "c" else "d") * 64,
        resolver_version="resolver-v1",
        resolution_config_checksum="e" * 64,
        ontology_version="research-v1",
        ontology_checksum="f" * 64,
        embedding_model_signature="embed-v1",
        documents=((document_id, collection_id + 200),),
    )


def test_ready_scope_binds_only_the_exact_selected_collections_and_documents():
    context = authorization(Policy())
    rows = (_authority(7, _DOC_A, "1"), _authority(9, _DOC_B, "2"))
    scope = assemble_selected_ready_scope(
        authorization=context,
        authorities=rows,
        codec=HmacSha256ProjectionIdentifierCodec(b"secret", key_version="key-v1"),
    )

    assert tuple(row.collection_id for row in scope.projections) == (7, 9)
    assert (
        scope.ready.authorization_context_signature
        == context.authorization_context_signature
    )
    assert len(scope.ready.selected_generations) == 2
    assert len(scope.ready.authorized_documents) == 2
    assert scope.selected_document_ids == (_DOC_A, _DOC_B)


def test_ready_scope_rejects_missing_extra_or_stale_selected_authority():
    context = authorization(Policy())
    codec = HmacSha256ProjectionIdentifierCodec(b"secret", key_version="key-v1")
    first, second = _authority(7, _DOC_A, "1"), _authority(9, _DOC_B, "2")

    for rows in ((first,), (first, second, _authority(10, _DOC_B, "3"))):
        with pytest.raises(ValueError, match="readiness_mismatch"):
            assemble_selected_ready_scope(
                authorization=context,
                authorities=rows,
                codec=codec,
            )

    with pytest.raises(ValueError, match="readiness_mismatch"):
        assemble_selected_ready_scope(
            authorization=context,
            authorities=(first, _authority(9, _DOC_A, "2")),
            codec=codec,
        )
