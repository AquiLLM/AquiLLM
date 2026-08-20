from types import SimpleNamespace
from uuid import UUID

import pytest

from apps.collections.services.retrieval_authorization import (
    OpaquePrincipalReference,
    _make_test_reauthorization_capability,
    freeze_retrieval_authorization_context,
)
from apps.knowledge_graph.projection.identifiers import (
    OpaqueProjectionKey,
    ProjectionIdentifierDomain,
)
from apps.knowledge_graph.projection.records import PrivateProjectionChunkReferenceV1
from apps.knowledge_graph.retrieval.materialization import (
    materialize_projected_chunks,
)

A = UUID("11111111-1111-4111-8111-111111111111")
B = UUID("22222222-2222-4222-8222-222222222222")
K = tuple(character * 64 for character in "123456")


class Policy:
    policy_version, policy_checksum = "policy-v1", K[0]
    rows = ((1, A), (1, B))

    def opaque_principal_reference(self, **_kwargs):
        return OpaquePrincipalReference(K[1])

    def current_authorized_document_scope(self, **_kwargs):
        return self.rows


class Repository:
    def __init__(self, rows, objects, checksum=K[4]):
        self.rows, self.objects, self.checksum, self.calls = (
            rows,
            objects,
            checksum,
            [],
        )

    def load_private_chunk_map(self, **kwargs):
        self.calls.append(("map", kwargs))
        return self.checksum, self.rows

    def load_chunk_objects(self, **kwargs):
        self.calls.append(("objects", kwargs))
        return self.objects


def _authorization(policy):
    principal = object()
    capability = _make_test_reauthorization_capability(
        principal=principal, policy=policy
    )
    return freeze_retrieval_authorization_context(
        principal=principal,
        database_alias="default",
        policy=policy,
        selected_collection_ids=(1,),
        selected_document_ids=(A, B),
        reauthorization_capability=capability,
    )


def _key(value):
    return OpaqueProjectionKey(ProjectionIdentifierDomain.CHUNK, value)


def _chunk(pk: int, document_id: UUID, number: int):
    return SimpleNamespace(pk=pk, doc_id=document_id, chunk_number=number)


def test_materialization_discards_all_graph_chunks_after_partial_revocation() -> None:
    policy = Policy()
    rows = (
        PrivateProjectionChunkReferenceV1(K[2], 10, str(A), 2),
        PrivateProjectionChunkReferenceV1(K[3], 11, str(B), 3),
    )
    objects = (_chunk(10, A, 2), _chunk(11, B, 3))
    repository = Repository(rows, objects)
    authorization = _authorization(policy)
    policy.rows = ((1, A),)  # B revoked after freeze
    result = materialize_projected_chunks(
        projection_id=UUID("33333333-3333-4333-8333-333333333333"),
        expected_private_mapping_checksum=K[4],
        chunk_keys=(_key(K[2]), _key(K[3])),
        authorization=authorization,
        repository=repository,
    )
    assert result == ()
    assert repository.calls == []


@pytest.mark.parametrize("mode", ("stale", "duplicate", "conflict"))
def test_materialization_rejects_stale_duplicate_or_conflicting_maps(mode: str) -> None:
    row = PrivateProjectionChunkReferenceV1(K[2], 10, str(A), 2)
    rows = () if mode == "stale" else (row, row)
    if mode == "conflict":
        rows = (row, PrivateProjectionChunkReferenceV1(K[3], 10, str(B), 3))
    with pytest.raises(ValueError, match=mode):
        materialize_projected_chunks(
            projection_id=UUID("33333333-3333-4333-8333-333333333333"),
            expected_private_mapping_checksum=K[4],
            chunk_keys=(_key(K[2]), _key(K[3]))
            if mode != "duplicate"
            else (_key(K[2]),),
            authorization=_authorization(Policy()),
            repository=Repository(rows, (_chunk(10, A, 2),)),
        )


def test_materialization_binds_checksum_and_exact_chunk_predicates() -> None:
    rows = (
        PrivateProjectionChunkReferenceV1(K[2], 10, str(A), 2),
        PrivateProjectionChunkReferenceV1(K[3], 11, str(B), 3),
    )
    objects = (_chunk(10, A, 2), _chunk(11, B, 3))
    repository = Repository(rows, objects)

    result = materialize_projected_chunks(
        projection_id=UUID("33333333-3333-4333-8333-333333333333"),
        expected_private_mapping_checksum=K[4],
        chunk_keys=(_key(K[2]), _key(K[3])),
        authorization=_authorization(Policy()),
        repository=repository,
    )

    assert tuple(row.candidate_object for row in result) == objects
    assert repository.calls[0][1]["expected_private_mapping_checksum"] == K[4]
    assert repository.calls[1][1]["chunk_predicates"] == (
        (10, A, 2),
        (11, B, 3),
    )
    assert repository.calls[1][1]["authorized_document_ids"] == (A, B)


def test_materialization_rejects_private_mapping_checksum_mismatch() -> None:
    row = PrivateProjectionChunkReferenceV1(K[2], 10, str(A), 2)
    with pytest.raises(ValueError, match="mapping checksum"):
        materialize_projected_chunks(
            projection_id=UUID("33333333-3333-4333-8333-333333333333"),
            expected_private_mapping_checksum=K[4],
            chunk_keys=(_key(K[2]),),
            authorization=_authorization(Policy()),
            repository=Repository((row,), (_chunk(10, A, 2),), checksum=K[5]),
        )


@pytest.mark.parametrize(
    ("mode", "objects"),
    (
        ("missing", ()),
        ("duplicate", (_chunk(10, A, 2), _chunk(10, A, 2))),
        ("pk", (_chunk(12, A, 2),)),
        ("document", (_chunk(10, B, 2),)),
        ("chunk_number", (_chunk(10, A, 9),)),
    ),
)
def test_materialization_rejects_missing_duplicate_or_corrupt_chunks(mode, objects):
    row = PrivateProjectionChunkReferenceV1(K[2], 10, str(A), 2)
    with pytest.raises(ValueError, match=mode):
        materialize_projected_chunks(
            projection_id=UUID("33333333-3333-4333-8333-333333333333"),
            expected_private_mapping_checksum=K[4],
            chunk_keys=(_key(K[2]),),
            authorization=_authorization(Policy()),
            repository=Repository((row,), objects),
        )
