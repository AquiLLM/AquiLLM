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
    def __init__(self, rows, objects):
        self.rows, self.objects, self.calls = rows, objects, []

    def load_private_chunk_map(self, **kwargs):
        self.calls.append(("map", kwargs))
        return self.rows

    def load_chunk_objects(self, **kwargs):
        self.calls.append(("objects", kwargs))
        return {
            pk: self.objects[pk]
            for pk in kwargs["integer_chunk_pks"]
            if pk in self.objects
        }


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


def test_materialization_uses_private_map_and_current_authorization() -> None:
    policy = Policy()
    rows = (
        PrivateProjectionChunkReferenceV1(K[2], 10, str(A), 2),
        PrivateProjectionChunkReferenceV1(K[3], 11, str(B), 3),
    )
    objects = {10: object(), 11: object()}
    repository = Repository(rows, objects)
    authorization = _authorization(policy)
    policy.rows = ((1, A),)  # B revoked after freeze
    result = materialize_projected_chunks(
        projection_id=UUID("33333333-3333-4333-8333-333333333333"),
        chunk_keys=(_key(K[2]), _key(K[3])),
        authorization=authorization,
        repository=repository,
    )
    assert tuple(row.integer_chunk_pk for row in result) == (10,)
    assert result[0].candidate_object is objects[10]
    assert repository.calls[0][1]["database_alias"] == "default"


@pytest.mark.parametrize("mode", ("stale", "duplicate", "conflict"))
def test_materialization_rejects_stale_duplicate_or_conflicting_maps(mode: str) -> None:
    row = PrivateProjectionChunkReferenceV1(K[2], 10, str(A), 2)
    rows = () if mode == "stale" else (row, row)
    if mode == "conflict":
        rows = (row, PrivateProjectionChunkReferenceV1(K[3], 10, str(B), 3))
    with pytest.raises(ValueError, match=mode):
        materialize_projected_chunks(
            projection_id=UUID("33333333-3333-4333-8333-333333333333"),
            chunk_keys=(_key(K[2]), _key(K[3]))
            if mode != "duplicate"
            else (_key(K[2]),),
            authorization=_authorization(Policy()),
            repository=Repository(rows, {10: object()}),
        )
