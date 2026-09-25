"""Two branch unions retain the low-level materializer's bounded checks."""

from types import SimpleNamespace

import pytest

from apps.documents.tests.hybrid_graph_test_support import Policy, authorization
from apps.knowledge_graph.projection.identifiers import (
    HmacSha256ProjectionIdentifierCodec,
    ProjectionIdentifierDomain,
)
from apps.knowledge_graph.projection.records import PrivateProjectionChunkReferenceV1
from apps.knowledge_graph.retrieval.materialization import materialize_projected_chunks
from apps.knowledge_graph.retrieval.ready_materialization import (
    materialize_selected_ready_chunks,
)
from apps.knowledge_graph.tests.test_production_hybrid_runtime import _ready_scope
from apps.knowledge_graph.tests.test_ready_scope import _DOC_A


class Repository:
    def __init__(self, scope, keys, *, after_first=None, corrupt_second=None):
        self.authority = scope.projections[0]
        self.keys = keys
        self.after_first = after_first
        self.corrupt_second = corrupt_second
        self.calls = []
        self.map_batches = []
        self.object_batches = []
        self.pks = {key.value: index + 1 for index, key in enumerate(keys)}

    def locate(self, *, projection_ids, chunk_keys):
        self.calls.append("locate")
        assert self.authority.projection_id in projection_ids
        return tuple((self.authority.projection_id, key) for key in chunk_keys)

    def load_private_chunk_map(
        self,
        *,
        projection_id,
        chunk_keys,
        expected_private_mapping_checksum,
        database_alias,
    ):
        self.calls.append("map")
        self.map_batches.append(chunk_keys)
        assert len(chunk_keys) <= 20
        assert projection_id == self.authority.projection_id
        assert (
            expected_private_mapping_checksum == self.authority.private_mapping_checksum
        )
        assert database_alias == "default"
        checksum = expected_private_mapping_checksum
        if len(self.map_batches) == 2 and self.corrupt_second == "checksum":
            checksum = "0" * 64
        return checksum, tuple(
            PrivateProjectionChunkReferenceV1(
                key, self.pks[key], str(_DOC_A), self.pks[key]
            )
            for key in chunk_keys
        )

    def load_chunk_objects(
        self, *, chunk_predicates, authorized_document_ids, database_alias
    ):
        self.calls.append("objects")
        self.object_batches.append(chunk_predicates)
        assert _DOC_A in authorized_document_ids
        assert database_alias == "default"
        rows = tuple(
            SimpleNamespace(pk=pk, doc_id=doc, chunk_number=number)
            for pk, doc, number in chunk_predicates
        )
        if len(self.object_batches) == 2 and self.corrupt_second == "coordinates":
            rows[0].chunk_number += 1
        if len(self.object_batches) == 1 and self.after_first:
            self.after_first()
        return rows


def _case(count=40, **kwargs):
    scope = _ready_scope()
    policy = Policy()
    auth = authorization(policy)
    codec = HmacSha256ProjectionIdentifierCodec(b"secret", key_version="key-v1")
    keys = tuple(
        codec.encode(
            ProjectionIdentifierDomain.CHUNK,
            generation=scope.projections[0].generation_id,
            source=index + 1,
        )
        for index in range(count)
    )
    return scope, policy, auth, keys, Repository(scope, keys, **kwargs)


def _materialize(scope, auth, keys, repository):
    return materialize_selected_ready_chunks(
        scope=scope,
        authorization=auth,
        chunk_keys=keys,
        repository=repository,
    )


@pytest.mark.parametrize("count", [20, 21, 40])
def test_complete_union_materializes_in_bounded_batches_preserving_order(count):
    scope, _, auth, keys, repository = _case(count)
    result = _materialize(scope, auth, tuple(reversed(keys)), repository)
    assert tuple(row.chunk_key for row in result) == tuple(
        key.value for key in reversed(keys)
    )
    assert tuple(row.integer_chunk_pk for row in result) == tuple(range(count, 0, -1))
    assert tuple(map(len, repository.map_batches)) == (
        (20,) if count == 20 else (20, count - 20)
    )
    assert len(repository.object_batches) == len(repository.map_batches)


@pytest.mark.parametrize(
    "invalid", ["oversized", "duplicate", "wrong_domain", "list", "empty"]
)
def test_invalid_complete_union_is_rejected_before_repository_reads(invalid):
    scope, _, auth, keys, repository = _case(41 if invalid == "oversized" else 40)
    if invalid == "duplicate":
        keys = keys[:20] + (keys[0],) + keys[21:]
    elif invalid == "wrong_domain":
        codec = HmacSha256ProjectionIdentifierCodec(b"secret", key_version="key-v1")
        keys = (
            keys[:20]
            + (
                codec.encode(
                    ProjectionIdentifierDomain.ENTITY,
                    generation=scope.projections[0].generation_id,
                    source=1,
                ),
            )
            + keys[21:]
        )
    elif invalid == "list":
        keys = list(keys)
    elif invalid == "empty":
        keys = ()
    with pytest.raises((ValueError, TypeError)):
        _materialize(scope, auth, keys, repository)
    assert repository.calls == []


def test_revocation_between_batches_returns_no_partial_result():
    scope, policy, auth, keys, repository = _case()
    repository.after_first = lambda: setattr(policy, "rows", ())
    with pytest.raises(ValueError, match="coverage"):
        _materialize(scope, auth, keys, repository)
    assert len(repository.map_batches) == len(repository.object_batches) == 1


@pytest.mark.parametrize("corruption", ["checksum", "coordinates"])
def test_second_batch_retains_checksum_and_coordinate_checks(corruption):
    scope, _, auth, keys, repository = _case(corrupt_second=corruption)
    with pytest.raises(ValueError, match="checksum|chunk_number"):
        _materialize(scope, auth, keys, repository)
    assert len(repository.map_batches) == 2


def test_low_level_materialization_still_rejects_21_keys_without_reads():
    scope, _, auth, keys, repository = _case(21)
    with pytest.raises(ValueError, match="cap"):
        materialize_projected_chunks(
            projection_id=scope.projections[0].projection_id,
            expected_private_mapping_checksum=scope.projections[
                0
            ].private_mapping_checksum,
            authorization=auth,
            chunk_keys=keys,
            repository=repository,
        )
    assert repository.calls == []
