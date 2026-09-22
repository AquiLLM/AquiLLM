from __future__ import annotations

import json
from dataclasses import replace
from uuid import uuid4

import pytest

from apps.knowledge_graph.projection import django_projection_rows as rows
from apps.knowledge_graph.projection import django_projection_topology as topology
from apps.knowledge_graph.projection.django_projection_topology import (
    expand_entity_mentions,
)
from apps.knowledge_graph.projection.identifiers import (
    HmacSha256ProjectionIdentifierCodec,
)
from apps.knowledge_graph.projection.projection_encoding import (
    encode_projection_snapshot,
)
from apps.knowledge_graph.projection.records import (
    CollectionGraphProjectionBundleV1,
    PrivateProjectionChunkReferenceV1,
)
from apps.knowledge_graph.projection.serialization import projection_checksum
from apps.knowledge_graph.projection.state_repository import (
    FunctionProjectionStateRepository,
)
from apps.knowledge_graph.tests.test_django_projection_source import _snapshot


class RowQuery:
    def __init__(self, count):
        self.rows = [{"id": value} for value in range(count)]
        self.fetch_sizes = []

    def order_by(self, *fields):
        return self

    def values(self, *fields):
        return self

    def __getitem__(self, selection):
        self.rows = self.rows[selection]
        return self

    def iterator(self, *, chunk_size):
        self.fetch_sizes.append(chunk_size)
        return iter(self.rows)


def test_complete_projection_family_exceeds_one_driver_page():
    query = RowQuery(10_005)
    loaded = rows._bounded(query, ("id",), 500)
    assert len(loaded) == 10_005
    assert loaded[-1] == {"id": 10_004}
    assert query.fetch_sizes == [500]


@pytest.mark.parametrize("count", [10, 11])
def test_projection_family_aggregate_boundary_is_exact(count):
    query = RowQuery(count)
    if count == 11:
        with pytest.raises(ValueError, match="hard cap"):
            rows._bounded(query, ("id",), 3, maximum=10)
    else:
        assert len(rows._bounded(query, ("id",), 3, maximum=10)) == count
    assert query.fetch_sizes == [3]


def test_expanded_mention_overflow_fails_instead_of_truncating(monkeypatch):
    monkeypatch.setattr(topology, "MAX_DETAIL_ROWS", 2)
    document = uuid4()
    chunks = {value: (document, value) for value in range(3)}
    source = (
        {
            "entity_id": 1,
            "mention_id": 2,
            "chunk_id": 0,
            "document_id": document,
            "confidence": 0.75,
            "metadata": {"observations": [{"chunk_id": value} for value in chunks]},
        },
    )
    with pytest.raises(ValueError, match="hard cap"):
        expand_entity_mentions(source, chunks)


def test_expanded_mentions_preserve_all_chunks_beyond_previous_family_cap():
    document = uuid4()
    chunks = {value: (document, value) for value in range(6001)}
    source = (
        {
            "entity_id": 1,
            "mention_id": 2,
            "chunk_id": 0,
            "document_id": document,
            "confidence": 0.75,
            "metadata": {"observations": [{"chunk_id": value} for value in chunks]},
        },
    )
    result = expand_entity_mentions(source, chunks)
    assert len(result) == 6001
    assert {item["chunk_id"] for item in result} == set(chunks)


def test_private_chunk_writes_use_bounded_pages_without_losing_rows(monkeypatch):
    repository = FunctionProjectionStateRepository(owner="bulk-test")
    calls = []
    monkeypatch.setattr(
        repository, "_one", lambda *args: calls.append(args) or {"stored_count": 1}
    )
    document = str(uuid4())
    values = tuple(
        PrivateProjectionChunkReferenceV1(f"{index:064x}", index + 1, document, index)
        for index in range(6001)
    )
    repository.create(projection_id=uuid4(), rows=values, batch_size=500)
    payloads = [json.loads(args[1][2]) for args in calls]
    assert [len(page) for page in payloads] == [500] * 12 + [1]
    assert [row["integer_chunk_pk"] for page in payloads for row in page] == list(
        range(1, 6002)
    )


def test_private_chunk_fence_accepts_complete_large_mapping(monkeypatch):
    repository = FunctionProjectionStateRepository(owner="bulk-test")
    calls = []
    monkeypatch.setattr(
        repository, "_one", lambda *args: calls.append(args) or {"fenced": True}
    )
    repository.fence(projection_id=uuid4(), checksum="a" * 64, row_count=10_005)
    assert calls[0][1][3] == 10_005


def test_private_chunk_fence_rejects_aggregate_overflow(monkeypatch):
    repository = FunctionProjectionStateRepository(owner="bulk-test")
    monkeypatch.setattr(
        repository, "_one", lambda *args: pytest.fail("overflow reached database")
    )
    with pytest.raises(ValueError, match="row_count"):
        repository.fence(projection_id=uuid4(), checksum="a" * 64, row_count=250_001)


def test_private_chunk_write_stops_immediately_after_lost_lease(monkeypatch):
    repository = FunctionProjectionStateRepository(owner="bulk-test")
    calls = []

    def write(*args):
        calls.append(args)
        return {"stored_count": 2} if len(calls) == 1 else None

    monkeypatch.setattr(repository, "_one", write)
    document = str(uuid4())
    values = tuple(
        PrivateProjectionChunkReferenceV1(f"{index:064x}", index + 1, document, index)
        for index in range(6)
    )
    with pytest.raises(ValueError, match="lease was lost"):
        repository.create(projection_id=uuid4(), rows=values, batch_size=2)
    assert len(calls) == 2


def test_bulk_generation_checksum_binds_mentions_after_first_page():
    snapshot = _snapshot(uuid4(), uuid4())
    document = snapshot["documents"][0]["document_id"]
    snapshot["chunks"] = tuple(
        {"id": 101 + index, "document_id": document, "chunk_number": 2 + index}
        for index in range(10_005)
    )
    snapshot["entity_mentions"] = tuple(
        {
            "entity_id": 11,
            "mention_id": 51 + index,
            "chunk_id": row["id"],
            "document_id": document,
            "chunk_number": row["chunk_number"],
            "confidence": 0.75,
        }
        for index, row in enumerate(snapshot["chunks"])
    )
    encoded = encode_projection_snapshot(
        snapshot=snapshot,
        codec=HmacSha256ProjectionIdentifierCodec(b"secret-a", key_version="key-v1"),
    )
    bundle = CollectionGraphProjectionBundleV1(**encoded)
    assert bundle.counts.chunk_count == 10_005
    assert bundle.counts.entity_mention_count == 10_005
    changed = replace(
        bundle,
        entity_mentions=(
            *bundle.entity_mentions[:-1],
            replace(bundle.entity_mentions[-1], confidence=0.5),
        ),
    )
    assert projection_checksum(bundle) != projection_checksum(changed)
