"""Phase-local metadata keeps singleton fanout, aliases and whole-family limits."""

import pytest
from django.db import connection
from django.test.utils import CaptureQueriesContext

from apps.knowledge_graph.models import DocumentEntityMention
from apps.knowledge_graph.projection import django_projection_topology as topology
from apps.knowledge_graph.projection.identifiers import ProjectionIdentifierDomain
from apps.knowledge_graph.retrieval.ready_provenance_v2 import _phase_provenance_reader
from apps.knowledge_graph.tests.ready_provenance_document_support import _set
from apps.knowledge_graph.tests.ready_provenance_document_support import case as case

pytestmark = pytest.mark.django_db


def reader(case):
    return _phase_provenance_reader(
        (case.authority, chunk) for pair in case.chunks for chunk in pair
    )


def read(case, phase, chunk):
    return phase(
        authority=case.authority, chunk=chunk, codec=case.codec, using="default"
    )


def expected(case, index, chunk):
    return {
        case.codec.encode(
            ProjectionIdentifierDomain.EVIDENCE,
            generation=case.authority.generation_id,
            source=f"entity-mention:{case.mentions[index].pk}:chunk:{chunk.pk}",
        ).value: None
    }


def test_each_chunk_keeps_its_fanout_cap_when_union_would_exceed_it(case, monkeypatch):
    # Two raw rows expand to four chunks; each original singleton still has one.
    monkeypatch.setattr(topology, "MAX_DETAIL_ROWS", 2)
    phase = reader(case)
    with CaptureQueriesContext(connection) as statements:
        for index, pair in enumerate(case.chunks):
            for chunk in pair:
                assert read(case, phase, chunk) == expected(case, index, chunk)
    metadata = [q["sql"] for q in statements if '"metadata"' in q["sql"]]
    assert len(metadata) == 1 and "EXISTS(" in metadata[0]
    assert all(
        str(pair[0].doc_id).replace("-", "") in metadata[0].replace("-", "")
        for pair in case.chunks
    )


@pytest.mark.parametrize("empty", [False, True])
def test_global_join_overflow_rejects_each_requested_subset(case, monkeypatch, empty):
    monkeypatch.setattr(topology, "MAX_DETAIL_ROWS", 1)
    if empty:
        _set(DocumentEntityMention, case.assignments[0].pk, status="superseded")
        _set(DocumentEntityMention, case.inactive_assignment.pk, status="active")
    phase = reader(case)
    with pytest.raises(ValueError, match="hard cap"):
        read(case, phase, case.chunks[0][1])


def test_next_phase_reloads_current_rows_and_rejects_global_entity_overflow(
    case, monkeypatch
):
    from apps.knowledge_graph.projection import limits

    assert read(case, reader(case), case.chunks[0][1]) == expected(
        case, 0, case.chunks[0][1]
    )
    _set(DocumentEntityMention, case.assignments[0].pk, status="superseded")
    assert read(case, reader(case), case.chunks[0][1]) == {}
    monkeypatch.setattr(limits, "MAX_ENTITY_ROWS", 1)
    with pytest.raises(ValueError, match="entity cap"):
        read(case, reader(case), case.chunks[1][1])


def test_malformed_one_document_does_not_poison_other_singleton(case, monkeypatch):
    original = topology._load_entity_mention_rows
    bad_document = case.chunks[0][0].doc_id

    def malformed(**kwargs):
        return tuple(
            {**row, "confidence": "bad"} if row["document_id"] == bad_document else row
            for row in original(**kwargs)
        )

    monkeypatch.setattr(topology, "_load_entity_mention_rows", malformed)
    phase = reader(case)
    with pytest.raises(ValueError, match="invalid"):
        read(case, phase, case.chunks[0][1])
    assert read(case, phase, case.chunks[1][1]) == expected(case, 1, case.chunks[1][1])


def test_changed_chunk_coordinates_cannot_use_prepared_metadata(case):
    chunk = case.chunks[0][1]
    phase = reader(case)
    chunk.chunk_number += 1
    with pytest.raises(ValueError, match="identity"):
        read(case, phase, chunk)


def test_distinct_generation_reconstructs_its_own_keys(case):
    from types import SimpleNamespace
    from uuid import uuid4

    chunk = case.chunks[0][1]
    other = SimpleNamespace(**{**vars(case.authority), "generation_id": uuid4()})
    phase = _phase_provenance_reader(((case.authority, chunk), (other, chunk)))
    with CaptureQueriesContext(connection) as statements:
        first = read(case, phase, chunk)
        second = phase(authority=other, chunk=chunk, codec=case.codec, using="default")
    assert first == expected(case, 0, chunk)
    assert second == {
        case.codec.encode(
            ProjectionIdentifierDomain.EVIDENCE,
            generation=other.generation_id,
            source=f"entity-mention:{case.mentions[0].pk}:chunk:{chunk.pk}",
        ).value: None
    }
    assert not set(first) & set(second)
    assert len([q for q in statements if '"metadata"' in q["sql"]]) == 2


@pytest.mark.django_db(transaction=True, databases=["default", "projection_source"])
def test_database_aliases_cannot_share_previously_read_rows(case):
    from django.db import connections

    chunk, phase = case.chunks[0][1], reader(case)
    assert read(case, phase, chunk) == expected(case, 0, chunk)
    _set(DocumentEntityMention, case.assignments[0].pk, status="superseded")
    with CaptureQueriesContext(connections["projection_source"]) as statements:
        assert (
            phase(
                authority=case.authority,
                chunk=chunk,
                codec=case.codec,
                using="projection_source",
            )
            == {}
        )
    assert len([q for q in statements if '"metadata"' in q["sql"]]) == 1
