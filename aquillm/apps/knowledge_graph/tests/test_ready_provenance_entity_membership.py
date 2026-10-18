"""Bound exact entity membership without broadening retrieval provenance."""

import json
import os
import re

import pytest
from django.db import connection
from django.db.models import Q

from apps.knowledge_graph.models import (
    CollectionEntityDocumentLink,
    DocumentEntityMention,
    EntityMention,
)
from apps.knowledge_graph.projection import django_projection_topology as topology
from apps.knowledge_graph.tests.ready_provenance_document_support import _set
from apps.knowledge_graph.tests.ready_provenance_document_support import case as case

pytestmark = [
    pytest.mark.django_db,
    pytest.mark.skipif(
        os.environ.get("KG_REQUIRE_POSTGRES_TESTS") != "1",
        reason="explicit PostgreSQL integration opt-in required",
    ),
]


def test_large_exact_entity_subset_keeps_metadata_query_parameters_bounded(case):
    ids = tuple(e.pk for e in case.entities) + tuple(
        range(9223372036854700000, 9223372036854723507)
    )
    assert len(ids) == 23509
    shapes = []

    def record(execute, sql, params, many, context):
        if sql.startswith("SELECT") and '"metadata"' in sql:
            shapes.append((len(sql.encode()), len(params)))
        return execute(sql, params, many, context)

    chunk = case.chunks[0][1]
    with connection.execute_wrapper(record):
        result = topology.load_entity_mentions(
            using="default",
            artifact_id=case.artifact.pk,
            entity_ids=ids,
            chunks={chunk.pk: (chunk.doc_id, chunk.chunk_number)},
            batch_size=1000,
            document_filtered=True,
        )
    assert len(result) == 1
    assert result[0]["mention_id"] == case.mentions[0].pk
    assert len(shapes) == 1
    assert shapes[0][1] < 30
    assert shapes[0][0] < 10000


@pytest.mark.parametrize("kind", ["all", "one", "duplicate", "missing", "empty"])
def test_exact_entity_membership_preserves_arbitrary_requested_subsets(case, kind):
    first, second = (entity.pk for entity in case.entities)
    ids, expected = {
        "all": ((first, second), {first, second}),
        "one": ((first,), {first}),
        "duplicate": ((first, first), {first}),
        "missing": ((first, 9223372036854775807), {first}),
        "empty": ((), set()),
    }[kind]
    args = dict(
        using="default",
        artifact_id=case.artifact.pk,
        entity_ids=ids,
        chunks={c.pk: (c.doc_id, c.chunk_number) for pair in case.chunks for c in pair},
        batch_size=1,
    )
    result = topology.load_entity_mentions(**args, document_filtered=True)
    assert result == topology.load_entity_mentions(**args)
    assert {row["entity_id"] for row in result} == expected
    assert len(result) == 2 * len(expected)


def test_exact_membership_relabels_overflow_alias_and_preserves_bound_ids(case):
    field = CollectionEntityDocumentLink._meta.get_field("collection_entity")
    original_lookups = dict(field.get_lookups())
    ids = (case.entities[0].pk, case.entities[0].pk, 9223372036854775807)
    statements = []

    def record(execute, sql, params, many, context):
        if sql.startswith("SELECT") and '"metadata"' in sql:
            statements.append((sql, params))
        return execute(sql, params, many, context)

    chunk = case.chunks[0][1]
    with connection.execute_wrapper(record):
        result = topology.load_entity_mentions(
            using="default",
            artifact_id=case.artifact.pk,
            entity_ids=ids,
            chunks={chunk.pk: (chunk.doc_id, chunk.chunk_number)},
            batch_size=1,
            document_filtered=True,
        )
    assert len(result) == 1 and result[0]["mention_id"] == case.mentions[0].pk
    assert len(statements) == 1
    sql, params = statements[0]
    aliases = re.findall(
        r'"?([A-Za-z0-9_]+)"?\."collection_entity_id" = ANY\(%s::bigint\[\]\)',
        sql,
    )
    assert len(aliases) == 2 and len(set(aliases)) == 2
    assert [value for value in params if type(value) is list] == [list(ids), list(ids)]
    assert field.get_lookups() == original_lookups


@pytest.mark.parametrize("ids", [(), (True,), (-1,), (0,), (2**63,), ("1",), [1]])
def test_unsupported_entity_ids_keep_legacy_lookup_input(ids):
    condition = topology._entity_membership("default", ids, True)
    assert type(condition) is Q
    assert condition.children == [("collection_entity_id__in", ids)]
    assert condition.children[0][1] is ids


def test_custom_and_oversized_entity_id_inputs_keep_legacy_lookup():
    class CustomInt(int):
        pass

    class CustomTuple(tuple):
        pass

    for ids in ((CustomInt(1),), CustomTuple((1,)), (1,) * 50001):
        condition = topology._entity_membership("default", ids, True)
        assert type(condition) is Q and condition.children[0][1] is ids


def test_nonpostgres_and_unfiltered_membership_keep_legacy_lookup(monkeypatch):
    ids = (1, 2)
    for document_filtered in (False, 1):
        condition = topology._entity_membership("default", ids, document_filtered)
        assert type(condition) is Q and condition.children[0][1] is ids
    monkeypatch.setattr(connection, "vendor", "sqlite")
    condition = topology._entity_membership("default", ids, True)
    assert type(condition) is Q and condition.children[0][1] is ids


def test_raw_alias_confidence_tie_and_expansion_cap_are_unchanged(case, monkeypatch):
    _set(
        DocumentEntityMention,
        case.assignments[1].pk,
        document_entity_id=case.locals[0].pk,
    )
    _set(
        EntityMention,
        case.mentions[1].pk,
        artifact_id=case.locals[0].artifact_id,
        document_id=case.chunks[0][0].doc_id,
        chunk_id=case.chunks[0][0].pk,
        extraction_confidence=0.95,
        metadata=json.dumps({"observations": [{"chunk_id": case.chunks[0][1].pk}]}),
    )
    args = dict(
        using="default",
        artifact_id=case.artifact.pk,
        entity_ids=(case.entities[0].pk,),
        chunks={c.pk: (c.doc_id, c.chunk_number) for c in case.chunks[0]},
        batch_size=1,
        document_filtered=True,
    )
    result = topology.load_entity_mentions(**args)
    assert len(result) == 2
    assert all(row["mention_id"] == case.mentions[1].pk for row in result)
    # One joined row may still exceed the per-chunk expansion cap through aliases.
    _set(DocumentEntityMention, case.assignments[0].pk, status="superseded")
    monkeypatch.setattr(topology, "MAX_DETAIL_ROWS", 1)
    with pytest.raises(ValueError, match="hard cap"):
        topology.load_entity_mentions(**args)
