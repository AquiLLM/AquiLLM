"""Retrieval reads only relevant metadata without weakening global join limits."""

import json
import os

import pytest
from django.db import IntegrityError, connection, transaction
from django.test.utils import CaptureQueriesContext

from apps.knowledge_graph.models import (
    CollectionArtifactInput,
    CollectionEntityDocumentLink,
    DocumentEntity,
    DocumentEntityMention,
    EntityMention,
)
from apps.knowledge_graph.projection import django_projection_topology as topology
from apps.knowledge_graph.projection.identifiers import (
    ProjectionIdentifierDomain,
)
from apps.knowledge_graph.retrieval.ready_provenance_v2 import provenance_for_chunk
from apps.knowledge_graph.tests.ready_provenance_document_support import _set
from apps.knowledge_graph.tests.ready_provenance_document_support import case as case

pytestmark = [
    pytest.mark.django_db,
    pytest.mark.skipif(
        os.environ.get("KG_REQUIRE_POSTGRES_TESTS") != "1",
        reason="explicit PostgreSQL integration opt-in required",
    ),
]


def _provenance(case, chunk=None):
    return provenance_for_chunk(
        authority=case.authority,
        chunk=chunk or case.chunks[0][1],
        codec=case.codec,
        using="default",
    )


def test_retrieval_narrows_metadata_sql_to_document_and_retains_observation_alias(case):
    with CaptureQueriesContext(connection) as queries:
        result = _provenance(case)
    chunk = case.chunks[0][1]
    expected = case.codec.encode(
        ProjectionIdentifierDomain.EVIDENCE,
        generation=case.authority.generation_id,
        source=f"entity-mention:{case.mentions[0].pk}:chunk:{chunk.pk}",
    ).value
    assert result == {expected: None}
    metadata_sql = [q["sql"] for q in queries if '"metadata"' in q["sql"]]
    assert len(metadata_sql) == 1
    where = metadata_sql[0].split(" WHERE ", 1)[1]
    assert '"document_id" IN (' in where
    assert str(chunk.doc_id).replace("-", "") in where.replace("-", "")
    assert "DISTINCT" not in metadata_sql[0]


def test_projection_default_and_retrieval_return_same_aliases_across_documents(case):
    chunks = {c.pk: (c.doc_id, c.chunk_number) for pair in case.chunks for c in pair}
    args = dict(
        using="default",
        artifact_id=case.artifact.pk,
        entity_ids=tuple(e.pk for e in case.entities),
        chunks=chunks,
        batch_size=1,
    )
    assert topology.load_entity_mentions(
        **args, document_filtered=True
    ) == topology.load_entity_mentions(**args)
    args["chunks"] = {case.chunks[0][1].pk: (case.chunks[0][1].doc_id, 1)}
    with CaptureQueriesContext(connection) as queries:
        assert len(topology.load_entity_mentions(**args)) == 1
    assert len(queries) == 1
    assert '"metadata"' in queries[0]["sql"]
    assert '"document_id" IN (' not in queries[0]["sql"]
    # An observation naming another document's chunk never creates provenance.
    with connection.cursor() as cursor:
        cursor.execute(
            f"UPDATE {EntityMention._meta.db_table} SET metadata = %s WHERE id = %s",
            [
                json.dumps({"observations": [{"chunk_id": case.chunks[0][1].pk}]}),
                case.mentions[1].pk,
            ],
        )
    assert len(_provenance(case)) == 1


@pytest.mark.parametrize("cap", [1, 2])
@pytest.mark.parametrize("empty", [False, True])
def test_global_raw_join_cap_applies_even_for_small_document_subset(
    case, monkeypatch, cap, empty
):
    monkeypatch.setattr(topology, "MAX_DETAIL_ROWS", cap)
    args = dict(
        using="default",
        artifact_id=case.artifact.pk,
        entity_ids=tuple(e.pk for e in case.entities),
        chunks={} if empty else {case.chunks[0][1].pk: (case.chunks[0][1].doc_id, 1)},
        batch_size=1,
        document_filtered=True,
    )
    if cap == 1:
        with CaptureQueriesContext(connection) as queries:
            with pytest.raises(ValueError, match="hard cap"):
                topology.load_entity_mentions(**args)
        assert len(queries) == 1
        assert "LIMIT 1 OFFSET 1" in queries[0]["sql"]
        if empty:
            assert '"metadata"' not in queries[0]["sql"]
        else:
            assert "EXISTS(" in queries[0]["sql"]
            assert '"document_id" IN (' in queries[0]["sql"]
        assert "DISTINCT" not in queries[0]["sql"]
    else:
        result = topology.load_entity_mentions(**args)
        assert len(result) == (0 if empty else 1)


@pytest.mark.parametrize(
    "defect",
    [
        "link_resolver",
        "manifest_artifact",
        "entity_status",
        "entity_artifact",
        "entity_document",
        "assignment_status",
        "assignment_resolver",
        "mention_artifact",
        "mention_document",
    ],
)
def test_retrieval_keeps_original_invalid_join_exclusions(case, defect):
    first, second = 0, 1
    updates = {
        "link_resolver": (
            CollectionEntityDocumentLink,
            case.links[first].pk,
            {"resolver_version": "wrong"},
        ),
        "manifest_artifact": (
            CollectionArtifactInput,
            case.manifests[first].pk,
            {"artifact_id": case.locals[second].artifact_id},
        ),
        "entity_status": (
            DocumentEntity,
            case.locals[first].pk,
            {"status": "superseded"},
        ),
        "entity_artifact": (
            DocumentEntity,
            case.locals[first].pk,
            {"artifact_id": case.locals[second].artifact_id},
        ),
        "entity_document": (
            DocumentEntity,
            case.locals[first].pk,
            {"document_id": case.chunks[second][0].doc_id},
        ),
        "assignment_status": (
            DocumentEntityMention,
            case.assignments[first].pk,
            {"status": "superseded"},
        ),
        "assignment_resolver": (
            DocumentEntityMention,
            case.assignments[first].pk,
            {"resolver_version": "wrong"},
        ),
        "mention_artifact": (
            EntityMention,
            case.mentions[first].pk,
            {"artifact_id": case.locals[second].artifact_id},
        ),
        "mention_document": (
            EntityMention,
            case.mentions[first].pk,
            {"document_id": case.chunks[second][0].doc_id},
        ),
    }
    model, pk, values = updates[defect]
    _set(model, pk, **values)
    assert _provenance(case) == {}


def test_repeated_join_rows_count_toward_global_cap_without_distinct(case, monkeypatch):
    # Two mentions join the same parent link. The second representative chunk is
    # unused, but its joined row still counts, before expansion deduplicates.
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
    )
    monkeypatch.setattr(topology, "MAX_DETAIL_ROWS", 1)
    with pytest.raises(ValueError, match="hard cap"):
        topology.load_entity_mentions(
            using="default",
            artifact_id=case.artifact.pk,
            entity_ids=(case.entities[0].pk,),
            chunks={case.chunks[0][1].pk: (case.chunks[0][1].doc_id, 1)},
            batch_size=1,
            document_filtered=True,
        )


def test_each_provenance_boundary_observes_current_mention_assignment(case):
    assert len(_provenance(case)) == 1
    _set(DocumentEntityMention, case.assignments[0].pk, resolver_version="changed")
    assert _provenance(case) == {}


def test_overflow_is_checked_in_the_actual_metadata_snapshot(case, monkeypatch):
    monkeypatch.setattr(topology, "MAX_DETAIL_ROWS", 2)
    changed = []

    def activate_before_metadata(execute, sql, params, many, context):
        if sql.startswith("SELECT") and '"metadata"' in sql and not changed:
            changed.append(True)
            # Model-close protections are deliberately bypassed, as in existing
            # corruption fixtures. The newly active unrelated row changes the
            # global joined family from exactly two to three before metadata SQL.
            _set(DocumentEntityMention, case.inactive_assignment.pk, status="active")
        return execute(sql, params, many, context)

    with connection.execute_wrapper(activate_before_metadata):
        with pytest.raises(ValueError, match="hard cap"):
            _provenance(case)
    assert changed == [True]


def test_global_entity_cap_remains_before_document_metadata_read(case, monkeypatch):
    from apps.knowledge_graph.projection import limits

    monkeypatch.setattr(limits, "MAX_ENTITY_ROWS", 1)
    with pytest.raises(ValueError, match="entity cap"):
        _provenance(case)


@pytest.mark.parametrize(
    "confidence", [float("nan"), float("inf"), -float("inf"), -0.1, 1.1, None]
)
def test_database_rejects_invalid_scalars_even_on_unrequested_document(
    case, confidence
):
    # The retrieval subset can omit these Python scalar checks only because the
    # actual database rejects every nonfinite/out-of-range/NULL confidence.
    with pytest.raises(IntegrityError), transaction.atomic():
        _set(EntityMention, case.mentions[1].pk, extraction_confidence=confidence)
