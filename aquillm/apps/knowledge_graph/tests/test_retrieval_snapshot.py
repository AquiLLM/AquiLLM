"""Contracts for the permission-first Task 15 database snapshot boundary."""

from __future__ import annotations

import inspect
import os
import socket
from uuid import UUID

import pytest
from django.conf import settings
from django.test import override_settings

from apps.knowledge_graph.retrieval import expansion
from apps.knowledge_graph.retrieval.ppr import (
    RetrievalDirection,
    raw_edge_weight,
)
from apps.knowledge_graph.retrieval.types import (
    GraphExpansionRequest,
    GraphExpansionSeed,
)

_DOC_A = UUID("11111111-1111-4111-8111-111111111111")
_DOC_B = UUID("22222222-2222-4222-8222-222222222222")


def _database_is_reachable() -> bool:
    database = settings.DATABASES["default"]
    try:
        with socket.create_connection(
            (database["HOST"], int(database.get("PORT") or 5432)),
            timeout=0.2,
        ):
            return True
    except OSError:
        return False


database_required = pytest.mark.skipif(
    not _database_is_reachable() and os.environ.get("KG_REQUIRE_POSTGRES_TESTS") != "1",
    reason="configured PostgreSQL database is not reachable",
)


def _request(
    *,
    documents: tuple[UUID, ...] = (_DOC_A, _DOC_B),
    collections: tuple[int, ...] = (1, 2),
) -> GraphExpansionRequest:
    return GraphExpansionRequest(
        seeds=(GraphExpansionSeed(10, 1, 1.0),),
        allowed_doc_ids=documents,
        allowed_collection_ids=collections,
    )


def test_snapshot_seams_exist_but_only_composition_is_public() -> None:
    from apps.knowledge_graph import retrieval

    assert callable(expansion.authorized_retrieval_snapshot)
    assert callable(expansion.load_authorized_graph_snapshot)
    assert callable(expansion.expand_chunk_candidates)
    assert retrieval.expand_chunk_candidates is expansion.expand_chunk_candidates
    assert retrieval.get_graph_expansion_config is expansion.get_graph_expansion_config
    assert set(retrieval.__all__) == {
        "GraphExpansionConfig",
        "GraphExpansionDiagnostics",
        "GraphExpansionRequest",
        "GraphExpansionResult",
        "GraphExpansionSeed",
        "expand_chunk_candidates",
        "get_graph_expansion_config",
    }
    assert not hasattr(retrieval, "AuthorizedGraphSnapshot")
    assert not hasattr(retrieval, "rank_authorized_graph_snapshot")
    assert not hasattr(retrieval, "_EvaluationTraceCapability")


@pytest.mark.parametrize("timeout_ms", (True, 0, 151, 1.5, "150"))
def test_snapshot_context_rejects_invalid_timeout_before_database_use(
    timeout_ms: object,
) -> None:
    with pytest.raises(ValueError, match="timeout"):
        with expansion.authorized_retrieval_snapshot(timeout_ms=timeout_ms):
            raise AssertionError("invalid context entered")


def test_scope_projection_requires_each_uuid_once_and_exact_collection_set() -> None:
    request = _request()
    rows = ((_DOC_B, 2), (_DOC_A, 1))

    assert expansion._validate_scope_membership(request, rows) == (
        (_DOC_A, 1),
        (_DOC_B, 2),
    )

    with pytest.raises(expansion._SnapshotMiss):
        expansion._validate_scope_membership(
            request,
            ((_DOC_A, 1), (_DOC_A, 1), (_DOC_B, 2)),
        )
    with pytest.raises(expansion._SnapshotMiss):
        expansion._validate_scope_membership(
            request,
            ((_DOC_A, 1), (_DOC_B, 1)),
        )
    with pytest.raises(expansion._SnapshotMiss):
        expansion._validate_scope_membership(request, ((_DOC_A, 1),))


def test_query_predicate_batches_never_exceed_five_thousand() -> None:
    values = tuple(range(1, 10_002))

    batches = tuple(expansion._query_batches(values))

    assert tuple(map(len, batches)) == (5_000, 5_000, 1)
    assert tuple(value for batch in batches for value in batch) == values


@override_settings(KG_OVERLAY_ALGORITHM="not-ppr-v1")
def test_unknown_overlay_algorithm_fails_before_snapshot_or_orm() -> None:
    with pytest.raises(ValueError, match="KG_OVERLAY_ALGORITHM"):
        expansion._load_algorithm_config()


@pytest.mark.parametrize(
    "setting",
    (
        {"KG_OVERLAY_MAX_NODES": 201},
        {"KG_OVERLAY_MAX_EDGES": 1_001},
        {"KG_OVERLAY_TIMEOUT_MS": 151},
        {"KG_OVERLAY_MAX_MENTIONS_PER_ENTITY": 3},
    ),
)
def test_overlay_settings_cannot_raise_immutable_ceilings(setting) -> None:
    with override_settings(**setting), pytest.raises(ValueError):
        expansion._load_algorithm_config()


def test_nonrepresentative_observation_maps_the_real_seed_chunk() -> None:
    metadata = {
        "observations": [
            {"chunk_id": 10, "confidence": 0.8},
            {"chunk_id": 11, "confidence": 0.7},
        ]
    }

    assert expansion._matching_observation_chunks(
        representative_chunk_id=10,
        metadata=metadata,
        candidate_chunk_ids=(11,),
    ) == (11,)


def test_relation_directions_are_admitted_only_from_the_current_frontier() -> None:
    source = ("local", "a" * 64)
    target = ("local", "b" * 64)

    assert expansion._directions_from_frontier(
        source,
        target,
        ontology_direction="directed",
        frontier=frozenset((source,)),
    ) == ((source, target, RetrievalDirection.FORWARD),)
    assert expansion._directions_from_frontier(
        source,
        target,
        ontology_direction="directed",
        frontier=frozenset((target,)),
    ) == ((target, source, RetrievalDirection.REVERSE_DIRECTED),)
    assert expansion._directions_from_frontier(
        source,
        target,
        ontology_direction="undirected",
        frontier=frozenset((source,)),
    ) == ((source, target, RetrievalDirection.UNDIRECTED),)
    assert expansion._directions_from_frontier(
        source,
        target,
        ontology_direction="undirected",
        frontier=frozenset((source, target)),
    ) == (
        (source, target, RetrievalDirection.UNDIRECTED),
        (target, source, RetrievalDirection.UNDIRECTED),
    )


def test_physical_weights_are_computed_before_canonical_copy_collapse() -> None:
    source = ("canonical", 1)
    target = ("canonical", 2)
    first_evidence = expansion.AuthorizedChunkEvidence(
        20,
        _DOC_A,
        1,
        1.0,
        "first",
    )
    second_evidence = expansion.AuthorizedChunkEvidence(
        21,
        _DOC_B,
        1,
        0.5,
        "second",
    )
    first_projection = expansion._AuthorizedEvidenceProjection(
        1,
        1,
        1,
        first_evidence,
        (1,),
    )
    second_projection = expansion._AuthorizedEvidenceProjection(
        2,
        2,
        2,
        second_evidence,
        (2,),
    )
    rows = (
        expansion._DirectionalPhysicalProjection(
            source,
            "supports",
            target,
            RetrievalDirection.FORWARD,
            1,
            0.0,
            (first_projection,),
        ),
        expansion._DirectionalPhysicalProjection(
            source,
            "supports",
            target,
            RetrievalDirection.FORWARD,
            1,
            1.0,
            (second_projection,),
        ),
    )

    (group,) = expansion._compose_authorized_relation_groups(rows)

    assert group.raw_weight == max(
        raw_edge_weight(
            direction=RetrievalDirection.FORWARD,
            confidence=1.0,
            support_count=1,
            destination_retrieval_utility=0.0,
        ),
        raw_edge_weight(
            direction=RetrievalDirection.FORWARD,
            confidence=0.5,
            support_count=1,
            destination_retrieval_utility=1.0,
        ),
    )
    assert tuple(item.provenance_key for item in group.evidence) == (
        "first",
        "second",
    )


def test_identity_projection_collapses_only_explicit_canonical_peers() -> None:
    rows = (
        expansion._AuthorizedEntityRow(1, 10, 100, "a" * 64, 0.5),
        expansion._AuthorizedEntityRow(2, 20, 200, "b" * 64, 0.5),
        expansion._AuthorizedEntityRow(3, 30, 300, "c" * 64, 0.5),
    )

    projected = expansion._project_identity_keys(rows, ((1, 99), (2, 99)))

    assert projected == {
        1: ("canonical", 99),
        2: ("canonical", 99),
        3: ("local", "c" * 64),
    }
    with pytest.raises(expansion._SnapshotMiss):
        expansion._project_identity_keys(rows, ((1, 99), (1, 100)))


def test_fallback_global_cap_is_confidence_first_then_stable_coordinates() -> None:
    identity = ("canonical", 9)
    rows = (
        ("canonical", "9", 20, _DOC_A, 1, 0.1, 1),
        ("canonical", "9", 21, _DOC_B, 9, 0.9, 2),
        ("canonical", "9", 22, _DOC_A, 2, 0.8, 3),
    )

    selected = expansion._select_fallback_rows(
        rows,
        maximum_per_identity=2,
    )

    assert tuple(row[2] for row in selected[identity]) == (21, 22)


def test_snapshot_context_is_outer_repeatable_read_read_only_without_locks() -> None:
    source = inspect.getsource(expansion.authorized_retrieval_snapshot)
    module_source = inspect.getsource(expansion)

    assert "transaction.atomic" in source
    assert "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY" in source
    assert "SET LOCAL statement_timeout" in source
    assert "in_atomic_block" in source
    assert "select_for_update" not in module_source


def test_loader_source_pins_permission_first_caps_and_exact_evidence_fences() -> None:
    source = inspect.getsource(expansion)

    for required in (
        "DESCENDED_FROM_DOCUMENT",
        "allowed_doc_ids",
        "allowed_collection_ids",
        "GraphArtifact.Status.ACTIVE",
        "CollectionRelationEvidence",
        "Exists(",
        "OuterRef(",
        "manifest_input__document_id",
        "head_mapping",
        "tail_mapping",
        "orientation",
        "ontology_checksum",
        "assembly_config_checksum",
        "load_ontology_yaml",
        "OntologyVersion.Status.SUPERSEDED",
        "ROW_NUMBER",
        "seed_chunk_ids",
        "document_artifact__status__in",
        "document_artifact.status IN ('active', 'superseded')",
        "document_artifact.evaluation_only = FALSE",
    ):
        assert required in source
    assert "authorized_canonical_lookup" not in source
    assert "select_for_update" not in source


@pytest.mark.django_db(transaction=True)
@database_required
def test_postgres_snapshot_is_outer_read_only_repeatable_read_with_local_timeout():
    from django.db import connection

    assert not connection.in_atomic_block
    with expansion.authorized_retrieval_snapshot(timeout_ms=150):
        with connection.cursor() as cursor:
            cursor.execute("SHOW transaction_isolation")
            assert cursor.fetchone() == ("repeatable read",)
            cursor.execute("SHOW transaction_read_only")
            assert cursor.fetchone() == ("on",)
            cursor.execute("SHOW statement_timeout")
            assert cursor.fetchone()[0] in {"150ms", "00:00:00.15"}


@pytest.mark.django_db(transaction=True)
@database_required
def test_postgres_snapshot_rejects_nested_or_reentrant_transaction_contexts():
    from django.db import transaction

    with transaction.atomic():
        with pytest.raises(RuntimeError, match="outer transaction"):
            with expansion.authorized_retrieval_snapshot(timeout_ms=150):
                raise AssertionError("nested retrieval snapshot entered")

    with expansion.authorized_retrieval_snapshot(timeout_ms=150):
        with pytest.raises(RuntimeError, match="already active"):
            with expansion.authorized_retrieval_snapshot(timeout_ms=150):
                raise AssertionError("reentrant retrieval snapshot entered")


@pytest.mark.django_db(transaction=True)
@database_required
def test_postgres_loader_maps_nonrepresentative_seed_and_loads_authorized_edge():
    from pathlib import Path

    from django.db import models

    from apps.documents.models import TextChunk
    from apps.knowledge_graph.models import (
        CollectionRelationEvidence,
        EntityMention,
        GraphArtifact,
        OntologyVersion,
    )
    from apps.knowledge_graph.services.ontology import load_ontology
    from apps.knowledge_graph.tests.test_models import (
        _persist_collection_relation_fixture,
    )

    fixture = _persist_collection_relation_fixture()
    definition = load_ontology(
        Path(__file__).resolve().parents[1] / "ontologies" / "research-v1.yaml"
    )
    OntologyVersion.objects.create(
        kind=OntologyVersion.Kind.GRAPH,
        version=definition.version,
        checksum=definition.checksum,
        status=OntologyVersion.Status.ACTIVE,
        metadata={"yaml": definition.raw_yaml},
    )
    models.QuerySet.update(
        GraphArtifact.objects.filter(
            pk__in=(
                fixture.collection_artifact.pk,
                fixture.document_artifact.pk,
            )
        ),
        ontology_version=definition.version,
        ontology_checksum=definition.checksum,
    )
    models.QuerySet.update(
        CollectionRelationEvidence.objects.filter(pk=fixture.evidence.pk),
        ontology_checksum=definition.checksum,
    )
    models.QuerySet.update(
        GraphArtifact.objects.filter(pk=fixture.collection_artifact.pk),
        status=GraphArtifact.Status.ACTIVE,
    )
    observation_chunk = TextChunk.objects.create(
        content="Overlapping observation",
        start_position=25,
        end_position=48,
        chunk_number=1,
        doc_id=fixture.relation_mention.document_id,
        embedding=[0.0] * 1024,
    )
    models.QuerySet.update(
        EntityMention.objects.filter(pk=fixture.relation_mention.head_id),
        metadata={"observations": [{"chunk_id": observation_chunk.pk}]},
    )
    request = GraphExpansionRequest(
        seeds=(GraphExpansionSeed(observation_chunk.pk, 1, 1.0),),
        allowed_doc_ids=(fixture.relation_mention.document_id,),
        allowed_collection_ids=(fixture.relation.source.collection_id,),
    )

    with expansion.authorized_retrieval_snapshot(timeout_ms=150) as deadline:
        snapshot = expansion.load_authorized_graph_snapshot(
            request,
            load_max_hops=2,
        )
        result = expansion.rank_authorized_graph_snapshot(
            snapshot,
            request,
            effective_max_hops=2,
            _deadline=deadline,
        )

    assert {row.seed_chunk_id for row in snapshot.seed_identities} == {
        observation_chunk.pk
    }
    assert {
        (group.source_key, group.target_key, group.admission_hop)
        for group in snapshot.relation_groups
    } == {
        (
            ("local", fixture.relation.source.cluster_key),
            ("local", fixture.relation.target.cluster_key),
            1,
        ),
        (
            ("local", fixture.relation.target.cluster_key),
            ("local", fixture.relation.source.cluster_key),
            2,
        ),
    }
    assert result.diagnostics.status == "hit"
    assert fixture.relation_mention.chunk_id in result.chunk_ids
