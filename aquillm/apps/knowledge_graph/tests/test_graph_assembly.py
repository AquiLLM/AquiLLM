from __future__ import annotations

import ast
import inspect
import os
import socket
import uuid
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest
from django.conf import settings
from django.db.models import CheckConstraint, UniqueConstraint

from apps.knowledge_graph.graph.assembly import (
    ASSEMBLY_V1_MAX_ENTITIES,
    ASSEMBLY_V1_MAX_EVIDENCE,
    ASSEMBLY_V1_MAX_LINKS,
    ASSEMBLY_V1_MAX_ORPHAN_ENTITIES,
    ASSEMBLY_V1_MAX_RELATIONS,
    AssemblyConfig,
    AssemblyEvidenceInput,
    EvidenceDisposition,
    activate_collection_graph,
    assemble_collection_graph,
    assembly_config_checksum,
    plan_collection_relations,
    validate_assembly_projection,
)
from apps.knowledge_graph.services.ontology import load_ontology


def test_evaluation_collection_completion_never_swaps_production_active():
    from apps.knowledge_graph.graph.assembly import _swap_active_collection_artifact
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    production_saves = []
    evaluation_saves = []
    run_saves = []
    production = SimpleNamespace(
        pk=1,
        status=GraphArtifact.Status.ACTIVE,
        evaluation_only=False,
        build_generation=1,
        activated_at=object(),
        superseded_at=None,
        save=lambda **kwargs: production_saves.append(kwargs),
    )
    evaluation = SimpleNamespace(
        pk=2,
        status=GraphArtifact.Status.BUILDING,
        evaluation_only=True,
        rebuild_request_id=object(),
        build_generation=2,
        activated_at=None,
        completed_at=None,
        superseded_at=None,
        save=lambda **kwargs: evaluation_saves.append(kwargs),
    )
    run = SimpleNamespace(
        evaluation_only=True,
        rebuild_request_id=evaluation.rebuild_request_id,
        orchestration_version=GraphArtifact.OrchestrationVersion.SCOPED_V1,
        stage=GraphBuildRun.Stage.VALIDATING,
        status=GraphBuildRun.Status.RUNNING,
        stage_marker={},
        finished_at=None,
        lease_owner="owner",
        lease_expires_at=object(),
        save=lambda **kwargs: run_saves.append(kwargs),
    )

    _swap_active_collection_artifact(
        artifact=evaluation,
        run=run,
        scope_artifacts=(production, evaluation),
    )

    assert production.status == GraphArtifact.Status.ACTIVE
    assert production.superseded_at is None
    assert not production_saves
    assert evaluation.status == GraphArtifact.Status.SUPERSEDED
    assert evaluation.activated_at is None
    assert evaluation.completed_at is not None
    assert evaluation.superseded_at == evaluation.completed_at
    assert evaluation_saves
    assert run.stage == GraphBuildRun.Stage.SUPERSEDED
    assert run.status == GraphBuildRun.Status.CANCELLED
    assert run.stage_marker["evaluation_completed"] is True
    assert run.lease_owner == ""
    assert run.lease_expires_at is None
    assert run_saves


def _database_is_reachable() -> bool:
    database = settings.DATABASES["default"]
    try:
        with socket.create_connection(
            (database["HOST"], int(database.get("PORT") or 5432)), timeout=0.2
        ):
            return True
    except OSError:
        return False


database_required = pytest.mark.skipif(
    not _database_is_reachable() and os.environ.get("KG_REQUIRE_POSTGRES_TESTS") != "1",
    reason="configured PostgreSQL database is not reachable",
)


@lru_cache(maxsize=1)
def _ontology():
    path = Path(__file__).resolve().parents[1] / "ontologies" / "research-v1.yaml"
    return load_ontology(path)


def _evidence(mention_id: int, relation_type: str = "uses_dataset", **overrides):
    values = {
        "relation_mention_id": mention_id,
        "document_artifact_id": 201,
        "chunk_id": 301,
        "head_mention_id": 401,
        "tail_mention_id": 402,
        "head_mapping_id": 501,
        "tail_mapping_id": 502,
        "head_collection_entity_id": 601,
        "tail_collection_entity_id": 602,
        "head_entity_type": "model",
        "tail_entity_type": "dataset",
        "head_status": "active",
        "tail_status": "active",
        "relation_type": relation_type,
        "extraction_confidence": 0.8,
    }
    values.update(overrides)
    return AssemblyEvidenceInput(**values)


def test_relation_projection_deduplicates_assertions_but_retains_each_evidence():
    first = _evidence(1, extraction_confidence=0.8)
    repeated = replace(first, relation_mention_id=2, extraction_confidence=0.5)

    result = plan_collection_relations((repeated, first), _ontology())

    assert len(result.relations) == 1
    relation = result.relations[0]
    assert relation.key == (601, "uses_dataset", 602)
    assert relation.support_count == 2
    assert relation.evidence_mention_ids == (1, 2)
    assert relation.confidence == pytest.approx(0.8)
    assert [item.disposition for item in result.evidence] == [
        EvidenceDisposition.PROMOTED,
        EvidenceDisposition.PROMOTED,
    ]


def test_contradictory_relation_types_remain_distinct_assertions():
    supports = _evidence(
        1,
        relation_type="supports",
        head_entity_type="claim",
        tail_entity_type="claim",
    )
    contradicts = replace(supports, relation_mention_id=2, relation_type="contradicts")

    result = plan_collection_relations((contradicts, supports), _ontology())

    assert {relation.relation_type for relation in result.relations} == {
        "supports",
        "contradicts",
    }
    assert all(relation.support_count == 1 for relation in result.relations)


def test_undirected_relations_canonicalize_by_stable_cluster_key_with_orientation():
    forward = _evidence(
        1,
        relation_type="compares_with",
        head_entity_type="model",
        tail_entity_type="model",
        head_collection_entity_id=700,
        tail_collection_entity_id=600,
        head_cluster_key="a" * 64,
        tail_cluster_key="b" * 64,
    )
    reverse = _evidence(
        2,
        relation_type="compares_with",
        head_entity_type="model",
        tail_entity_type="model",
        head_collection_entity_id=600,
        tail_collection_entity_id=700,
        head_cluster_key="b" * 64,
        tail_cluster_key="a" * 64,
    )

    result = plan_collection_relations((reverse, forward), _ontology())

    assert len(result.relations) == 1
    assert result.relations[0].key == (700, "compares_with", 600)
    assert result.relations[0].support_count == 2
    assert [item.orientation for item in result.evidence] == [
        "head_to_tail",
        "tail_to_head",
    ]


@pytest.mark.parametrize(
    ("evidence", "disposition", "reason"),
    [
        (
            _evidence(1, tail_entity_type="author"),
            EvidenceDisposition.REJECTED,
            "ontology_endpoint_types_invalid",
        ),
        (
            _evidence(2, tail_collection_entity_id=601, tail_mapping_id=501),
            EvidenceDisposition.SUPPRESSED,
            "self_loop",
        ),
        (
            _evidence(3, head_status="suppressed"),
            EvidenceDisposition.REJECTED,
            "inactive_endpoint",
        ),
        (
            _evidence(4, relation_type="same_as"),
            EvidenceDisposition.SUPPRESSED,
            "generic_identity_relation",
        ),
        (
            _evidence(
                5,
                head_mapping_id=None,
                head_collection_entity_id=None,
                head_entity_type=None,
                head_status=None,
            ),
            EvidenceDisposition.REJECTED,
            "missing_active_mapping",
        ),
    ],
)
def test_non_promotable_evidence_is_retained_with_an_audit_reason(
    evidence, disposition, reason
):
    result = plan_collection_relations((evidence,), _ontology())

    assert result.relations == ()
    assert len(result.evidence) == 1
    assert result.evidence[0].relation_mention_id == evidence.relation_mention_id
    assert result.evidence[0].disposition is disposition
    assert result.evidence[0].reason == reason


def test_graph_only_or_duplicate_evidence_fails_closed():
    with pytest.raises(ValueError, match="real chunk provenance"):
        plan_collection_relations((_evidence(1, chunk_id=None),), _ontology())

    with pytest.raises(ValueError, match="duplicate relation mention"):
        plan_collection_relations((_evidence(1), _evidence(1)), _ontology())


def test_assembly_config_checksum_covers_caps_and_generic_relations():
    base = AssemblyConfig()
    same = AssemblyConfig()
    changed_cap = replace(base, max_relations=base.max_relations - 1)
    changed_generic = replace(
        base,
        generic_identity_relations=base.generic_identity_relations | {"co_refers_to"},
    )

    assert assembly_config_checksum(base) == assembly_config_checksum(same)
    assert assembly_config_checksum(base) != assembly_config_checksum(changed_cap)
    assert assembly_config_checksum(base) != assembly_config_checksum(changed_generic)
    assert base.max_relations <= 50_000
    assert base.max_evidence <= 200_000


@pytest.mark.parametrize(
    ("field_name", "upper_bound"),
    (
        ("max_entities", ASSEMBLY_V1_MAX_ENTITIES),
        ("max_links", ASSEMBLY_V1_MAX_LINKS),
        ("max_relations", ASSEMBLY_V1_MAX_RELATIONS),
        ("max_evidence", ASSEMBLY_V1_MAX_EVIDENCE),
        ("max_orphan_entities", ASSEMBLY_V1_MAX_ORPHAN_ENTITIES),
    ),
)
def test_assembly_config_rejects_values_above_the_v1_operational_envelope(
    field_name, upper_bound
):
    with pytest.raises(ValueError, match="bounded range"):
        replace(AssemblyConfig(), **{field_name: upper_bound + 1})


def test_assembly_commit_uses_constant_size_ordered_row_roots():
    from apps.knowledge_graph.graph.assembly import (
        _assembly_marker,
        _ordered_checksum_root,
    )

    first = tuple(f"{index:064x}" for index in range(1, 1_001))
    reversed_rows = tuple(reversed(first))
    assert _ordered_checksum_root(first, "rows") == _ordered_checksum_root(
        first, "rows"
    )
    assert _ordered_checksum_root(first, "rows") != _ordered_checksum_root(
        reversed_rows, "rows"
    )
    source = inspect.getsource(_assembly_marker)
    assert "relation_row_checksum_root" in source
    assert "evidence_row_checksum_root" in source
    assert "relation_row_checksums" not in source
    assert "evidence_row_checksums" not in source


def test_projection_validation_checks_caps_provenance_and_orphans():
    plan = plan_collection_relations((_evidence(1),), _ontology())
    config = AssemblyConfig(max_orphan_entities=0, max_orphan_ratio=0.0)

    stats = validate_assembly_projection(
        plan,
        active_entity_ids=frozenset({601, 602}),
        provenanced_entity_ids=frozenset({601, 602}),
        config=config,
    )
    assert stats.entity_count == 2
    assert stats.relation_count == 1
    assert stats.evidence_count == 1
    assert stats.orphan_count == 0
    assert stats.orphan_ratio == 0.0

    with pytest.raises(ValueError, match="provenance"):
        validate_assembly_projection(
            plan,
            active_entity_ids=frozenset({601, 602}),
            provenanced_entity_ids=frozenset({601}),
            config=config,
        )
    with pytest.raises(ValueError, match="orphan"):
        validate_assembly_projection(
            plan_collection_relations((), _ontology()),
            active_entity_ids=frozenset({601}),
            provenanced_entity_ids=frozenset({601}),
            config=config,
        )


def test_orm_assembly_boundary_declares_locking_and_has_no_provider_imports():
    import apps.knowledge_graph.graph.assembly as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    forbidden = ("gliner2", "openai", "anthropic", "lib.llm")

    assert not any(
        name == prefix or name.startswith(f"{prefix}.")
        for name in imported
        for prefix in forbidden
    )
    assembly_source = inspect.getsource(assemble_collection_graph)
    assembly_lock_source = inspect.getsource(module._locked_candidate)
    assembly_row_source = inspect.getsource(module._load_locked_assembly_rows)
    activation_source = inspect.getsource(activate_collection_graph)
    assert "transaction.atomic" in assembly_source
    assert "_locked_candidate" in assembly_source
    assert "_load_locked_assembly_rows" in assembly_source
    assert "select_for_update" in assembly_lock_source
    assert "select_for_update" in assembly_row_source
    assert "select_for_update" in activation_source
    assert "transaction.atomic" in activation_source
    assert "_locked_candidate" in activation_source


def test_relation_and_evidence_rows_are_immutable_and_evidence_can_audit_rejection():
    from apps.knowledge_graph.models import (
        CollectionRelation,
        CollectionRelationEvidence,
    )

    assert {
        "artifact",
        "source",
        "relation_type",
        "target",
        "support_count",
        "confidence",
        "metadata",
        "created_at",
    } <= set(CollectionRelation._IMMUTABLE_FIELDS)
    assert set(CollectionRelation._IMMUTABLE_FIELDS) <= set(
        CollectionRelation._QUERYSET_IMMUTABLE_FIELDS
    )

    relation_field = CollectionRelationEvidence._meta.get_field("relation")
    assert relation_field.null
    assert relation_field.remote_field.on_delete.__name__ == "RESTRICT"
    assert (
        CollectionRelationEvidence._meta.get_field(
            "relation_mention"
        ).remote_field.on_delete.__name__
        == "PROTECT"
    )
    assert CollectionRelationEvidence._meta.get_field("head_mapping").null
    assert CollectionRelationEvidence._meta.get_field("tail_mapping").null
    assert {
        "artifact",
        "relation",
        "relation_mention",
        "head_mapping",
        "tail_mapping",
        "status",
        "reason",
        "ontology_checksum",
        "assembly_config_checksum",
        "orientation",
        "metadata",
        "created_at",
    } <= set(CollectionRelationEvidence._IMMUTABLE_FIELDS)
    assert set(CollectionRelationEvidence._IMMUTABLE_FIELDS) <= set(
        CollectionRelationEvidence._QUERYSET_IMMUTABLE_FIELDS
    )

    constraints = CollectionRelationEvidence._meta.constraints
    assert any(
        isinstance(item, UniqueConstraint)
        and tuple(item.fields) == ("artifact", "relation_mention")
        for item in constraints
    )
    orientation = CollectionRelationEvidence._meta.get_field("orientation")
    assert set(dict(orientation.choices)) == {"head_to_tail", "tail_to_head"}
    assert any(
        isinstance(item, CheckConstraint)
        and item.name == "kg_relation_evidence_decision_valid"
        for item in constraints
    )


def test_artifact_source_hashes_are_typed_sha256_constraints():
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    artifact_constraints = {item.name for item in GraphArtifact._meta.constraints}
    run_constraints = {item.name for item in GraphBuildRun._meta.constraints}

    assert "kg_artifact_source_hash_valid" in artifact_constraints
    assert "kg_build_source_hash_valid" in run_constraints


def test_current_state_querysets_exclude_building_artifacts_and_relations():
    from apps.knowledge_graph.models import CollectionRelation, GraphArtifact

    artifact_query = str(GraphArtifact.objects.current().query)
    relation_query = str(CollectionRelation.objects.current().query)

    assert GraphArtifact.Status.ACTIVE in artifact_query
    assert GraphArtifact.Status.BUILDING not in artifact_query
    assert "artifact" in relation_query.lower()
    assert GraphArtifact.Status.ACTIVE in relation_query
    assert "EXISTS" in relation_query.upper()
    assert "DISTINCT" not in relation_query.upper()


def test_task9_marker_fingerprints_all_raw_relation_evidence_for_task10():
    from apps.knowledge_graph.graph import assembly as module
    from apps.knowledge_graph.graph.assembly import _validate_task9_lineage_node
    from apps.knowledge_graph.resolution.collection import (
        _raw_relation_snapshot,
        persist_collection_resolution,
    )

    task9_source = inspect.getsource(persist_collection_resolution)
    task10_source = inspect.getsource(_validate_task9_lineage_node)
    fingerprint_source = inspect.getsource(_raw_relation_snapshot)

    assert "raw_relation_fingerprint" in task9_source
    assert "raw_relation_count" in task9_source
    assert "raw_relation_fingerprint" in task10_source
    assert "_raw_relation_snapshot" in task10_source
    assert "RelationMention.objects" in fingerprint_source
    assert ".iterator(" in fingerprint_source
    assert "ASSEMBLY_V1_MAX_EVIDENCE" not in fingerprint_source
    assert "max_relations" in fingerprint_source
    assert "max_relations=result.config.max_relations" in "".join(task9_source.split())
    assert 'max_relations=marker["max_relations"]' in "".join(task10_source.split())
    assert "tuple(query)" not in fingerprint_source

    projection_source = inspect.getsource(module._load_assembly_evidence)
    assert "config.max_evidence" in projection_source
    assert "ASSEMBLY_V1_MAX_EVIDENCE" not in projection_source


def test_activation_fence_remembers_a_newer_artifact_that_previously_won():
    from apps.knowledge_graph.graph.assembly import _newer_activation_exists

    never_activated = SimpleNamespace(
        pk=12,
        status="building",
        activated_at=None,
    )
    previously_activated = SimpleNamespace(
        pk=13,
        status="superseded",
        activated_at=object(),
    )

    assert not _newer_activation_exists(11, (never_activated,))
    assert _newer_activation_exists(11, (previously_activated,))


def test_scoped_activation_fence_isolates_evaluation_occurrences():
    from apps.knowledge_graph.graph.assembly import _newer_activation_exists

    production = SimpleNamespace(
        build_generation=2,
        evaluation_only=False,
        rebuild_request_id=None,
    )
    evaluation = SimpleNamespace(
        build_generation=3,
        status="building",
        evaluation_only=True,
        rebuild_request_id=uuid.uuid4(),
    )
    newer_production = SimpleNamespace(
        build_generation=4,
        status="building",
        evaluation_only=False,
        rebuild_request_id=None,
    )

    assert not _newer_activation_exists(production, (evaluation,))
    assert _newer_activation_exists(production, (newer_production,))


def test_contributor_locks_follow_task9_source_artifact_then_document_order():
    from apps.knowledge_graph.graph.assembly import _lock_current_contributors

    source = inspect.getsource(_lock_current_contributors)

    assert source.index("GraphArtifact.objects.select_for_update") < source.index(
        "model.objects.select_for_update"
    )


def test_collection_operations_share_advisory_collection_artifact_lock_order():
    from apps.knowledge_graph.graph.assembly import (
        _locked_candidate,
        lock_collection_graph_scope,
    )
    from apps.knowledge_graph.graph.filtering import create_filter_rerun_artifact
    from apps.knowledge_graph.resolution.collection import build_collection_snapshot

    entry = inspect.getsource(lock_collection_graph_scope)
    candidate = inspect.getsource(_locked_candidate)
    snapshot = inspect.getsource(build_collection_snapshot)
    filtering = inspect.getsource(create_filter_rerun_artifact)

    assert entry.index("_lock_collection_scope") < entry.index(
        "Collection.objects.select_for_update"
    )
    assert candidate.index("lock_collection_graph_scope") < candidate.index(
        "GraphArtifact.objects.select_for_update"
    )
    assert (
        snapshot.index("lock_collection_graph_scope")
        < snapshot.index("scope_artifacts = tuple")
        < snapshot.index("sources = _bounded_batched_query_rows")
    )
    assert (
        filtering.index("lock_collection_graph_scope")
        < filtering.index("scope_artifacts = tuple")
        < filtering.index("source_manifest = lock_collection_manifest_sources")
    )


def test_locked_projection_persistence_uses_batched_trusted_base_managers():
    from apps.knowledge_graph.graph.assembly import _write_assembly

    source = inspect.getsource(_write_assembly)

    assert source.count("._base_manager.bulk_create") == 2
    assert source.count("batch_size=_ASSEMBLY_INSERT_BATCH_SIZE") == 2
    assert ".objects.bulk_create" not in source


@pytest.mark.django_db(transaction=True)
@database_required
def test_postgres_assembly_reuses_task9_rows_and_activates_atomically():
    from django.contrib.auth.models import User

    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument, TextChunk
    from apps.knowledge_graph.graph.filtering import (
        FilterPolicy,
        filter_collection_resolution,
    )
    from apps.knowledge_graph.models import (
        CollectionEntity,
        CollectionEntityDocumentLink,
        CollectionRelation,
        CollectionRelationEvidence,
        DocumentEntity,
        DocumentEntityMention,
        EntityMention,
        GraphArtifact,
        GraphBuildRun,
        RelationMention,
    )
    from apps.knowledge_graph.resolution.collection import (
        CollectionEmbeddingSession,
        CollectionResolutionConfig,
        build_collection_snapshot,
        load_collection_filter_inputs,
        load_collection_resolution_inputs,
        persist_collection_resolution,
        resolve_collection_entities,
    )

    user = User.objects.create_user(username="kg-assembly", password="unused")
    collection = Collection.objects.create(name="Graph assembly")
    text = "Atlas uses MMLU. Atlas cites Ghost."
    document = RawTextDocument(
        title="Atlas",
        full_text=text,
        collection=collection,
        ingested_by=user,
        full_text_hash=RawTextDocument.hash_fn(text),
    )
    document.save(dont_rechunk=True)
    document_artifact = GraphArtifact.objects.create(
        scope_type=GraphArtifact.ScopeType.DOCUMENT,
        scope_id=document.id,
        status=GraphArtifact.Status.BUILDING,
        source_hash=document.full_text_hash,
        ontology_version=_ontology().version,
        extractor_version="extractor-v1",
        resolver_version="document-coreference-v1",
        filter_policy_version="document-filter-v1",
        ontology_checksum=_ontology().checksum,
    )
    chunk = TextChunk.objects.create(
        content=text,
        start_position=0,
        end_position=len(text),
        chunk_number=0,
        modality=TextChunk.Modality.TEXT,
        doc_id=document.id,
        embedding=[0.0] * 1024,
    )

    def mention(start, end, raw_text, entity_type):
        return EntityMention.objects.create(
            artifact=document_artifact,
            document_id=document.id,
            chunk=chunk,
            start=start,
            end=end,
            position_basis=EntityMention.PositionBasis.DOCUMENT_GLOBAL,
            raw_text=raw_text,
            normalized_text=raw_text,
            entity_type=entity_type,
            extraction_confidence=0.9,
        )

    atlas = mention(0, 5, "Atlas", "model")
    mmlu = mention(11, 15, "MMLU", "dataset")
    atlas_repeat = mention(17, 22, "Atlas", "model")
    ghost = mention(29, 34, "Ghost", "dataset")
    atlas_entity = DocumentEntity.objects.create(
        artifact=document_artifact,
        document_id=document.id,
        cluster_key="a" * 64,
        label="Atlas",
        normalized_label="atlas",
        entity_type="model",
        resolution_confidence=0.95,
    )
    mmlu_entity = DocumentEntity.objects.create(
        artifact=document_artifact,
        document_id=document.id,
        cluster_key="b" * 64,
        label="MMLU",
        normalized_label="mmlu",
        entity_type="dataset",
        resolution_confidence=0.95,
    )
    for raw_mention, entity, method in (
        (atlas, atlas_entity, DocumentEntityMention.Method.ROOT),
        (atlas_repeat, atlas_entity, DocumentEntityMention.Method.NORMALIZED_NAME),
        (mmlu, mmlu_entity, DocumentEntityMention.Method.ROOT),
    ):
        DocumentEntityMention.objects.create(
            document_entity=entity,
            mention=raw_mention,
            method=method,
            resolver_version=document_artifact.resolver_version,
            parent_mention_id=(
                "" if method == DocumentEntityMention.Method.ROOT else str(atlas.pk)
            ),
        )
    for confidence in (0.8, 0.7):
        RelationMention.objects.create(
            artifact=document_artifact,
            document_id=document.id,
            chunk=chunk,
            head=atlas,
            tail=mmlu,
            relation_type="uses_dataset",
            extraction_confidence=confidence,
        )
    RelationMention.objects.create(
        artifact=document_artifact,
        document_id=document.id,
        chunk=chunk,
        head=atlas,
        tail=atlas_repeat,
        relation_type="uses_dataset",
        extraction_confidence=0.6,
    )
    RelationMention.objects.create(
        artifact=document_artifact,
        document_id=document.id,
        chunk=chunk,
        head=atlas,
        tail=ghost,
        relation_type="uses_dataset",
        extraction_confidence=0.5,
    )
    document_artifact.status = GraphArtifact.Status.ACTIVE
    document_artifact.save(update_fields=["status"])
    embedding_signature = (
        f"test-local:model@rev:endpoint={'e' * 64}:dims=1024:"
        "prep=kg-entity-v1:max_chars=8192:batch=64"
    )
    policy = FilterPolicy(version="filter-v1", utility_activation_threshold=0.0)
    config = CollectionResolutionConfig()
    artifact, _manifest = build_collection_snapshot(
        collection=collection,
        document_artifacts=(document_artifact,),
        ontology=_ontology(),
        extractor_version="extractor-v1",
        resolver_version="collection-resolution-v1",
        filter_policy=policy,
        resolution_config=config,
        embedding_model_signature=embedding_signature,
    )
    run = GraphBuildRun.objects.create(
        artifact=artifact,
        stage=GraphBuildRun.Stage.RESOLUTION,
        status=GraphBuildRun.Status.RUNNING,
        attempt=1,
    )
    snapshot, source_entities, source_relations = load_collection_resolution_inputs(
        artifact.pk, run.pk
    )

    def no_embedding(_texts):
        raise AssertionError("type-separated singleton resolution must not embed")

    resolution = resolve_collection_entities(
        snapshot,
        source_entities,
        _ontology(),
        relations=source_relations,
        config=config,
        embedding_session=CollectionEmbeddingSession(
            expected_model_signature=embedding_signature,
            backend=no_embedding,
        ),
    )
    filter_inputs = load_collection_filter_inputs(artifact.pk, run.pk, resolution)
    filter_result = filter_collection_resolution(
        resolution, filter_inputs, _ontology(), policy
    )
    persist_collection_resolution(
        artifact.pk,
        run.pk,
        resolution,
        filter_result,
        filter_policy=policy,
        ontology=_ontology(),
    )
    entity_ids_before = tuple(
        CollectionEntity.objects.filter(artifact=artifact)
        .order_by("pk")
        .values_list("pk", flat=True)
    )
    link_ids_before = tuple(
        CollectionEntityDocumentLink.objects.filter(artifact=artifact)
        .order_by("pk")
        .values_list("pk", flat=True)
    )

    assembled = assemble_collection_graph(
        collection.pk,
        run.pk,
        artifact.source_hash,
        ontology=_ontology(),
    )

    artifact.refresh_from_db()
    assert artifact.status == GraphArtifact.Status.BUILDING
    assert not GraphArtifact.objects.current_collection(collection.pk).exists()
    assert not CollectionRelation.objects.current().filter(artifact=artifact).exists()
    assert (
        tuple(
            CollectionEntity.objects.filter(artifact=artifact)
            .order_by("pk")
            .values_list("pk", flat=True)
        )
        == entity_ids_before
    )
    assert (
        tuple(
            CollectionEntityDocumentLink.objects.filter(artifact=artifact)
            .order_by("pk")
            .values_list("pk", flat=True)
        )
        == link_ids_before
    )
    relation = CollectionRelation.objects.get(artifact=artifact)
    assert relation.support_count == 2
    assert relation.confidence == 0.8
    assert (
        CollectionRelationEvidence.objects.filter(
            artifact=artifact, status=CollectionRelationEvidence.Status.ACTIVE
        ).count()
        == 2
    )
    assert (
        CollectionRelationEvidence.objects.filter(
            artifact=artifact, status=CollectionRelationEvidence.Status.SUPPRESSED
        ).count()
        == 1
    )
    rejected = CollectionRelationEvidence.objects.get(
        artifact=artifact, status=CollectionRelationEvidence.Status.REJECTED
    )
    assert rejected.reason == "missing_active_mapping"
    assert rejected.head_mapping_id is not None
    assert rejected.tail_mapping_id is None

    activated = activate_collection_graph(
        collection.pk,
        run.pk,
        artifact.source_hash,
        ontology=_ontology(),
    )
    artifact.refresh_from_db()
    run.refresh_from_db()
    assert activated == assembled
    assert artifact.status == GraphArtifact.Status.ACTIVE
    assert run.stage == GraphBuildRun.Stage.COMPLETE
    assert run.status == GraphBuildRun.Status.SUCCEEDED
    assert CollectionRelation.objects.current().get(pk=relation.pk) == relation
    assert (
        activate_collection_graph(
            collection.pk,
            run.pk,
            artifact.source_hash,
            ontology=_ontology(),
        )
        == activated
    )
