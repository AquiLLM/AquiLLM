from __future__ import annotations

import inspect
from dataclasses import replace
from types import SimpleNamespace

import pytest

from apps.knowledge_graph.graph import assembly


def test_assembly_config_addresses_a_hard_v1_link_cap():
    config = assembly.AssemblyConfig()

    assert hasattr(assembly, "ASSEMBLY_V1_MAX_LINKS")
    assert config.max_links == assembly.ASSEMBLY_V1_MAX_LINKS
    default_checksum = assembly.assembly_config_checksum(config)
    lower_cap_checksum = assembly.assembly_config_checksum(
        replace(config, max_links=config.max_links - 1)
    )
    assert default_checksum != lower_cap_checksum
    with pytest.raises(ValueError, match="max_links.*bounded range"):
        replace(config, max_links=assembly.ASSEMBLY_V1_MAX_LINKS + 1)


def test_task9_links_are_counted_before_they_are_materialized():
    source = inspect.getsource(assembly._load_locked_task9_rows)

    assert "_bounded_query_rows(" in source
    assert "link_query" in source
    assert "link_cap" in source
    assert "tuple(link_query)" not in source


def test_task9_entities_are_counted_before_they_are_materialized():
    source = inspect.getsource(assembly._load_locked_task9_rows)

    assert "_bounded_query_rows(" in source
    assert "entity_query" in source
    assert "entity_cap" in source
    assert "tuple(entity_query)" not in source


def test_persisted_assembly_rows_are_bounded_before_materialization():
    loader = getattr(assembly, "_load_locked_assembly_rows", None)

    assert callable(loader)
    source = inspect.getsource(loader)
    compact = " ".join(source.split())
    assert "_bounded_query_rows( relation_query" in compact
    assert "_bounded_query_rows( evidence_query" in compact
    assert "config.max_relations" in source
    assert "config.max_evidence" in source
    assert "tuple(relation_query)" not in source
    assert "tuple(evidence_query)" not in source
    assert "_load_locked_assembly_rows(" in inspect.getsource(
        assembly.assemble_collection_graph
    )
    assert "_load_locked_assembly_rows(" in inspect.getsource(
        assembly._validate_locked_complete_artifact
    )


def test_collection_paths_lock_contributors_before_the_ontology_row():
    projection_source = inspect.getsource(assembly._locked_projection)
    active_source = inspect.getsource(
        assembly.validate_locked_active_collection_snapshot
    )

    assert projection_source.index("_validate_locked_manifest(") < (
        projection_source.index("_resolve_ontology(")
    )
    assert active_source.index("_validate_locked_manifest(") < active_source.index(
        "_resolve_ontology("
    )


def test_filter_source_lineage_uses_exact_persisted_assembly_config(monkeypatch):
    from apps.knowledge_graph.graph import filtering
    from apps.knowledge_graph.resolution.collection import build_collection_snapshot

    config = replace(
        assembly.AssemblyConfig(),
        max_document_inputs=7,
        max_filter_lineage_depth=2,
    )
    checksum = assembly.assembly_config_checksum(config)
    identity = {
        "scope_type": "collection",
        "scope_id": "17",
        "source_hash": "a" * 64,
        "ontology_version": "1.0.0",
        "extractor_version": "extractor-v1",
        "resolver_version": "resolver-v1",
        "filter_policy_version": "filter-v1",
        "embedding_model_signature": "embedding-v1",
        "ontology_checksum": "b" * 64,
        "filter_policy_checksum": "c" * 64,
        "resolution_config_checksum": "d" * 64,
        "assembly_version": config.version,
        "assembly_config_checksum": checksum,
    }
    source = SimpleNamespace(
        pk=11,
        metadata={"assembly_config": assembly._config_payload(config)},
        **identity,
    )
    run = SimpleNamespace(
        pk=12,
        attempt=1,
        stats={"collection_resolution_commit": {"version": 1}},
        **identity,
    )
    captured = []
    envelope_configs = []

    def validate_envelope(_artifact, _run, *, config=None):
        envelope_configs.append(config)
        return (
            "resolution",
            {
                "max_document_inputs": 7,
                "max_entities": 50_000,
                "max_memberships": 250_000,
                "max_relations": 250_000,
                "max_links": 250_000,
            },
        )

    def validate_lineage(*_args, config=None, **_kwargs):
        captured.append(config)
        return "e" * 64

    monkeypatch.setattr(assembly, "_validate_task9_lineage", validate_lineage)
    monkeypatch.setattr(
        assembly,
        "_validate_task9_marker_envelope",
        validate_envelope,
    )
    filtering._lock_filter_source_commit(
        source=source,
        source_manifest=(),
        source_entities=(),
        source_links=(),
        source_runs=(run,),
    )

    assert envelope_configs == [config]
    assert captured == [config]
    assert captured[0].max_document_inputs == 7
    assert captured[0].max_filter_lineage_depth == 2
    snapshot_source = inspect.getsource(build_collection_snapshot)
    assert '"assembly_config": _config_payload(assembly_config)' in snapshot_source


def test_persisted_assembly_config_rejects_checksum_drift():
    config = replace(assembly.AssemblyConfig(), max_filter_lineage_depth=1)
    artifact = SimpleNamespace(
        metadata={"assembly_config": assembly._config_payload(config)},
        assembly_version=config.version,
        assembly_config_checksum="0" * 64,
    )
    run = SimpleNamespace(
        stats={},
        assembly_version=config.version,
        assembly_config_checksum="0" * 64,
    )

    with pytest.raises(assembly.CollectionGraphAssemblyError, match="immutable"):
        assembly._persisted_assembly_config(artifact, run)


def test_scoped_task10_loads_custom_persisted_config_instead_of_default():
    from apps.knowledge_graph.models import GraphArtifact

    config = replace(
        assembly.AssemblyConfig(),
        max_evidence=123,
        max_filter_lineage_depth=2,
    )
    checksum = assembly.assembly_config_checksum(config)
    artifact = SimpleNamespace(
        metadata={"assembly_config": assembly._config_payload(config)},
        orchestration_version=GraphArtifact.OrchestrationVersion.SCOPED_V1,
        assembly_version=config.version,
        assembly_config_checksum=checksum,
    )
    run = SimpleNamespace(
        stats={},
        orchestration_version=GraphArtifact.OrchestrationVersion.SCOPED_V1,
        assembly_version=config.version,
        assembly_config_checksum=checksum,
    )

    assert assembly._resolve_config(artifact, run, None) == config
    source = inspect.getsource(assembly._resolve_config)
    assert "_persisted_assembly_config(artifact, run)" in source


def test_task9_marker_row_caps_reject_forged_counts_before_row_audits():
    from apps.knowledge_graph.models import GraphBuildRun

    config = assembly.AssemblyConfig()
    marker = {
        "version": 1,
        "policy_checksum": "a" * 64,
        "ontology_checksum": "b" * 64,
        "resolution_config_checksum": "c" * 64,
        "max_document_inputs": config.max_document_inputs,
        "max_entities": 1,
        "max_memberships": 1,
        "max_relations": 1,
        "max_links": 1,
        "filter_result_checksum": "d" * 64,
        "source_artifact_id": 2,
        "source_build_run_id": 3,
        "source_task9_marker_checksum": "e" * 64,
        "source_assembly_marker_checksum": None,
        "source_hash": "f" * 64,
        "assembly_version": config.version,
        "assembly_config_checksum": assembly.assembly_config_checksum(config),
        "manifest_count": 0,
        "entity_count": 2,
        "link_count": 2,
    }
    artifact = SimpleNamespace(
        pk=1,
        source_hash=marker["source_hash"],
        ontology_checksum=marker["ontology_checksum"],
        resolution_config_checksum=marker["resolution_config_checksum"],
        assembly_version=config.version,
        assembly_config_checksum=marker["assembly_config_checksum"],
        filter_policy_checksum=marker["policy_checksum"],
    )
    run = SimpleNamespace(
        Stage=GraphBuildRun.Stage,
        Status=GraphBuildRun.Status,
        stage=GraphBuildRun.Stage.FILTERING,
        status=GraphBuildRun.Status.SUCCEEDED,
        stats={"filter_commit": marker},
    )

    with pytest.raises(assembly.CollectionGraphAssemblyError, match="row cap"):
        assembly._validate_task9_marker_envelope(artifact, run, config=config)
    with pytest.raises(assembly.CollectionGraphAssemblyError, match="row cap"):
        assembly._validate_task9_lineage_node(
            artifact,
            run,
            (),
            (SimpleNamespace(), SimpleNamespace()),
            (SimpleNamespace(), SimpleNamespace()),
            config=config,
        )
    source = inspect.getsource(assembly._validate_task9_lineage_node)
    assert source.index('len(entities) > marker["max_entities"]') < source.index(
        "for row in entities"
    )
    assert source.index('len(links) > marker["max_links"]') < source.index(
        "for row in links"
    )


def test_task9_rows_are_materialized_under_exact_marker_caps():
    source = inspect.getsource(assembly._load_locked_task9_rows)

    assert "artifact: object, run: object, config: AssemblyConfig" in source
    assert "_task9_marker_query_caps(run, config)" in source
    assert "entity_query,\n            entity_cap," in source
    assert "link_query,\n            link_cap," in source


def test_filter_lineage_depth_is_hard_bounded_and_checksum_addressed():
    hard_cap = getattr(assembly, "ASSEMBLY_V1_MAX_FILTER_LINEAGE_DEPTH", None)

    assert type(hard_cap) is int and 1 <= hard_cap <= 128
    config = assembly.AssemblyConfig()
    assert config.max_filter_lineage_depth == hard_cap
    assert assembly.assembly_config_checksum(
        config
    ) != assembly.assembly_config_checksum(
        replace(config, max_filter_lineage_depth=hard_cap - 1)
    )
    with pytest.raises(ValueError, match="max_filter_lineage_depth.*bounded range"):
        replace(config, max_filter_lineage_depth=hard_cap + 1)


def test_filter_lineage_cycle_is_rejected_before_reloading_the_seen_source():
    walker = getattr(assembly, "_walk_task9_lineage", None)
    assert callable(walker)
    loaded = []

    def context(artifact_id):
        return SimpleNamespace(artifact=SimpleNamespace(pk=artifact_id))

    def validate_node(current):
        source_id = {1: 2, 2: 1}[current.artifact.pk]
        return "filter_rerun", {"source_artifact_id": source_id}, source_id

    def load_source(_current, marker):
        loaded.append(marker["source_artifact_id"])
        return context(marker["source_artifact_id"])

    with pytest.raises(assembly.CollectionGraphAssemblyError, match="cycle"):
        walker(
            context(1),
            max_filter_lineage_depth=4,
            validate_node=validate_node,
            load_source=load_source,
        )
    assert loaded == [2]


def test_filter_lineage_depth_is_rejected_before_loading_the_over_cap_hop():
    walker = getattr(assembly, "_walk_task9_lineage", None)
    assert callable(walker)
    loaded = []

    def context(artifact_id):
        return SimpleNamespace(artifact=SimpleNamespace(pk=artifact_id))

    def validate_node(current):
        source_id = current.artifact.pk + 1
        return "filter_rerun", {"source_artifact_id": source_id}, source_id

    def load_source(_current, marker):
        loaded.append(marker["source_artifact_id"])
        return context(marker["source_artifact_id"])

    with pytest.raises(assembly.CollectionGraphAssemblyError, match="depth cap"):
        walker(
            context(1),
            max_filter_lineage_depth=2,
            validate_node=validate_node,
            load_source=load_source,
        )
    assert loaded == [2, 3]


def test_filter_lineage_rows_are_counted_before_materialization_without_recursion():
    from apps.knowledge_graph.graph.manifest_locking import _read_manifest

    loader = getattr(assembly, "_load_filter_source_lineage", None)
    assert callable(loader)
    source = inspect.getsource(loader)

    compact = " ".join(source.split())
    assert "source_manifest = lock_collection_manifest_sources(" in compact
    manifest_reader = inspect.getsource(_read_manifest)
    assert "_bounded_rows(" in manifest_reader
    assert "maximum" in manifest_reader
    assert "_bounded_query_rows( source_entity_query" in compact
    assert "_bounded_query_rows( source_link_query" in compact
    assert "tuple(source_manifest_query)" not in source
    assert "tuple(source_entity_query)" not in source
    assert "tuple(source_link_query)" not in source
    assert "_validate_task9_lineage(" not in source


def test_contributor_chunk_validation_reuses_task7_bounded_loader():
    source = inspect.getsource(assembly._lock_current_contributors)

    assert "_ordered_chunks(" in source
    assert "_bounded_query_rows(" in source
    assert 'values_list("id", flat=True)' in source
    assert "TextChunk.objects.select_for_update" not in source


def test_endpoint_ids_are_split_into_bounded_deterministic_query_batches():
    batcher = getattr(assembly, "_id_batches", None)
    batch_size = getattr(assembly, "_ENDPOINT_ID_BATCH_SIZE", None)

    assert callable(batcher)
    assert type(batch_size) is int and 1 <= batch_size <= 10_000
    endpoint_ids = tuple(range(1, 20_003))
    batches = tuple(batcher(reversed(endpoint_ids)))

    assert all(1 <= len(batch) <= batch_size for batch in batches)
    assert tuple(value for batch in batches for value in batch) == endpoint_ids


def test_membership_loading_never_uses_the_full_endpoint_set_in_one_in_predicate():
    source = inspect.getsource(assembly._load_assembly_evidence)

    assert "_id_batches(endpoint_ids)" in source
    assert "mention_id__in=endpoint_batch" in source
    assert "mention_id__in=endpoint_ids" not in source


def test_provenance_memberships_batch_document_entity_predicates_too():
    source = inspect.getsource(assembly._active_entity_provenance)

    assert "_id_batches(document_entity_ids)" in source
    assert "document_entity_id__in=document_entity_batch" in source
    assert "document_entity_id__in=document_entity_ids" not in source
    assert 'select_related("mention", "document_entity")' in source
