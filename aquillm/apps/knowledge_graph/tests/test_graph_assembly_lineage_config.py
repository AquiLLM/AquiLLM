from __future__ import annotations

import inspect
from dataclasses import replace
from types import SimpleNamespace

from apps.knowledge_graph.graph import assembly


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
