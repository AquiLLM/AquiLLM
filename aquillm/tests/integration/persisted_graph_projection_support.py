"""Real projection outbox, worker and private mapping."""
import os

import pytest

from apps.knowledge_graph.models import CollectionGraphProjection, GraphProjectionOutbox
from apps.knowledge_graph.projection.runtime import load_projection_runtime_settings
from apps.knowledge_graph.projection.worker import project_generation
from lib.knowledge_graph.query_extractor.contracts import QUERY_EXTRACTION_RESPONSE_SCHEMA_CHECKSUM
from lib.knowledge_graph.retrieval_config import load_django_hybrid_retrieval_settings


@pytest.fixture
def graph_configuration(monkeypatch, settings):
    assert os.environ["KG_MEMGRAPH_URI"] == "bolt://memgraph:7687"
    values = {
        # Declared identity of the owned synthetic HTTP model; no GPU claim.
        "APP_EMBED_MODEL": "synthetic-rag-model",
        "APP_EMBED_MODEL_REVISION": "synthetic-rag-model-v1",
        "KG_BUILD_ENABLED": "true",
        "KG_GRAPH_REMEDIATION_MODE": "v2",
        "KG_IDENTITY_POLICY": "contextual_v2",
        "KG_MEMGRAPH_PROJECTION_ENABLED": "1",
        "KG_MEMGRAPH_PROJECTION_HOOK_ENABLED": "1",
        "KG_MEMGRAPH_DATABASE": "memgraph",
        "KG_MEMGRAPH_PROJECTION_USERNAME": "synthetic_writer",
        "KG_MEMGRAPH_PROJECTION_PASSWORD": "synthetic_writer",
        "KG_PROJECTION_IDENTIFIER_HMAC_KEY": "p1-synthetic-identifier-key",
        "KG_PROJECTION_IDENTIFIER_KEY_VERSION": "p1-graph-v1",
        "KG_MEMGRAPH_TRAVERSAL_ENABLED": "1",
        "KG_GRAPH_DIRECT_ENABLED": "1",
        "KG_GRAPH_EXTENDED_ENABLED": "0",
        "KG_GRAPH_OVERALL_TIMEOUT_MS": "5000",
        "KG_GRAPH_DIRECT_TIMEOUT_MS": "5000",
        "KG_TOPOLOGY_GATEWAY_TIMEOUT_MS": "5000",
        "KG_TOPOLOGY_GATEWAY_MAX_REQUEST_BYTES": "4194304",
        "KG_TOPOLOGY_GATEWAY_MAX_RESPONSE_BYTES": "1048576",
        "KG_QUERY_EXTRACTOR_BEARER_TOKEN": "p1-synthetic-query",
        "KG_QUERY_EXTRACTOR_BUILD_HASH": "b" * 64,
        "KG_QUERY_EXTRACTOR_EXPECTED_SCHEMA_CHECKSUM": QUERY_EXTRACTION_RESPONSE_SCHEMA_CHECKSUM,
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    for name, value in load_django_hybrid_retrieval_settings(os.environ).items():
        setattr(settings, name, value)
    config = load_projection_runtime_settings()
    # Suppress only external broker dispatch; actual durable outbox remains.
    monkeypatch.setattr(
        "apps.knowledge_graph.projection.tasks.reconcile_knowledge_graph_projections.delay",
        lambda **kwargs: None,
    )
    return config


def project_active_collection(source):
    from apps.knowledge_graph.models import ProjectionChunkReference

    artifact = source.build_emitted_collection()
    projection = CollectionGraphProjection.objects.get(
        artifact_id=artifact.pk, state="pending"
    )
    assert projection.representation_snapshot is not None
    assert GraphProjectionOutbox.objects.filter(
        projection_id=projection.pk, operation="project", state="pending"
    ).exists()
    outcome = project_generation(projection_id=projection.pk, lease_owner="p1-graph-worker")
    assert outcome.ready, outcome.failure_code
    projection.refresh_from_db()
    assert projection.state == "ready"
    assert projection.graph_checksum == projection.snapshot_checksum
    refs = tuple(ProjectionChunkReference.objects.filter(projection_id=projection.pk))
    assert refs, "real function-backed worker must publish private chunk mappings"
    current = set(source.document.chunks.values_list("pk", flat=True))
    assert {r.integer_chunk_pk for r in refs} == current
    assert all(r.document_uuid == source.document.id for r in refs)
    assert all(r.chunk.representation_generation_id is not None for r in refs)
    return projection
