import pytest

from apps.knowledge_graph.projection import memgraph_records, topology_adapter
from apps.knowledge_graph.retrieval.topology import memgraph
from apps.knowledge_graph.retrieval.topology.contracts import ProjectedSeedV1
from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError
from apps.knowledge_graph.tests.test_projected_topology_adapter import (
    ProjectionDriver,
    _caps,
    _ready,
)
from apps.knowledge_graph.tests.test_projection_records import _bundle


def test_diagnostics_reject_payload_labels_and_logging_failure_is_nonfatal(monkeypatch):
    from time import monotonic

    from apps.knowledge_graph.retrieval.topology import diagnostics as d

    for kwargs in (
        {"phase": "private"},
        {"family": "private"},
        {"branch": "private"},
        {"maximum": 50_001},
    ):
        with pytest.raises((TypeError, ValueError)):
            d.record_topology_failure(
                **{
                    "phase": d.TopologyDiagnosticPhase.FAMILY_SCHEMA,
                    "started": monotonic(),
                    **kwargs,
                }
            )
    monkeypatch.setattr(
        d.logger,
        "info",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("logger down")),
    )
    d.record_topology_failure(
        phase=d.TopologyDiagnosticPhase.FAMILY_SCHEMA, started=monotonic()
    )


def test_compose_failure_emits_safe_boundary(caplog, monkeypatch):
    from apps.knowledge_graph.tests.test_memgraph_topology import (
        Driver,
        K,
        ready,
        snapshot,
    )

    monkeypatch.setattr(
        memgraph,
        "compose_projected_snapshot_families",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("private-source-text")),
    )
    with caplog.at_level("INFO"), pytest.raises(TopologyLoadError):
        memgraph.MemgraphProjectedTopologyLoader(Driver(snapshot())).load(
            ready=ready(),
            seeds=(ProjectedSeedV1(K[4], 1.0),),
            caps=_caps(),
            deadline=42.5,
        )
    assert "snapshot_compose" in caplog.text
    assert "private-source-text" not in caplog.text


def test_snapshot_build_failure_emits_safe_boundary(caplog, monkeypatch):
    bundle = _bundle()
    monkeypatch.setattr(
        topology_adapter,
        "build_projected_topology_snapshot",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("private-source-text")),
    )
    with caplog.at_level("INFO"), pytest.raises(TopologyLoadError):
        memgraph.MemgraphProjectedTopologyLoader(
            topology_adapter.Neo4jProjectedTopologyQueryAdapter(
                ProjectionDriver(bundle), clock=lambda: 40.0
            )
        ).load(
            ready=_ready(bundle),
            seeds=(ProjectedSeedV1(bundle.entities[0].entity_key, 1.0),),
            caps=_caps(),
            deadline=42.5,
        )
    assert "snapshot_build" in caplog.text
    assert "private-source-text" not in caplog.text
    assert bundle.generation.generation_key not in caplog.text


@pytest.mark.parametrize(
    "maximum,mutate,phase",
    [(1, False, "source_family_cap"), (200, True, "family_schema")],
)
def test_source_family_rejection_emits_fixed_family_without_payload(
    caplog, maximum, mutate, phase
):
    bundle = _bundle()
    driver = ProjectionDriver(bundle)
    if mutate:
        original = driver.execute_read

        def malformed(*args, **kwargs):
            rows = original(*args, **kwargs)
            rows[0]["record"]["opaque_key"] = "private-source-text"
            return rows

        driver.execute_read = malformed
    with caplog.at_level("INFO"), pytest.raises(ValueError):
        memgraph_records._read_topology_family(
            driver,
            query="MATCH (n:ProjectedEntity {}) RETURN n",
            parameters={},
            label="ProjectedEntity",
            kind=memgraph_records.FAMILIES[0][1],
            maximum=maximum,
            timeout=2.5,
            reject_full_pages=True,
        )
    assert phase in caplog.text
    assert "ProjectedEntity" in caplog.text
    assert "private-source-text" not in caplog.text
    assert bundle.entities[0].entity_key not in caplog.text
