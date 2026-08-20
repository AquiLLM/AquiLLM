from dataclasses import replace

import pytest

from apps.knowledge_graph.retrieval.topology.contracts import (
    HybridBranchKind,
    ProjectedSeedV1,
    TopologyCapsV1,
)
from apps.knowledge_graph.retrieval.topology.factory import (
    create_evaluation_projected_topology_loader,
    create_projected_topology_loader,
)
from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError
from apps.knowledge_graph.retrieval.topology.postgres import (
    PostgresProjectedTopologyLoader,
    _make_test_postgres_parity_capability,
)
from apps.knowledge_graph.tests.test_memgraph_topology import K, ready, snapshot


class Source:
    def __init__(self, result=None):
        self.calls, self.result = [], result

    def load(self, **kwargs):
        self.calls.append(kwargs)
        return self.result if self.result is not None else object()


def test_production_factory_accepts_only_memgraph_and_never_falls_back() -> None:
    class Driver:
        def execute_read(self, **_kwargs):
            return ()

    driver = Driver()
    loader = create_projected_topology_loader(backend="memgraph", driver=driver)
    assert loader.driver is driver
    with pytest.raises(ValueError, match="production.*memgraph"):
        create_projected_topology_loader(backend="postgres", driver=driver)


def test_postgres_parity_requires_exact_private_capability() -> None:
    source = Source()
    capability = _make_test_postgres_parity_capability(source=source)
    loader = create_evaluation_projected_topology_loader(
        backend="postgres", source=source, capability=capability
    )
    assert loader.source is source
    with pytest.raises(ValueError, match="capability"):
        create_evaluation_projected_topology_loader(
            backend="postgres", source=source, capability=object()
        )


@pytest.mark.parametrize(
    ("loaded_depth", "requested_depth", "rejected"),
    ((2, 1, True), (1, 2, False)),
)
def test_postgres_parity_loader_enforces_requested_depth_cap(
    loaded_depth: int, requested_depth: int, rejected: bool
) -> None:
    source = Source(replace(snapshot(), load_max_hops=loaded_depth))
    capability = _make_test_postgres_parity_capability(source=source)
    loader = PostgresProjectedTopologyLoader(source, capability)

    def load():
        return loader.load(
            capability=capability,
            ready=ready(),
            seeds=(ProjectedSeedV1(K[4], 1.0),),
            caps=TopologyCapsV1(
                HybridBranchKind.DIRECT, 32, requested_depth, 200, 1_000, 20
            ),
            deadline=42.5,
        )

    if rejected:
        with pytest.raises(TopologyLoadError) as captured:
            load()
        assert captured.value.reason.value == "direct_topology_invalid"
    else:
        assert load().load_max_hops == loaded_depth
