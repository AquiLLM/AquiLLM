import pytest

from apps.knowledge_graph.retrieval.topology.factory import (
    create_evaluation_projected_topology_loader,
    create_projected_topology_loader,
)
from apps.knowledge_graph.retrieval.topology.postgres import (
    _make_test_postgres_parity_capability,
)


class Source:
    def __init__(self):
        self.calls = []

    def load(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs.pop("snapshot") if "snapshot" in kwargs else object()


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
