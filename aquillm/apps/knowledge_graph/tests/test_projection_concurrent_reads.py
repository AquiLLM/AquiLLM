from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from threading import Event

from apps.knowledge_graph.projection import topology_adapter as module
from apps.knowledge_graph.projection.memgraph_driver import Neo4jMemgraphDriver
from apps.knowledge_graph.tests.test_projected_topology_adapter import (
    ProjectionDriver, _caps, _ready,
)
from apps.knowledge_graph.tests.test_projection_records import _bundle


def test_concurrent_first_driver_reads_initialize_one_client(monkeypatch):
    from neo4j import GraphDatabase

    entered, release, second_entered = Event(), Event(), Event()
    clients = []

    def create(*args, **kwargs):
        client = object()
        clients.append(client)
        entered.set()
        assert release.wait(2)
        return client

    monkeypatch.setattr(GraphDatabase, "driver", create)
    driver = Neo4jMemgraphDriver("bolt://localhost", "", "", database="memgraph")
    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(driver._connection)
        assert entered.wait(1)

        def second_read():
            second_entered.set()
            return driver._connection()

        second = workers.submit(second_read)
        try:
            assert second_entered.wait(1)
        finally:
            release.set()
        assert first.result() is second.result()
        assert len(clients) == 1


def test_cache_hit_cannot_be_evicted_between_lookup_and_lru_update(monkeypatch):
    entered, release, hydration = Event(), Event(), Event()
    bundle = _bundle()
    ready = _ready(bundle)
    adapter = module.Neo4jProjectedTopologyQueryAdapter(ProjectionDriver(), clock=lambda: 1.0)
    hit_parameters = {"seed_checksum": "hit", "caps_json": "caps"}
    miss_parameters = {"seed_checksum": "miss", "caps_json": "caps"}
    hit_key = (ready.bundle_checksum, "hit", "caps")
    hit, fresh = object(), object()

    class PausingCache(OrderedDict):
        def get(self, key):
            value = super().get(key)
            if key == hit_key:
                entered.set()
                assert release.wait(2)
            return value

    adapter._cache = PausingCache({hit_key: hit})
    for index in range(7):
        adapter._cache[("other", str(index), "caps")] = object()

    def read(*args, **kwargs):
        hydration.set()
        return bundle

    monkeypatch.setattr(module, "read_bundle", read)
    monkeypatch.setattr(module, "build_projected_topology_snapshot", lambda **_: fresh)
    with ThreadPoolExecutor(max_workers=2) as workers:
        # Start the cache miss first, pause its DB read until hit lookup is held.
        def paused_read(*args, **kwargs):
            assert entered.wait(1)
            return read()

        monkeypatch.setattr(module, "read_bundle", paused_read)
        miss_future = workers.submit(adapter._snapshot, ready, (), _caps(), miss_parameters, deadline=2.0)
        hit_future = workers.submit(adapter._snapshot, ready, (), _caps(), hit_parameters, deadline=2.0)
        try:
            assert hydration.wait(1), "cache locking serialized database hydration"
            # The old unprotected insertion finishes and evicts the held hit.
            try:
                miss_future.result(timeout=0.05)
            except TimeoutError:
                pass
        finally:
            release.set()
        assert hit_future.result() is hit
        assert miss_future.result() is fresh
    assert len(adapter._cache) == 8
