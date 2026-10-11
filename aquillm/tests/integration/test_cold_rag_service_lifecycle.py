"""Real fresh-process admission, native leases, recovery, and shutdown."""

from concurrent.futures import ThreadPoolExecutor
import os
from time import monotonic

import pytest

from apps.knowledge_graph.retrieval.topology.frontier_gateway_client_v2 import (
    FrontierGatewayClientV2,
)
from apps.knowledge_graph.retrieval.topology.factory import (
    create_canonical_frontier_loader_v2,
)
from apps.knowledge_graph.retrieval.topology.gateway_client import (
    TopologyGatewayClient, TopologyLoadError,
)
from apps.knowledge_graph.tests.frontier_fixture_support import FrontierFixture
from lib.knowledge_graph.query_extractor.client import (
    QueryExtractorClient,
    QueryExtractorClientError,
    reconstruct_entity_texts,
)
from tests.integration.cold_service_support import cold_service, eventually


def extract(child):
    return QueryExtractorClient(child.settings).extract(
        query="A😀B", ontology=child.ontology, deadline=monotonic() + 2,
    )


def frontier(child):
    fixture = FrontierFixture()
    budget = fixture.budget()
    deadline = budget.deadline
    result = FrontierGatewayClientV2(TopologyGatewayClient(
        child.url, "synthetic-token", 1.0,
    )).execute_frontier_read(
        kind="manifest", generation=fixture.bundles[0].generation.generation_key,
        keys=(), ready=fixture.group.scope.ready, deadline=deadline, budget=budget,
    )
    assert budget.deadline == deadline and 0 < budget.wire_bytes <= 1048576
    return result


def bridge(child):
    fixture = FrontierFixture()
    budget = fixture.budget()
    deadline = budget.deadline
    loader = create_canonical_frontier_loader_v2(
        selection=fixture.selection, codec=fixture.codec,
        gateway_client=TopologyGatewayClient(child.url, "synthetic-token", 1.0),
    )
    result = loader.load(
        group=fixture.group, seeds=fixture.seeds, request=fixture.request,
        caps=fixture.caps, budget=budget, deadline=deadline,
    )
    assert fixture.answer_identity in result.snapshot.identity_keys
    assert budget.deadline == deadline
    assert 0 < budget.queries == child.count("frontier_entered") <= 64
    assert 0 < budget.source_rows <= 5000
    bodies = [row["size"] for row in child.events()
              if row["event"] == "http_body" and row["path"] == "/v2/topology/frontier"]
    assert len(bodies) == budget.queries
    assert 0 < budget.wire_bytes == sum(bodies) <= 1048576


def test_cold_extractor_failed_startup_requires_new_process_then_serves(tmp_path):
    with cold_service(tmp_path / "failed", "extractor", markers=("warmup_fail",)) as child:
        eventually(lambda: child.process.poll() is not None)
        assert child.count("lifespan.startup.failed") == 1
        assert child.count("lifespan.startup.complete") == 0
        assert child.count("warmup_entered") == 1
        assert child.count("inference_entered") == 0
        assert child.status("/healthz") is None
        failed_pid = child.process.pid
    with cold_service(tmp_path / "recovered", "extractor") as child:
        eventually(lambda: child.status("/healthz") == 200)
        assert child.process.pid not in (failed_pid, os.getpid())
        result = extract(child)
        assert reconstruct_entity_texts(query="A😀B", response=result) == ("😀",)
        names = [row["event"] for row in child.events()]
        assert names.index("warmup_finished") < names.index("lifespan.startup.complete")
        assert names.index("lifespan.startup.complete") < names.index("inference_entered")
        assert child.count("backend_constructed") == child.count("warmup_entered") == 1
        child.stop()
        assert child.count("lifespan.shutdown.complete") == 1
        assert child.status("/healthz") is None


def test_cold_extractor_timed_out_work_holds_native_lease_until_completion(tmp_path):
    with cold_service(tmp_path / "extractor", "extractor") as child:
        eventually(lambda: child.status("/healthz") == 200)
        child.marker("inference_block")
        with pytest.raises(QueryExtractorClientError) as timed_out:
            extract(child)
        assert timed_out.value.reason.value == "extractor_timeout"
        assert child.count("inference_entered") == 1
        assert child.count("inference_finished") == 0
        with pytest.raises(QueryExtractorClientError) as full:
            extract(child)
        assert full.value.reason.value == "extractor_provenance"
        assert child.count("inference_entered") == 1
        child.marker("inference_block", False)
        eventually(lambda: child.count("inference_finished") == 1)
        assert reconstruct_entity_texts(query="A😀B", response=extract(child)) == ("😀",)
        assert child.count("inference_entered") == 2
        child.stop()
        assert child.status("/healthz") is None


def test_cold_gateway_readiness_failure_recovers_without_replacing_runtime(tmp_path):
    with cold_service(tmp_path / "gateway", "gateway", markers=("readiness_fail",)) as child:
        eventually(lambda: child.status("/healthz") == 200)
        assert child.status("/readyz") == 503
        assert child.count("backend_constructed") == 1
        assert child.count("frontier_entered") == 0
        child.marker("readiness_fail", False)
        assert child.status("/readyz") == 200
        bridge(child)
        assert child.count("backend_constructed") == 1
        assert child.count("frontier_entered") == child.count("frontier_finished")
        child.stop()
        assert child.status("/readyz") is None


def test_cold_gateway_timed_out_work_retains_all_four_native_slots(tmp_path):
    with cold_service(tmp_path / "gateway", "gateway") as child:
        eventually(lambda: child.status("/readyz") == 200)
        child.marker("frontier_block")
        with ThreadPoolExecutor(max_workers=4) as callers:
            futures = [callers.submit(frontier, child) for _ in range(4)]
            for future in futures:
                with pytest.raises(TimeoutError):
                    future.result(timeout=5)
        assert child.count("frontier_entered") == 4
        assert child.count("frontier_finished") == 0
        readiness_count = child.count("readiness_entered")
        assert child.status("/readyz") == 503
        assert child.count("readiness_entered") == readiness_count
        with pytest.raises(TopologyLoadError):
            frontier(child)
        assert child.count("frontier_entered") == 4
        child.marker("frontier_block", False)
        eventually(lambda: child.count("frontier_finished") == 4)
        eventually(lambda: child.status("/readyz") == 200)
        assert frontier(child)
        assert child.count("frontier_entered") == 5
        child.stop()
        assert child.status("/readyz") is None
