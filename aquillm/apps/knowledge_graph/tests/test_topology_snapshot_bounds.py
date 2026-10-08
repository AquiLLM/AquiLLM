import asyncio
from dataclasses import replace
from threading import Event

import pytest

from apps.knowledge_graph.projection import topology_adapter
from apps.knowledge_graph.projection.topology_results import family_response
from apps.knowledge_graph.retrieval import projected_types as t
from apps.knowledge_graph.retrieval.topology import contracts as c
from apps.knowledge_graph.retrieval.topology import gateway_contracts as v1
from apps.knowledge_graph.retrieval.topology import gateway_snapshot_contracts as v2
from apps.knowledge_graph.retrieval.topology import gateway_snapshot_service as service2
from apps.knowledge_graph.retrieval.topology import memgraph
from apps.knowledge_graph.retrieval.topology.failures import (
    TopologyLoadError,
    TopologyResultCapError,
)
from apps.knowledge_graph.tests.test_topology_gateway_service import _body, _call
from apps.knowledge_graph.tests.test_topology_snapshot_adapter import fixture
from apps.knowledge_graph.tests.test_topology_snapshot_transport import headers, setup


def large_snapshot():
    """A real closed DTO whose individually valid V1 families exceed V2's wire cap."""
    driver, args, _ = fixture()
    adapter = topology_adapter.Neo4jProjectedTopologyQueryAdapter(
        driver, clock=lambda: 40.0
    )
    small = memgraph.MemgraphProjectedTopologyLoader(adapter).load(**args)
    original = next(
        row
        for row in small.artifact_provenance
        if row.scope_type is t.ProjectedScopeTypeV1.DOCUMENT
    )
    added = tuple(
        replace(original, scope_key=f"{i + 1:064x}", artifact_key=f"{i + 10001:064x}")
        for i in range(700)
    )
    documents = tuple(
        sorted((*small.allowed_scope.document_keys, *(row.scope_key for row in added)))
    )
    provenance = tuple(
        sorted((*small.artifact_provenance, *added), key=t._provenance_key)
    )
    snapshot = replace(
        small,
        allowed_scope=replace(small.allowed_scope, document_keys=documents),
        artifact_provenance=provenance,
    )
    query = c.TopologyQueryName.AUTOMATIC_MEMBERSHIPS
    wire = v1.encode_response(
        v1.TopologyGatewaySuccessV1(family_response(query, snapshot))
    )
    # Tune only harmless bounded document metadata. Leave room for V1 framing,
    # but not the other two complete canonical families and V2 manifest envelope.
    remaining = v1.MAX_RESPONSE_BYTES - 1000 - len(wire)
    assert 0 < remaining < len(added) * 500
    rows = []
    for row in provenance:
        grow = min(512 - len(row.extractor_version), remaining) if row in added else 0
        rows.append(replace(row, extractor_version=row.extractor_version + "x" * grow))
        remaining -= grow
    assert remaining == 0
    snapshot = replace(snapshot, artifact_provenance=tuple(rows))
    ready = args["ready"]
    authorized = tuple(
        c.AuthorizedProjectedDocumentV1(
            key,
            ready.selected_generations[0].collection_key,
            ready.selected_generations[0].generation_key,
        )
        for key in documents
    )
    ready = replace(
        ready,
        authorized_documents=authorized,
        bundle_checksum=c.ready_generation_bundle_checksum(
            ready.selected_generations,
            authorized,
            ready.authorization_context_signature,
        ),
    )
    return adapter, {**args, "ready": ready}, snapshot


def test_v1_valid_large_snapshot_selects_precisely_three_remaining_families(
    monkeypatch,
):
    adapter, args, snapshot = large_snapshot()
    monkeypatch.setattr(adapter, "_snapshot", lambda *_args, **_kwargs: snapshot)
    monkeypatch.setattr(memgraph, "monotonic", lambda: 40.0)
    monkeypatch.setattr(service2.v1, "monotonic", lambda: 40.0)
    parameters = memgraph._parameters(args["ready"], args["seeds"], args["caps"])
    complete = adapter.execute_snapshot(
        parameters=parameters, deadline=args["deadline"]
    )
    with pytest.raises(OverflowError):
        v2.encode_response(complete)
    assert len(complete.snapshot_json.encode()) <= 2_000_000
    golden = memgraph.MemgraphProjectedTopologyLoader(adapter).load(**args)

    class Driver:
        snapshot_enabled = True

        def __init__(self):
            self.calls = []

        def execute_snapshot(self, **kwargs):
            self.calls.append(("snapshot", kwargs))
            return v2.decode_response(
                service2._snapshot_payload(
                    adapter, maximum=v1.MAX_RESPONSE_BYTES, **kwargs
                )
            )

        def execute_read(self, **kwargs):
            self.calls.append((kwargs["query"], kwargs))
            # Real V1 wire validation proves each family remains in its old domain.
            return v1.decode_response(
                v1.encode_response(
                    v1.TopologyGatewaySuccessV1(adapter.execute_read(**kwargs))
                )
            ).rows

    driver = Driver()
    actual = memgraph.MemgraphProjectedTopologyLoader(driver).load(**args)
    assert t.canonical_projected_snapshot_bytes(
        actual
    ) == t.canonical_projected_snapshot_bytes(golden)
    assert [query for query, _ in driver.calls] == [
        "snapshot",
        *list(c.TopologyQueryName)[1:],
    ]
    assert all(
        kwargs["deadline"] == args["deadline"] and kwargs["parameters"] == parameters
        for _, kwargs in driver.calls
    )


@pytest.mark.parametrize("stage", ["manifest", "hydrate", "serialize"])
def test_adapter_rejects_deadline_expiry_at_each_stage(monkeypatch, stage):
    driver, _, parameters = fixture()
    clock = [40.0]
    adapter = topology_adapter.Neo4jProjectedTopologyQueryAdapter(
        driver, clock=lambda: clock[0]
    )
    owner, name = (
        (topology_adapter, "canonical_projected_snapshot_bytes")
        if stage == "serialize"
        else (adapter, "_manifests" if stage == "manifest" else "_snapshot")
    )
    original = getattr(owner, name)

    def late(*args, **kwargs):
        result = original(*args, **kwargs)
        clock[0] = 43.0
        return result

    monkeypatch.setattr(owner, name, late)
    with pytest.raises(TimeoutError):
        adapter.execute_snapshot(parameters=parameters, deadline=42.5)


def test_loader_cpu_decode_completion_cannot_publish_late_success(monkeypatch):
    driver, args, parameters = fixture()
    adapter = topology_adapter.Neo4jProjectedTopologyQueryAdapter(
        driver, clock=lambda: 40.0
    )
    result = adapter.execute_snapshot(parameters=parameters, deadline=42.5)
    clock = [40.0]
    monkeypatch.setattr(memgraph, "monotonic", lambda: clock[0])
    original = memgraph.decode_projected_snapshot_json

    def late(raw):
        decoded = original(raw)
        clock[0] = 43.0
        return decoded

    monkeypatch.setattr(memgraph, "decode_projected_snapshot_json", late)

    class Driver:
        snapshot_enabled = True

        def execute_snapshot(self, **_):
            return result

        def execute_read(self, **_):
            pytest.fail("no fallback")

    with pytest.raises(TopologyLoadError) as caught:
        memgraph.MemgraphProjectedTopologyLoader(Driver()).load(**args)
    assert caught.value.reason is c.TopologyFailureReason.DIRECT_TOPOLOGY_TIMEOUT


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure,reason",
    [
        (TopologyResultCapError(), "result_cap"),
        (TimeoutError(), "deadline"),
        (
            TopologyLoadError(c.TopologyFailureReason.BACKEND_PROVENANCE_MISMATCH),
            "provenance",
        ),
    ],
)
async def test_source_and_attestation_errors_never_select_family_transport(
    monkeypatch, failure, reason
):
    runtime, _, parameters = setup(monkeypatch, cap=800)

    def fail(**_):
        raise failure

    monkeypatch.setattr(runtime.adapter, "execute_snapshot", fail)
    body = v2.encode_request(v2.TopologyGatewayRequestV2(parameters, 42.5))
    result = await _call(v2.SNAPSHOT_PATH, body=body, headers=headers(body))
    assert v2.decode_response(_body(result)).reason.value == reason


@pytest.mark.asyncio
async def test_final_encoding_is_inside_worker_ownership_and_deadline(monkeypatch):
    from apps.knowledge_graph.retrieval.topology.gateway_workers import (
        GatewayWorkerPool,
    )

    runtime, _, parameters = setup(monkeypatch)
    pool = GatewayWorkerPool()
    monkeypatch.setattr(service2.v1, "GATEWAY_WORKERS", pool)
    entered, release = Event(), Event()
    original = v2.encode_response
    clock = [40.0]
    monkeypatch.setattr(service2.v1, "monotonic", lambda: clock[0])

    def blocked(value):
        if type(value) is v2.TopologyGatewaySnapshotV2:
            entered.set()
            assert release.wait(3)
            clock[0] = 43.0
        return original(value)

    monkeypatch.setattr(v2, "encode_response", blocked)
    body = v2.encode_request(v2.TopologyGatewayRequestV2(parameters, 42.5))
    task = asyncio.create_task(
        _call(v2.SNAPSHOT_PATH, body=body, headers=headers(body))
    )
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        assert pool._slots.acquire(blocking=False)
        assert pool._slots.acquire(blocking=False)
        assert pool._slots.acquire(blocking=False)
        assert not pool._slots.acquire(blocking=False)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not pool._slots.acquire(blocking=False)
    finally:
        release.set()
        await asyncio.to_thread(pool._executor.shutdown, wait=True)
    # The CPU encoding path cannot return success after its absolute deadline.
    with pytest.raises(TimeoutError):
        service2._snapshot_payload(
            runtime.adapter,
            parameters=parameters,
            deadline=42.5,
            maximum=v1.MAX_RESPONSE_BYTES,
        )


@pytest.mark.asyncio
async def test_independent_final_snapshot_byte_cap_never_selects_family_delivery(
    monkeypatch,
):
    _, _, parameters = setup(monkeypatch, cap=800)
    monkeypatch.setattr(
        topology_adapter,
        "canonical_projected_snapshot_bytes",
        lambda _: b"x" * 2_000_001,
    )
    body = v2.encode_request(v2.TopologyGatewayRequestV2(parameters, 42.5))
    result = await _call(v2.SNAPSHOT_PATH, body=body, headers=headers(body))
    assert (
        v2.decode_response(_body(result)).reason is v1.GatewayFailureReason.RESULT_CAP
    )


@pytest.mark.parametrize("branch", list(c.HybridBranchKind))
def test_local_snapshot_failure_does_not_poison_an_independent_request(
    monkeypatch, branch
):
    driver, args, parameters = fixture()
    adapter = topology_adapter.Neo4jProjectedTopologyQueryAdapter(
        driver, clock=lambda: 40.0
    )
    result = adapter.execute_snapshot(parameters=parameters, deadline=42.5)
    monkeypatch.setattr(memgraph, "monotonic", lambda: 40.0)

    class Driver:
        snapshot_enabled = True
        fail = True

        def execute_snapshot(self, **_):
            if self.fail:
                raise TopologyResultCapError
            return result

        def execute_read(self, **_):
            pytest.fail("no fallback")

    selected = Driver()
    loader = memgraph.MemgraphProjectedTopologyLoader(selected)
    with pytest.raises(TopologyLoadError) as caught:
        loader.load(**{**args, "caps": replace(args["caps"], branch_kind=branch)})
    assert caught.value.reason.value == f"{branch.value}_topology_invalid"
    selected.fail = False
    assert loader.load(**args).identity_keys
