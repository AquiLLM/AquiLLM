"""Actual family sentinel overflow must remain a branch-local failure."""

from dataclasses import replace

import pytest

from apps.knowledge_graph.projection.topology_adapter import (
    Neo4jProjectedTopologyQueryAdapter,
)
from apps.knowledge_graph.retrieval.branch_contracts import (
    BranchStatusV1,
    DirectBranchFailureReason,
    ExtendedBranchFailureReason,
    SharedBranchFailureReason,
)
from apps.knowledge_graph.retrieval.scheduler import HybridGraphBranchScheduler
from apps.knowledge_graph.retrieval.topology import gateway_service as service
from apps.knowledge_graph.retrieval.topology.contracts import (
    HybridBranchKind,
    ProjectedSeedV1,
    TopologyQueryName,
)
from apps.knowledge_graph.retrieval.topology.gateway_contracts import (
    GatewayFailureReason,
    TopologyGatewayRequestV1,
    decode_response,
    encode_request,
)
from apps.knowledge_graph.retrieval.topology.memgraph import (
    MemgraphProjectedTopologyLoader,
    _parameters,
)
from apps.knowledge_graph.tests.test_projected_topology_adapter import (
    ProjectionDriver,
    _caps,
    _ready,
)
from apps.knowledge_graph.tests.test_projection_records import _bundle
from apps.knowledge_graph.tests.test_retrieval_branch_scheduler import (
    _Runtime,
    _settings,
)
from apps.knowledge_graph.tests.test_topology_gateway_service import (
    _body,
    _call,
    _headers,
)
from apps.knowledge_graph.tests.test_topology_gateway_service import (
    _settings as _gateway_settings,
)


class FamilyDriver(ProjectionDriver):
    def __init__(self, *, malformed=False):
        super().__init__(_bundle())
        self.malformed = malformed

    def execute_read(self, cypher, parameters, **kwargs):
        rows = super().execute_read(cypher, parameters, **kwargs)
        if self.malformed and "(n:ProjectedEntity " in cypher:
            # The extra row is malformed, so schema corruption must still win
            # over the otherwise explicit maximum+1 result-cap sentinel.
            rows[-1]["record"].pop("entity_key")
        return rows


def _request_parts(branch, *, malformed=False):
    driver = FamilyDriver(malformed=malformed)
    adapter = Neo4jProjectedTopologyQueryAdapter(driver, clock=lambda: 40.0)
    ready = _ready(driver.bundle)
    seeds = (ProjectedSeedV1(driver.bundle.entities[0].entity_key, 1.0),)
    caps = replace(_caps(), branch_kind=branch, max_nodes=1)
    return adapter, ready, seeds, caps


@pytest.mark.parametrize("branch", tuple(HybridBranchKind))
@pytest.mark.parametrize("malformed", (False, True))
def test_family_sentinel_cap_preserves_sibling_but_schema_corruption_does_not(
    branch,
    malformed,
):
    adapter, ready, seeds, caps = _request_parts(branch, malformed=malformed)
    runtime = _Runtime()

    def load(**_kwargs):
        return MemgraphProjectedTopologyLoader(adapter).load(
            ready=ready,
            seeds=seeds,
            caps=caps,
            deadline=42.5,
        )

    failed_name = "direct" if branch is HybridBranchKind.DIRECT else "extended"
    sibling_name = "extended" if failed_name == "direct" else "direct"
    setattr(runtime, "run_" + failed_name, load)
    outcome = HybridGraphBranchScheduler(runtime, clock=lambda: 40.0).run(
        query="q",
        baseline=object(),
        authorization=object(),
        settings=_settings(),
        deadline=42.5,
    )
    if malformed:
        assert (
            outcome.shared_failure_reason
            is SharedBranchFailureReason.BACKEND_SCHEMA_MISMATCH
        )
        assert outcome.direct.failure_reason is outcome.extended.failure_reason
    else:
        expected = (
            DirectBranchFailureReason.DIRECT_TOPOLOGY_INVALID
            if branch is HybridBranchKind.DIRECT
            else ExtendedBranchFailureReason.EXTENDED_TOPOLOGY_INVALID
        )
        assert getattr(outcome, failed_name).failure_reason is expected
        assert getattr(outcome, sibling_name).status is BranchStatusV1.SUCCEEDED
        assert outcome.shared_failure_reason is None
    assert not adapter._cache


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed", (False, True))
async def test_actual_adapter_sentinel_uses_result_cap_gateway_envelope(
    monkeypatch,
    malformed,
):
    adapter, ready, seeds, caps = _request_parts(
        HybridBranchKind.DIRECT,
        malformed=malformed,
    )
    settings = _gateway_settings()
    runtime = service.TopologyGatewayRuntime(settings, adapter._driver, adapter)
    monkeypatch.setattr(service, "monotonic", lambda: 40.0)
    monkeypatch.setattr(
        service, "load_topology_gateway_settings", lambda _env: settings
    )
    monkeypatch.setattr(service, "_get_runtime", lambda *_args: runtime)
    body = encode_request(
        TopologyGatewayRequestV1(
            TopologyQueryName.AUTOMATIC_MEMBERSHIPS,
            _parameters(ready, seeds, caps),
            42.5,
            caps.max_nodes,
        )
    )
    messages = await _call(body=body, headers=_headers(body))
    response = decode_response(_body(messages))
    assert messages[0]["status"] == (502 if malformed else 422)
    assert response.reason is (
        GatewayFailureReason.SCHEMA if malformed else GatewayFailureReason.RESULT_CAP
    )
