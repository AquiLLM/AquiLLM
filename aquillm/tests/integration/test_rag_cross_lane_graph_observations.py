"""Original producer capture across actual modes, with honest gates."""
from copy import deepcopy
from dataclasses import asdict

import pytest

from apps.chat.evals.evidence_graph_capture import graph_lane_observations
from apps.chat.evals.evidence_quality_runtime import comparison_controls
from apps.knowledge_graph.evals.graph_activation_attachment_v2 import validate_graph_activation_attachment_v2
from apps.knowledge_graph.evals.graph_frontier_observations import validate_frontier, validate_invocations
from apps.knowledge_graph.evals.graph_gate_observations import digest
from lib.embeddings.tests.test_document_provider import transport as transport

from .cross_lane_graph_route_observation import assert_graph_originals
from .cross_lane_graph_route_support import (
    close_graph_route_worker as close_graph_route_worker, graph_route_turn,
    route_corpus as route_corpus, QUESTION,
)
from .cross_lane_graph_route_transport import (
    GraphRouteProvider, graph_route_score_transport as graph_route_score_transport,
)
from .cross_lane_publication_support import publish
from .persisted_graph_corpus_support import graph_corpus as graph_corpus
from .persisted_graph_gateway_support import owned_gateway
from .persisted_graph_projection_support import graph_configuration as graph_configuration
from .persisted_graph_role_support import projection_roles as projection_roles
from .persisted_graph_source_support import upload as upload

pytestmark = pytest.mark.django_db(
    transaction=True, databases=["default", "projection_source", "projection_state"])
MODES = ("baseline", "selection", "preservation", "combined")


def capture_negatives(events):
    """Explicit detached corruptions, never substitute producer observations."""
    originals = [event for event in events if event["event"] == "graph_frontier_budget"]
    assert len(originals) == 1
    invalid = [[], deepcopy(originals + originals)]
    missing_time = deepcopy(originals)
    del missing_time[0]["at"]
    invalid.append(missing_time)
    overflow = deepcopy(originals)
    overflow[0]["frontier_budget"]["rows_used"] = overflow[0]["frontier_budget"]["max_rows"] + 1
    invalid.append(overflow)
    for damaged in invalid:
        graph = graph_lane_observations(damaged)["graph"]
        assert graph["frontier_capture_status"] == "inconclusive"
        assert graph["status"] == "unknown" and graph["frontier_capture_reasons"]
        assert graph["frontier_events"] == damaged
        assert len(graph["frontier_invocations"]) == len(damaged)
    assert graph_lane_observations(originals)["graph"]["frontier_capture_status"] == "singleton"


@pytest.mark.asyncio
async def test_four_real_mode_executions_retain_producer_events_without_inventing_attestation(
    route_corpus, graph_route_score_transport, monkeypatch, settings, tmp_path,
):
    corpus, _, _ = route_corpus
    quality, ids, turns = {}, [], []
    with owned_gateway(tmp_path) as gateway:
        for mode in MODES:
            provider = GraphRouteProvider()
            async with graph_route_turn(corpus, gateway, mode=mode, question=QUESTION,
                collection_names=("selected",), query_entity=("model", "Mercury"),
                provider=provider, monkeypatch=monkeypatch, settings=settings) as observed:
                envelope = assert_graph_originals(observed)
                stored, _ = await publish(observed.source, observed.consumer)
                result = observed.trace.result(stored)
                lanes = result["lane_observations"]
                assert lanes == graph_lane_observations(observed.trace.events)
                graph = lanes["graph"]
                original = asdict(envelope.frontier_invocation)
                assert graph["frontier_invocations"] == [original]
                assert graph["frontier_capture_status"] == "singleton"
                assert graph["frontier_events"] == [event for event in observed.trace.events
                    if event["event"] == "graph_frontier_budget"]
                assert graph["status"] == "unknown"
                assert "source_generation_sha256" not in graph and "runtime_sha256" not in graph
                validate_frontier(graph["frontier_budget"])
                consumer_row = {"inputs": {"question": QUESTION},
                    "frontier_budget": graph["frontier_budget"],
                    "request_algorithm_signature": graph["request_algorithm_signature"],
                    "frontier_invocations": graph["frontier_invocations"]}
                validate_invocations(consumer_row)
                changed = deepcopy(consumer_row)
                changed["inputs"]["question"] = "A different original question"
                with pytest.raises(ValueError, match="graph_invocation_binding_mismatch"):
                    validate_invocations(changed)
                capture_negatives(observed.trace.events)
                controls = comparison_controls()
                quality[mode] = {"observations": [{"comparison_controls": controls,
                    "snapshot": {"configuration": digest(controls)}, "lane_observations": lanes}]}
                ids.append(original["invocation_id"])
                turns.append(observed.turn)
    assert len(ids) == len(set(ids)) == 4
    assert len({id(turn) for turn in turns}) == 4
    # Actual frontier completeness is insufficient for runtime/study attachment.
    gate = validate_graph_activation_attachment_v2(None, quality_reports=quality)
    assert gate["passed"] is False and gate["applicable"] is None
    assert gate["blocking_reasons"] == ["graph_applicability_unknown"]
