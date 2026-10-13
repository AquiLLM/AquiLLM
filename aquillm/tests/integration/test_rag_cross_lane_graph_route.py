"""First genuine nonnull graph-to-answer route in four actual modes."""
import pytest

from apps.chat.tests.rag_query_treatment_transport import selected_rows
from lib.embeddings.tests.test_document_provider import transport as transport
from .cross_lane_graph_route_observation import assert_graph_originals
from .cross_lane_graph_route_support import (
    close_graph_route_worker as close_graph_route_worker, graph_route_turn,
    route_corpus as route_corpus, TEXT, QUESTION,
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


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["baseline", "selection", "preservation", "combined"])
async def test_actual_graph_route_keeps_original_evidence_through_all_sinks(
    route_corpus, graph_route_score_transport, mode, monkeypatch, settings, tmp_path,
):
    corpus, projection, mapping = route_corpus
    provider = GraphRouteProvider()
    with owned_gateway(tmp_path) as gateway:
        async with graph_route_turn(corpus, gateway, mode=mode, question=QUESTION,
            collection_names=("selected",), query_entity=("model", "Mercury"),
            provider=provider, monkeypatch=monkeypatch, settings=settings) as observed:
            envelope = assert_graph_originals(observed)
            assert {p.projection_id for group in envelope.selection.groups
                    for p in group.scope.projections} == {projection.pk}
            chunk = observed.chunks[0]
            assert {mapping[ref.chunk_key] for ref in envelope.handoff.candidates} == {chunk.pk}
            assert all(ref.representation_generation_key == chunk.representation_generation_id
                       for ref in envelope.handoff.candidates)
            assert provider.answers and observed.message._validated_answer is not None
            rows = selected_rows(provider.answers[-1])
            assert any(row["text"] == TEXT for row in rows)
            packet = observed.turn.answer_publication.packet
            assert packet.query == QUESTION and packet.source_authorization is observed.runtime.authorization
            assert packet._publication_representation is observed.representation
            assert observed.trace.delivered_packet is packet
            citation = f"[doc:{chunk.doc_id} chunk:{chunk.pk}]"
            stored, events = await publish(observed.source, observed.consumer)
            assert len(events) == 2
            for content in (observed.message.content, stored, *(text for _, text in events)):
                assert citation in content
                assert content.index("Mercury uses DatasetA.") < content.index("This is a reported observation.")
            frontier = [event for event in observed.trace.events if event["event"] == "graph_frontier_budget"]
            assert len(frontier) == 1
            assert frontier[0]["invocation_id"] == envelope.frontier_invocation.invocation_id
            assert observed.trace.result(stored)["events"] == observed.trace.events
            assert observed.turn.budget.limits.actions == 3
            assert observed.turn.budget.actions_used == 1
            expected_plans = int(mode in ("preservation", "combined"))
            assert len(provider.plans) == observed.turn.budget.planner_calls == expected_plans
            # The one overlapping candidate uses legacy rank fallback; adaptive
            # final selection performs its own actual score dispatch.
            assert all(not kwargs["force_complete_rerank"]
                       for _, kwargs, _ in observed.audit.rankings)
            assert bool(graph_route_score_transport.calls) is (mode in ("selection", "combined"))
