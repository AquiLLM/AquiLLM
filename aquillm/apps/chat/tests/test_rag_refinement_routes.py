"""Bounded refinement through the actual adapter and one final coordinator."""

import json
from types import SimpleNamespace

import pytest

from apps.chat.refs import CollectionsRef
from apps.chat.tests.test_rag_preservation_routes import configure, fake_search
from apps.chat.tests.test_rag_source_document_tools import docs as docs
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, UserMessage

pytestmark = [pytest.mark.asyncio, pytest.mark.django_db(transaction=True)]


@pytest.mark.parametrize("cap", [1, 2, 3])
async def test_configured_cap_gathers_then_selects_and_synthesizes_once(
    docs, monkeypatch, cap
):
    from apps.chat.services import rag_pipeline, rag_preservation_turn
    from apps.chat.services.rag_turn import preservation_turn
    from apps.documents.models import TextChunk
    from apps.documents.services.source_loading import source_query_rows

    configure(monkeypatch)
    monkeypatch.setenv("RAG_ITERATIVE_RETRIEVAL_ENABLED", "1")
    monkeypatch.setenv("RAG_DIRECT_MAX_QUERIES", str(cap))
    user, doc, chunks, _ = docs
    searches, planners, selections, synthesis = [], [], [], []

    def search(query, top_k, documents, **kwargs):
        searches.append(query)
        rows = source_query_rows(
            TextChunk.objects.filter(pk=chunks[len(searches) - 1].pk)
        )
        return None, None, rows, {}

    async def planner(**kwargs):
        planners.append(kwargs)
        return SimpleNamespace(
            text=json.dumps(
                {
                    "requested_aspects": ["measurement"],
                    "support": [],
                    "unresolved_aspects": ["measurement"],
                    "next_action": {
                        "kind": "vector",
                        "query": f"measurement {len(planners)}",
                        "aspect": "measurement",
                    },
                }
            )
        )

    original = rag_preservation_turn.prepare_selection_turn

    def select(*args, **kwargs):
        selections.append(True)
        return original(*args, **kwargs)

    async def complete(request, *args, **kwargs):
        synthesis.append(request)
        return request + [
            AssistantMessage(content="Supported portions", stop_reason="end_turn")
        ], "changed"

    monkeypatch.setattr(TextChunk, "text_chunk_search", search)
    monkeypatch.setattr(rag_pipeline, "_prepare_selection_sync", select)
    convo = Conversation(
        system="sys", messages=[UserMessage(content="Explain the measurement")]
    )
    consumer = SimpleNamespace(
        user=user, col_ref=CollectionsRef([doc.collection_id]), convo=convo
    )
    llm = SimpleNamespace(complete=complete, get_message=planner, base_args={})
    async with preservation_turn(consumer, max_func_calls=5) as runtime:
        await rag_pipeline.run_direct_rag_turn(consumer, llm, convo)
        assert runtime.budget.actions_used == cap
        assert runtime.budget.planner_calls == cap - 1
        assert runtime.budget.pairs_used["final"] <= 45
        if cap == 3:
            assert (
                runtime.observation["acquisition"].assessment.certainty == "unassessed"
            )
        delivered = runtime.observation["delivered_packet"]
        assert {r["chunk_id"] for r in delivered.chunks} == {c.pk for c in chunks[:cap]}
    assert len(searches) == cap
    assert len(planners) == cap - 1
    assert all(p["max_tokens"] == 512 for p in planners)
    assert len(selections) == len(synthesis) == 1


async def test_manual_revocation_before_handoff_has_no_provider_request(
    docs, monkeypatch
):
    from apps.chat.services import rag_preservation_turn
    from apps.chat.services.manual_search_turn import run_manual_search_turn
    from apps.chat.services.rag_turn import preservation_turn
    from apps.collections.models import CollectionPermission

    configure(monkeypatch)
    fake_search(monkeypatch)
    user, doc, _, _ = docs
    original = rag_preservation_turn.prepare_selection_turn

    def select(*args, **kwargs):
        result = original(*args, **kwargs)
        CollectionPermission.objects.filter(user=user).delete()
        return result

    async def complete(*args, **kwargs):
        pytest.fail("revoked manual source must not reach the provider")

    monkeypatch.setattr(rag_preservation_turn, "prepare_selection_turn", select)
    convo = Conversation(
        system="sys", messages=[UserMessage(content=f"/search {doc.id} summarize")]
    )
    consumer = SimpleNamespace(user=user, col_ref=CollectionsRef([]), convo=convo)
    async with preservation_turn(consumer, max_func_calls=3) as runtime:
        await run_manual_search_turn(
            consumer, SimpleNamespace(complete=complete), convo
        )
        assert runtime.budget.actions_used == 1
    assert "limits" in consumer.convo[-1].content


async def test_all_off_keeps_the_legacy_direct_route(docs, monkeypatch):
    from apps.chat.services import rag_pipeline, rag_preservation_turn
    from apps.chat.services.rag_turn import preservation_turn

    configure(monkeypatch, source=False)
    fake_search(monkeypatch)
    user, doc, _, _ = docs

    async def forbidden(*args, **kwargs):
        pytest.fail("all-off route must not enter preservation controller")

    async def complete(request, *args, **kwargs):
        return request + [
            AssistantMessage(content="Legacy answer", stop_reason="end_turn")
        ], "changed"

    monkeypatch.setattr(rag_preservation_turn, "run_preservation_rag", forbidden)
    convo = Conversation(
        system="sys", messages=[UserMessage(content="Summarize selected documents")]
    )
    consumer = SimpleNamespace(
        user=user, col_ref=CollectionsRef([doc.collection_id]), convo=convo
    )
    async with preservation_turn(consumer, max_func_calls=3) as runtime:
        assert runtime is None
        assert (
            await rag_pipeline.run_direct_rag_turn(
                consumer, SimpleNamespace(complete=complete), convo
            )
            == "handled"
        )
    assert consumer.convo[-1].content == "Legacy answer"
