"""Final selection and targeted refinement through the real common handoff."""

from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import pytest
from channels.db import database_sync_to_async

from apps.chat.refs import CollectionsRef
from apps.chat.tests.test_rag_preservation_routes import configure, fake_search
from apps.chat.tests.test_rag_source_document_tools import docs as docs
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage

pytestmark = [pytest.mark.asyncio, pytest.mark.django_db(transaction=True)]


@pytest.mark.parametrize("omit", ["source", "tail"])
async def test_real_final_selection_loses_lookup_support_without_replanning(
    docs, monkeypatch, omit
):
    from apps.chat.services import rag_pipeline, rag_preservation_turn
    from apps.chat.services.rag_turn import preservation_turn
    from apps.documents.models import TextChunk
    from lib.retrieval.evidence import (
        PreparedEvidence,
        SourceSpan,
        fingerprint_prepared_evidence,
    )

    configure(monkeypatch)
    monkeypatch.setenv("RAG_ITERATIVE_RETRIEVAL_ENABLED", "1")
    user, doc, chunks, _ = docs
    text = "Background. The release codename is Borealis."
    await database_sync_to_async(TextChunk.objects.filter(pk=chunks[0].pk).update)(
        content=text
    )
    await database_sync_to_async(TextChunk.objects.exclude(pk=chunks[0].pk).update)(
        content="Other context."
    )
    fake_search(monkeypatch)
    selects, requests = [], []
    original = rag_preservation_turn.prepare_selection_turn

    def select(*args, **kwargs):
        selects.append(True)
        turn = original(*args, **kwargs)
        kept = []
        for candidate in turn.selection.candidates:
            if candidate.chunk_id == chunks[0].pk:
                if omit == "source":
                    continue
                source = candidate.prepared_evidence.source
                spans = (
                    SourceSpan(
                        source.chunk_id, source.source_fingerprint, 0, 11, "Background."
                    ),
                )
                prepared = PreparedEvidence(
                    source,
                    spans,
                    fingerprint_prepared_evidence(source, spans),
                    3,
                    "partial",
                )
                candidate = replace(
                    candidate,
                    text="Background.",
                    row=dict(candidate.row) | {"text": "Background."},
                    prepared_evidence=prepared,
                    token_cost=3,
                )
            kept.append(candidate)
        return replace(turn, selection=replace(turn.selection, candidates=tuple(kept)))

    monkeypatch.setattr(rag_pipeline, "_prepare_selection_sync", select)

    async def complete(request, *args, **kwargs):
        requests.append(request)
        return request + [
            AssistantMessage(
                content="The delivered evidence does not establish the codename.",
                stop_reason="end_turn",
            )
        ], "changed"

    convo = Conversation(
        system="sys", messages=[UserMessage(content="What is the release codename?")]
    )
    consumer = SimpleNamespace(
        user=user, col_ref=CollectionsRef([doc.collection_id]), convo=convo
    )
    async with preservation_turn(consumer, max_func_calls=3) as runtime:
        await rag_pipeline.run_direct_rag_turn(
            consumer, SimpleNamespace(complete=complete), convo
        )
        acquired = runtime.observation["acquisition"]
        assert acquired.assessment.support
        packet = runtime.observation["delivered_packet"]
        assert packet.coverage_assessment.unresolved_aspects == ("release codename",)
        assert not packet.coverage_assessment.support
        assert runtime.budget.planner_calls == 0
    assert len(selects) == len(requests) == 1
    assert "Borealis" not in requests[0][-1].content
    assert "partial" in requests[0][-1].content


async def test_normal_handoff_refinement_obeys_strict_tool_deadline(docs, monkeypatch):
    import asyncio

    from apps.chat.services import rag_preservation_turn
    from apps.chat.services.rag_turn import preservation_turn
    from apps.chat.services.tool_wiring.documents import vector_search_tool
    from apps.documents.models import TextChunk
    from apps.documents.services.source_loading import current_source_runtime

    configure(monkeypatch)
    monkeypatch.setenv("RAG_ITERATIVE_RETRIEVAL_ENABLED", "1")
    monkeypatch.setenv("TOOL_CALL_TIMEOUT_SECONDS", "0.05")
    fake_search(monkeypatch)
    user, doc, chunks, _ = docs
    started, release, finished = Event(), Event(), Event()
    late, requests, selections = [], [], []
    original = rag_preservation_turn.prepare_selection_turn

    def select(*args, **kwargs):
        selections.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(rag_preservation_turn, "prepare_selection_turn", select)

    async def planner(**kwargs):
        return SimpleNamespace(
            text=(
                '{"requested_aspects":["missing aspect"],"support":[],'
                '"unresolved_aspects":["missing aspect"],"next_action":'
                '{"kind":"vector","query":"missing aspect",'
                '"aspect":"missing aspect"}}'
            )
        )

    def blocked_search(*args, **kwargs):
        runtime = current_source_runtime()
        started.set()
        release.wait(2)
        runtime.budget.publish(lambda: late.append(True))
        finished.set()
        return None, None, [], {}

    async def complete(request, *args, **kwargs):
        requests.append(request)
        return request + [
            AssistantMessage(
                content="Available evidence remains qualified.", stop_reason="end_turn"
            )
        ], "changed"

    convo = Conversation(
        system="sys",
        messages=[UserMessage(content="Summarize the source and missing aspect")],
    )
    consumer = SimpleNamespace(
        user=user, col_ref=CollectionsRef([doc.collection_id]), convo=convo
    )
    async with preservation_turn(consumer, max_func_calls=3) as runtime:
        raw = await database_sync_to_async(vector_search_tool(user, consumer.col_ref))(
            search_string="initial", top_k=5
        )
        monkeypatch.setattr(TextChunk, "text_chunk_search", blocked_search)
        working = convo + [
            ToolMessage(
                tool_name="vector_search",
                for_whom="assistant",
                content="initial",
                arguments={},
                result_dict=raw,
            )
        ]
        task = asyncio.create_task(
            rag_preservation_turn.normal_source_handoff(
                consumer,
                SimpleNamespace(complete=complete, base_args={}, get_message=planner),
                working,
                2048,
                None,
            )
        )
        try:
            assert await asyncio.to_thread(started.wait, 1)
            done, _ = await asyncio.wait({task}, timeout=0.6)
            assert task in done
            assert runtime.observation["acquisition"].stop_reason == "partial_unknown"
            assert runtime.budget.actions_used == 2
        finally:
            release.set()
            await task
            assert await asyncio.to_thread(finished.wait, 1)
    assert late == []
    assert len(selections) == len(requests) == 1
    assert chunks[0].content in requests[0][-1].content
