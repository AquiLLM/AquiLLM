"""Final selection and targeted refinement through the real common handoff."""

from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import pytest
from apps.chat.tests.persisted_direct_test_support import close_direct_worker
from channels.db import database_sync_to_async

from apps.chat.refs import CollectionsRef
from apps.chat.services.rag_planner_contract import PLANNER_SYSTEM
from apps.chat.tests.test_rag_preservation_routes import configure, fake_search
from apps.chat.tests.test_rag_source_document_tools import docs as docs
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage

pytestmark = [pytest.mark.asyncio, pytest.mark.django_db(transaction=True), pytest.mark.usefixtures("close_direct_worker")]


@pytest.mark.parametrize("omit", ["source", "tail"])
async def test_real_final_selection_loses_lookup_support_without_replanning(
    monkeypatch, omit
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
    from apps.chat.tests.persisted_direct_test_support import persisted_direct_case
    from apps.chat.tests.chat_message_test_support import _FakeLLMInterface
    from lib.llm.types.response import LLMResponse
    from hashlib import sha256
    import json
    text = "Background. The release codename is Borealis."
    case = await persisted_direct_case([
        dict(doc_id="handoff", chunk_id=1, title="Handoff source", text=text),
        dict(doc_id="handoff", chunk_id=2, title="Handoff source", text="Other context."),
    ])
    user, doc = case.user, case.documents["handoff"]
    chunks = tuple(case.chunks.values())
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

    assessed = json.dumps({"requested_aspects": ["release codename"],
        "support": [{"aspect": "release codename", "chunk_id": chunks[0].pk,
            "source_fingerprint": sha256(text.encode()).hexdigest(), "start": 12, "end": len(text)}],
        "unresolved_aspects": [], "next_action": None})
    answer = f"The delivered evidence does not establish the codename [doc:{doc.id} chunk:{chunks[1].pk}]."
    llm = _FakeLLMInterface([LLMResponse(text=value, tool_call=None, stop_reason="end_turn",
        input_usage=1, output_usage=1) for value in (assessed, answer)])
    provider = llm.get_message
    planners = []
    async def get_message(*args, **kwargs):
        if kwargs["system"] == PLANNER_SYSTEM:
            planners.append(kwargs)
        else:
            requests.append(kwargs["messages_pydantic"])
        return await provider(*args, **kwargs)
    llm.get_message = get_message

    convo = Conversation(
        system="sys", messages=[UserMessage(content="What is the release codename?")]
    )
    consumer = SimpleNamespace(
        user=user, col_ref=CollectionsRef([doc.collection_id]), convo=convo
    )
    async with preservation_turn(consumer, max_func_calls=3) as runtime:
        await rag_pipeline.run_direct_rag_turn(
            consumer, llm, convo
        )
        acquired = runtime.observation["acquisition"]
        assert acquired.assessment.support
        packet = runtime.observation["delivered_packet"]
        assert packet.coverage_assessment.unresolved_aspects == ("release codename",)
        assert not packet.coverage_assessment.support
        assert runtime.budget.planner_calls == len(planners) == 1
    assert len(selects) == len(requests) == 1
    assert len(llm.calls) == 2  # One original assessment, one synthesis, no replan.
    assert "Borealis" not in requests[0][-1].content
    assert "partial" in requests[0][-1].content


async def test_normal_handoff_refinement_obeys_strict_tool_deadline(docs, monkeypatch):
    import asyncio

    from apps.chat.services import rag_preservation_turn
    from apps.chat.services.rag_turn import preservation_turn
    from apps.chat.services.tool_wiring import documents
    from apps.chat.services.tool_wiring.documents import vector_search_tool
    from apps.documents.services.source_loading import current_source_runtime

    configure(monkeypatch)
    monkeypatch.setenv("RAG_ITERATIVE_RETRIEVAL_ENABLED", "1")
    monkeypatch.setenv("TOOL_CALL_TIMEOUT_SECONDS", "0.05")
    fake_search(monkeypatch)
    user, doc, chunks, _ = docs
    started, release, finished = Event(), Event(), Event()
    acquisition_finished = asyncio.Event()
    late, requests, selections = [], [], []
    original = rag_preservation_turn.prepare_selection_turn
    original_acquire = rag_preservation_turn.acquire_evidence

    async def acquire(*args, **kwargs):
        result = await original_acquire(*args, **kwargs)
        acquisition_finished.set()
        return result

    monkeypatch.setattr(rag_preservation_turn, "acquire_evidence", acquire)

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

    def blocked_metadata(*args, **kwargs):
        runtime = current_source_runtime()
        started.set()
        release.wait()
        runtime.budget.publish(lambda: late.append(True))
        finished.set()
        return []

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
        # Block the first dependency after real action admission. Metadata and
        # authorization can exhaust the 50 ms deadline before search starts.
        monkeypatch.setattr(documents, "selected_document_metadata", blocked_metadata)
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
            # The tool deadline bounds acquisition, not the subsequent real DB
            # selection and synthesis. Both must advance without this worker.
            await asyncio.wait_for(acquisition_finished.wait(), timeout=0.6)
            assert not release.is_set() and not finished.is_set()
            await task
            assert not finished.is_set()
            assert runtime.observation["acquisition"].stop_reason == "acquisition_timeout"
            assert runtime.budget.actions_used == 2
        finally:
            release.set()
            await task
            assert await asyncio.to_thread(finished.wait, 1)
    assert late == []
    assert len(selections) == len(requests) == 1
    assert chunks[0].content in requests[0][-1].content
