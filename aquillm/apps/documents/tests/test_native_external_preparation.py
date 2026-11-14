"""Normal first-tool completion retains its exact initial answer obligation."""

import asyncio
from collections import Counter
from hashlib import sha256
from threading import Event
from types import SimpleNamespace

from asgiref.sync import async_to_sync
import pytest

from apps.chat.services.rag_turn import preservation_turn
from apps.documents.tests.native_external_preparation_support import external_case
from lib.llm import turn_context

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("cheap,retry", [(False, False), (True, False), (False, True)],
                         ids=["long-additions", "cheap-addition-wins", "retry-question"])
def test_normal_first_native_completion_preserves_payable_answer(monkeypatch, cheap, retry):
    with external_case(monkeypatch, cheap=cheap, retry=retry) as case:
        async def run():
            async with preservation_turn(case.consumer, max_func_calls=3):
                tool = await turn_context.call_tool_async(case.llm, case.message)
                assert len(tool.result_dict.get("result", ())) == 45, tool.result_dict
                working = case.conversation + [case.message, tool]
                return await turn_context.evidence_handoff(case.llm, working, 4096, None)

        result, status = async_to_sync(run)()
        budget = case.source.budget
        initial = {row.pk for row in case.native_initial}
        assert len(initial) == 25 and len(case.native_final) == 45
        assert len(case.dispatched) == 1, (case.failures, result[-1].content, budget.text_used)
        answer, packet = case.dispatched[0]
        assert status == "changed" and answer in result[-1].content
        assert len(case.prepared) == 1
        prepared = case.prepared[0]
        assert prepared.prepaid_selection is not None
        assert prepared.prepaid_selection == packet.selection  # Revalidation rebuilds the wrapper.
        assert prepared.acquired_count == budget.sources_used == 45
        assert prepared.fallback_reason != "optional_recovery_pruned"
        assert initial <= {row.chunk_id for row in prepared.candidates if row.raw_score is not None}
        texts = Counter(row.content for row in case.native_initial)
        assert Counter(pair[1] for pair in case.score_calls if pair[1] in texts) == texts
        assert (packet.chunks[0]["chunk_id"] not in initial) is cheap
        if cheap:
            assert len(prepared.candidates) == 45 and prepared.score_status == "model"
        else:
            assert {row["chunk_id"] for row in packet.chunks} <= initial
        missing = {row.pk for row in case.native_final} - {row.chunk_id for row in prepared.candidates}
        assert {item.chunk_id for item in prepared.omitted_optional} == missing
        assert [stage for stage, _, _ in case.validations] == ["prepare_request", "provider_dispatch"]
        assert all(before == after for _, before, after in case.validations)
        assert len(case.native_actions) == 1 and case.native_actions[0] is not None
        action = case.native_actions[0]
        assert action.accepted and not action.operation.live
        assert action.origins.query == sha256(case.question.encode()).hexdigest()
        assert len(action.origins.occurrences) == 1  # Handoff copies cannot add another vote.
        occurrence = action.origins.occurrences[0]
        assert {identity[0] for identity in occurrence.preparation} == initial
        assert len(occurrence.ordinary) == 45  # Completion is still ordinary provenance.
        assert budget.actions_used == 1 and budget.planner_calls == 0
        assert 25 <= budget.pairs_used["final"] <= budget.limits.final_pairs == 45
        assert budget.text_used["tokenized"] <= budget.limits.tokenized_codepoints == 1_000_000
        assert budget.text_used["materialized"] <= budget.limits.materialized_codepoints == 250_000
        assert budget.limits.final_scoring_ms == 3000 and budget.limits.retrieval_ms == 15000


@pytest.mark.parametrize("interruption", ["timeout", "cancel"])
def test_outer_tool_interruption_cannot_accept_finished_native_result(monkeypatch, interruption):
    with external_case(monkeypatch, cheap=True) as case:
        finished, release, unwound = Event(), Event(), Event()
        completed = []
        real_call = case.llm.call_tool

        def held_result(message):
            try:
                result = real_call(message)  # Actual inner future and native source work finish.
                completed.append(result)
                finished.set()
                assert release.wait(5), "outer worker was not released"
                return result
            finally:
                unwound.set()

        monkeypatch.setattr(case.llm, "call_tool", held_result)
        real_wait = asyncio.wait_for

        async def interrupt_after_inner(awaitable, timeout):
            task = asyncio.create_task(awaitable)
            assert await real_wait(asyncio.to_thread(finished.wait, 5), 6)
            assert len(completed[0].result_dict.get("result", ())) == 45, completed[0].result_dict
            # Exercise real outer wait_for expiry/cancellation after the inner
            # future succeeds, without changing production budget ceilings.
            if interruption == "cancel":
                asyncio.current_task().cancel()
            return await real_wait(task, .001 if interruption == "timeout" else timeout)

        monkeypatch.setattr(turn_context, "asyncio", SimpleNamespace(
            wait_for=interrupt_after_inner, to_thread=asyncio.to_thread,
            CancelledError=asyncio.CancelledError))

        async def run():
            try:
                async with preservation_turn(case.consumer, max_func_calls=3):
                    if interruption == "cancel":
                        with pytest.raises(asyncio.CancelledError):
                            await turn_context.call_tool_async(case.llm, case.message)
                    else:
                        result = await turn_context.call_tool_async(case.llm, case.message)
                        assert result.result_dict == {"result": [], "retrieval_status": "context_limited"}
            finally:
                release.set()
                assert await real_wait(asyncio.to_thread(unwound.wait, 5), 6)

        async_to_sync(run)()
        assert len(case.native_actions) == 1 and case.native_actions[0] is not None
        action = case.native_actions[0]
        assert action.preparation and not action.accepted and not action.operation.live
        assert action.origins.occurrences == []
        assert case.prepared == case.dispatched == case.validations == []
        assert case.source.budget.sources_used == 45 and case.source.budget.actions_used == 1
        assert case.source.budget.pairs_used["final"] == 0
        assert (case.source.budget.stop_reason == "cancelled") is (interruption == "cancel")
