"""Owned-DB adapter for case checking through real direct-turn publication."""
from apps.chat.evals import run_rag_eval
from apps.chat.tests.chat_message_test_support import _FakeLLMInterface
from apps.chat.tests.direct_rag_test_support import _user_convo, _admit_query
from apps.chat.tests.persisted_direct_test_support import persisted_direct_case
from lib.llm.types.response import LLMResponse

TEXT = "Calibration and spectral analysis use flat fields and dark frames."


async def run_persisted_case(case, *, verbose=False):
    observed = {}
    async def factory(spec):
        labels = tuple(spec.get("collection_ids") or ())
        declared = [dict(doc_id="eval", chunk_id=1, title="Eval Paper", text=TEXT,
                         citation="[doc:eval chunk:1]")]
        fixture = await persisted_direct_case(declared, collection_labels=labels or (1,))
        consumer = fixture.consumer(_user_convo(spec["message"]), labels)
        answer = fixture.text(TEXT + " [doc:eval chunk:1]")
        llm = _FakeLLMInterface([LLMResponse(text=answer, tool_call=None,
            stop_reason="end_turn", input_usage=1, output_usage=1)])
        empty = spec.get("mock_retrieval_status") == "no_results"
        count = int(spec.get("mock_retrieved_count", 1))
        rows = [] if empty else [dict(declared[0]) for _ in range(max(1, count))]
        raw = fixture.rows(dict(result=rows, retrieval_status="no_results" if empty else "results_found",
            retrieved_count=len(rows), retrieved_documents=[] if empty else ["Eval Paper"]))
        def search(_consumer, query, top_k):
            assert _consumer is consumer
            _admit_query(query)
            return raw
        observed.update(consumer=consumer, llm=llm, empty=empty)
        return consumer, llm, search

    result = await run_rag_eval._run_case(case, case_factory=factory, verbose=verbose)
    calls = observed["llm"].calls
    expected_sdk = bool(case.get("expect_retrieval_called")) and not observed["empty"]
    assert len(calls) == int(expected_sdk)
    if expected_sdk:
        assert observed["consumer"].convo[-1]._validated_answer is not None
    return result
