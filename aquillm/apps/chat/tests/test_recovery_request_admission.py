"""Optional answer expansion requires a supported proof and keeps paid fallback."""

from types import SimpleNamespace

import pytest

from apps.chat.services import rag_recovery_preparation as preparation
from apps.chat.services.rag_recovery_origins import take_recovery_order
from apps.chat.services.rag_recovery_request import UnsupportedRecoveryRequest, recovery_request
from apps.chat.services.rag_retrieval import fuse_ranked_tool_results
from apps.chat.services.rag_selection import select_evidence
from apps.chat.services.rag_selection_types import (
    PreparedSelection, SelectionCandidate, SelectionLimits, SelectionProfile,
)
from apps.chat.tests.test_recovery_origins import accept, frame, source


def _run(case, pool, order, preflight, monkeypatch):
    calls = []

    def prepare(**kwargs):
        rows = kwargs["pool"].rows
        calls.append(tuple(row["chunk_id"] for row in rows))
        values = tuple(SelectionCandidate(row["chunk_id"], row["doc_id"], row["chunk"],
            row["text"], .5, rank, source(row["chunk_id"]).source_fingerprint, row,
            token_cost=1, raw_score=float(row["chunk_id"]), scorer_fingerprint="fixed", scoring_kind="pointwise")
            for rank,row in enumerate(rows,1))
        return PreparedSelection(values,"model",0,0,0,None)

    scorer = SimpleNamespace(provider=object(), pair_counter=object(), scorer_fingerprint="fixed",
        scoring_kind="pointwise", budget=case.budget, rerank_state=SimpleNamespace(_failed=False))
    # These tests isolate request admission after scoring. Actual source auth,
    # paid estimates and publication are covered by test_recovery_answer_completion.
    monkeypatch.setattr(preparation,"revalidate_source_candidates",lambda rows,*a,**k: tuple(rows))
    result = preparation.prepare_recovery_selection(order, prepare=prepare,pool=pool,
        primary_query="question",authorization=case.runtime.authorization,deadline=3,
        allow_new_scores=True,scorer=scorer,chunk_loader=None,clock=lambda:case.now[0],
        turn_budget=case.budget,source_mode=True,token_ceiling=7000,source_windows=None,
        answer_preflight=preflight)
    return result,calls


def _select(prepared):
    return select_evidence(prepared.candidates,profile=SelectionProfile("test",1,0,"v1"),
        limits=SelectionLimits(1,1,10),score_status=prepared.score_status)


def test_unsupported_combined_request_keeps_exact_paid_ordinary_selection(monkeypatch):
    with frame(monkeypatch) as case:
        pool=fuse_ranked_tool_results([accept(case,(source(1),source(2)),ordinary=(source(1),))])
        order=take_recovery_order(pool,"question",case.runtime.authorization)
        paid=[]

        def preflight(prepared, *, pin):
            if not pin:
                raise UnsupportedRecoveryRequest("unsupported_recovery_request")
            paid.append(_select(prepared))
            return paid[-1]

        result,calls=_run(case,pool,order,preflight,monkeypatch)
        assert calls == [(1,),(2,)]
        assert result.prepaid_selection is paid[0]
        assert [row.chunk_id for row in result.candidates] == [1]
        assert result.score_status == "partial_model_ranked" and result.acquired_count == 2
        assert [row.chunk_id for row in result.omitted_optional] == [2]


@pytest.mark.parametrize("unsupported",["missing","provider","thinking_budget","response_format","tools"])
def test_missing_or_unsupported_request_proof_skips_optional_work(monkeypatch, unsupported):
    from lib.llm.providers.openai import OpenAIInterface
    from lib.llm.types.conversation import Conversation
    from lib.llm.types.messages import ToolMessage, UserMessage

    with frame(monkeypatch) as case:
        pool=fuse_ranked_tool_results([accept(case,(source(1),source(2)),ordinary=(source(1),))])
        order=take_recovery_order(pool,"question",case.runtime.authorization)
        preflight=None
        if unsupported != "missing":
            provider=OpenAIInterface(SimpleNamespace(base_url="http://vllm:8000/v1"),"local-test-model")
            if unsupported == "provider":
                provider=object()
            else:
                provider.base_args[unsupported] = {"type":"json_object"} if unsupported == "response_format" else 0
            conversation=Conversation(system="system",messages=[UserMessage(content="question"),
                ToolMessage(content="evidence",tool_name="vector_search",for_whom="assistant")])
            request=recovery_request(conversation,"question",provider,max_tokens=512,stage=None,
                                     initial_acquired=True,top_k=1)

            def preflight(prepared, *, pin):
                return request.pay(_select(prepared),case.runtime.authorization,pin=pin)

        result,calls=_run(case,pool,order,preflight,monkeypatch)
        assert calls == [(1,)]
        assert result.prepaid_selection is None and result.acquired_count == 2
        assert result.omitted_optional[0].reason == "answer_preflight_unavailable"
        assert case.budget.text_used["tokenized"] == 0
