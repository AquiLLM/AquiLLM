"""Actual optional preparation must leave a normal validated answer dispatch viable."""

from types import SimpleNamespace
import json

from asgiref.sync import async_to_sync
import pytest

from apps.documents.tests import test_scoped_recovery_final_preparation as flow

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("recovery,cheap,changed_request", [
    (False,False,False), (True,False,False), (True,True,False), (True,False,True),
], ids=["ordinary-control", "optional-long-recovery", "optional-cheap-wins", "changed-request-cannot-reuse-payment"])
def test_optional_recovery_preserves_normal_validated_answer_dispatch(monkeypatch, recovery, cheap, changed_request):
    from apps.chat.services.rag_synthesis import synthesize_from_evidence
    from lib.llm import evidence_guard
    from lib.llm.providers import openai_request
    from lib.llm.providers.openai import OpenAIInterface
    from lib.llm.turn_context import TurnContext, bind_turn
    from lib.llm.types.conversation import Conversation
    from lib.llm.types.messages import ToolMessage, UserMessage

    # Match the development vLLM context window. This changes no retrieval
    # work/source/pair allowance; the actual1M codepoint ledger is unchanged.
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "131072")
    monkeypatch.setenv("PROMPT_BUDGET_CONTEXT_LIMIT", "131072")
    monkeypatch.setenv("OPENAI_STREAM_RESPONSES", "0")
    validations, dispatched, outcomes = [], [], []
    validate = evidence_guard.validate_request

    def finish(case, selected, packet):
        budget = case.source.budget
        packet.source_mode = True
        packet.source_authorization = case.source.authorization
        packet.selection = selected
        assert packet.source_evidence and selected.candidates
        ordinary = {row.pk for row in case.snapshot.baseline_candidates}
        if not cheap:
            assert {row["chunk_id"] for row in packet.chunks} <= ordinary
        else:
            assert packet.chunks[0]["chunk_id"] not in ordinary
        before_answer = budget.text_used["tokenized"]
        row = max(packet.chunks, key=lambda row: len(row["text"]))
        answer = f'The source states: "{row["text"]}" {row["citation"]}'

        def observed_validation(stage):
            def checked(payload, *, output_reserve):
                before = budget.text_used["tokenized"]
                reason = None
                protection = evidence_guard.current_protection()
                assert protection.model_context == 131072
                try:
                    return validate(payload, output_reserve=output_reserve)
                except evidence_guard.ContextLimited as exc:
                    reason = str(exc)
                    raise
                finally:
                    validations.append(dict(stage=stage, before=before,
                        after=budget.text_used["tokenized"], reason=reason,
                        context=protection.model_context, output_reserve=output_reserve))
            return checked

        # Passive wrappers preserve both real checks. Only the network SDK
        # boundary returns fabricated inference; source/current-auth, request
        # shaping, tokenization, synthesis lease and publication remain real.
        monkeypatch.setattr(openai_request, "validate_request", observed_validation("prepare_request"))
        monkeypatch.setattr(evidence_guard, "validate_request", observed_validation("provider_dispatch"))

        async def create(**kwargs):
            assert row["citation"] in repr(kwargs["messages"])
            assert evidence_guard.contains_selected_evidence(kwargs["messages"])
            delivered = json.loads(evidence_guard.current_protection().payload)["result"]
            assert [(item["chunk_id"], item["text"]) for item in delivered] == [
                (item["chunk_id"], item["text"]) for item in packet.chunks]
            assert budget._synthesis is not None and budget._synthesis.calls == 1
            dispatched.append(dict(arguments=kwargs, tokenized=budget.text_used["tokenized"]))
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop",
                message=SimpleNamespace(content=answer, tool_calls=None))], usage=None)

        llm, conversation = case.answer.llm, case.answer.conversation
        llm.client.chat.completions.create = create
        if changed_request:
            conversation.system += " This is a different request after the exact original payment."
        turn = TurnContext(budget)
        turn._media_source_runtime = case.source
        with bind_turn(turn):
            result = async_to_sync(synthesize_from_evidence)(
                llm, conversation, packet, max_tokens=512)
        outcomes.append(dict(before_answer=before_answer, after_answer=budget.text_used["tokenized"],
            response=result[-1].content, validations=list(validations)))
        if changed_request:
            assert not dispatched and validations[0]["reason"] == "request_tokenization_limit", outcomes
            return
        assert len(dispatched) == 1, outcomes
        assert [item["stage"] for item in validations] == ["prepare_request", "provider_dispatch"]
        assert all(item["reason"] is None and item["after"] >= item["before"] for item in validations)
        assert case.answer.prepared.prepaid_selection is selected
        assert dispatched[0]["tokenized"] == before_answer, "exact paid request must reuse its tokenization"
        if not recovery:
            assert case.answer.prepared.fallback_reason != "optional_recovery_pruned"
        assert answer in result[-1].content
        assert budget.text_used["tokenized"] <= budget.limits.tokenized_codepoints == 1_000_000
        assert budget.pairs_used["final"] == (45 if cheap else 25)

    flow.test_optional_scoped_recovery_preserves_ordinary_final_preparation(
        monkeypatch, recovery, cheap, finish=finish)
    assert outcomes
