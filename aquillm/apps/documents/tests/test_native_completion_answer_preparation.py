"""Ordinary completion preserves payable answers without scoped source priority."""

import json
from collections import Counter
from types import SimpleNamespace

from asgiref.sync import async_to_sync
import pytest

from apps.documents.tests import test_scoped_recovery_final_preparation as flow

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("cheap,changed_request", [(False, False), (True, False), (False, True)],
                         ids=["long-additions", "cheap-addition-wins", "changed-request"])
def test_ordinary_completion_prepays_exact_answer_before_deferred_work(
        monkeypatch, cheap, changed_request):
    from apps.chat.services import rag_recovery_origins
    from apps.chat.services.rag_synthesis import synthesize_from_evidence
    from apps.documents.services.chunk_rerank_window_adapter import WindowSelectionScorer
    from lib.llm import evidence_guard
    from lib.llm.providers import openai_request
    from lib.llm.turn_context import TurnContext, bind_turn
    from lib.llm.utils import paid_request_estimates

    monkeypatch.setattr(rag_recovery_origins, "record_native_recovery",
                        lambda *a, **kw: pytest.fail("ordinary additions acquired scoped origins"))
    validations, dispatched, completed, providers, lookups = [], [], [], [], []
    validate = evidence_guard.validate_request
    initialize = WindowSelectionScorer.__init__

    def scorer(self, provider, **kwargs):
        providers.append(provider)
        initialize(self, provider, **kwargs)

    monkeypatch.setattr(WindowSelectionScorer, "__init__", scorer)

    def finish(case, selected, packet):
        budget, prepared = case.source.budget, case.answer.prepared
        initial = {row.pk for row in case.snapshot.baseline_candidates}
        assert prepared.prepaid_selection is selected
        assert prepared.acquired_count == budget.sources_used == 45
        assert prepared.fallback_reason != "optional_recovery_pruned"
        assert initial <= {c.chunk_id for c in prepared.candidates if c.raw_score is not None}
        initial_text = Counter(row.content for row in case.snapshot.baseline_candidates)
        calls = Counter(pair[1] for pair in providers[-1].seen if pair[1] in initial_text)
        assert calls == initial_text  # Each original source is scored once, even shared text.
        missing = set(case.snapshot.source_admission.references) - {c.chunk_id for c in prepared.candidates}
        assert {item.chunk_id for item in prepared.omitted_optional} == missing
        assert all(item.reason for item in prepared.omitted_optional)
        assert (packet.chunks[0]["chunk_id"] not in initial) is cheap
        if cheap:
            assert len(prepared.candidates) == 45 and prepared.score_status == "model"
        else:
            assert {row["chunk_id"] for row in packet.chunks} <= initial
        packet.source_mode, packet.source_authorization, packet.selection = True, case.source.authorization, selected
        before_answer = budget.text_used["tokenized"]
        row = max(packet.chunks, key=lambda item: len(item["text"]))
        answer = f'The source states: "{row["text"]}" {row["citation"]}'

        def observed(stage):
            def check(payload, *, output_reserve):
                before, reason = budget.text_used["tokenized"], None
                try:
                    return validate(payload, output_reserve=output_reserve)
                except evidence_guard.ContextLimited as exc:
                    reason = str(exc)
                    raise
                finally:
                    needed = len(evidence_guard._encoded_estimate_input(payload))
                    validations.append((stage, before, budget.text_used["tokenized"], reason, needed))
            return check

        monkeypatch.setattr(openai_request, "validate_request", observed("prepare_request"))
        monkeypatch.setattr(evidence_guard, "validate_request", observed("provider_dispatch"))
        lookup = paid_request_estimates.lookup_paid_estimate

        def observed_lookup(*args):
            value = lookup(*args)
            lookups.append(value)
            return value

        monkeypatch.setattr(paid_request_estimates, "lookup_paid_estimate", observed_lookup)

        async def create(**kwargs):
            assert evidence_guard.contains_selected_evidence(kwargs["messages"])
            delivered = json.loads(evidence_guard.current_protection().payload)["result"]
            assert [(item["chunk_id"], item["text"]) for item in delivered] == [
                (item["chunk_id"], item["text"]) for item in packet.chunks]
            assert budget._synthesis.calls == 1
            dispatched.append(budget.text_used["tokenized"])
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop",
                message=SimpleNamespace(content=answer, tool_calls=None))], usage=None)

        llm, conversation = case.answer.llm, case.answer.conversation
        llm.client.chat.completions.create = create  # Only external inference is replaced.
        if changed_request:
            conversation.system += " This changes the exact paid request."
        turn = TurnContext(budget)
        turn._media_source_runtime = case.source
        with bind_turn(turn):
            result = async_to_sync(synthesize_from_evidence)(llm, conversation, packet, max_tokens=512)
        if changed_request:
            assert lookups and all(value is None for value in lookups)
            if dispatched:
                assert dispatched[0] > before_answer and answer in result[-1].content
                assert any(after > before for _, before, after, _, _ in validations)
            else:
                _, before, _, reason, needed = validations[-1]
                assert reason == "request_tokenization_limit"
                assert needed > budget.limits.tokenized_codepoints - before
        else:
            assert dispatched == [before_answer]
            assert [item[0] for item in validations] == ["prepare_request", "provider_dispatch"]
            assert all(before == after and reason is None for _, before, after, reason, _ in validations)
            assert answer in result[-1].content
        assert 25 <= budget.pairs_used["final"] <= 45
        assert not cheap or budget.pairs_used["final"] == 45
        assert budget.text_used["tokenized"] <= budget.limits.tokenized_codepoints == 1_000_000
        assert budget.text_used["materialized"] <= budget.limits.materialized_codepoints == 250_000
        assert budget.limits.final_scoring_ms == 3000 and budget.limits.retrieval_ms == 15000
        completed.append(True)

    flow.test_optional_scoped_recovery_preserves_ordinary_final_preparation(
        monkeypatch, False, cheap, finish=finish)
    assert completed == [True]
