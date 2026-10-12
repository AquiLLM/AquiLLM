"""Real persisted rehydration, preserving the original scope and turn budget."""
import pytest

from apps.chat.services import rag_source_continuity
from apps.chat.services.rag_source_continuity_turn import continuity_candidates
from apps.chat.services.rag_turn import preservation_turn
from apps.chat.tests.persisted_direct_test_support import close_direct_worker as close_direct_worker
from lib.llm.turn_context import current_turn
from .auxiliary_followup_test_support import FOLLOWUPS, PAIRED_LOCAL, controls, persisted_history


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
@pytest.mark.parametrize("question,local", [(q, False) for q in FOLLOWUPS] + [(q, True) for q in PAIRED_LOCAL])
async def test_auxiliary_continuity_reads_original_persisted_coordinates_only(
    question, local, monkeypatch, settings, close_direct_worker,
):
    controls(monkeypatch, settings)
    case, history, expected = await persisted_history(question)
    original_history = history.model_dump()
    consumer = case.consumer(history, (1,))
    original_read, reads = rag_source_continuity.source_query_rows, []

    def observe_rows(*args, **kwargs):
        rows = original_read(*args, **kwargs)
        reads.append((kwargs["runtime"], tuple(
            (row.pk, str(row.doc_id), row.chunk_number, row.content) for row in rows)))
        return rows

    monkeypatch.setattr(rag_source_continuity, "source_query_rows", observe_rows)
    async with preservation_turn(consumer, max_func_calls=3) as runtime:
        assert runtime is not None and runtime.authorization is not None
        turn, frozen = current_turn(), runtime._frozen_retrieval_scope
        assert runtime is turn._media_source_runtime
        assert runtime.authorization.selected_document_ids == frozenset(
            document.id for document in case.documents.values())
        budget, deadline = turn.budget, turn.budget._deadline
        result, notice = await continuity_candidates(
            history, question, user=case.user,
            selected_scope=consumer.col_ref.collections, enabled=True)
        assert notice is None
        if local:
            assert result is None and reads == [] and budget.sources_used == 0
        else:
            assert result is not None and result["retrieved_count"] == 1
            assert result["result"] == [dict(rank=1, **expected)]
            assert reads == [(runtime, ((
                expected["chunk_id"], expected["doc_id"], expected["chunk"], expected["text"]),))]
            assert budget.sources_used == 1
        assert runtime.budget is turn.budget is budget
        assert runtime._frozen_retrieval_scope is frozen
        assert runtime.authorization is frozen.authorization
        assert budget._deadline == deadline and budget.can_publish()
        assert budget.actions_used == budget.planner_calls == 0
        assert budget.pairs_used == {"acquisition": 0, "final": 0}
    assert history.model_dump() == original_history
