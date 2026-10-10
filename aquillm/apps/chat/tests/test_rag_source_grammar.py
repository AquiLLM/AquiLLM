"""Grammatical pronouns do not invent document antecedents."""

from types import SimpleNamespace

import pytest

from apps.chat.services.rag_source_continuity import resolve_source_anchors
from apps.chat.tests.test_rag_source_continuity import _history
from apps.chat.tests.test_rag_selection_scoring import Policy, _authorization
from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import UserMessage
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


@pytest.mark.parametrize("question", [
    "What causes stars to lose their mass?",
    "Is it true that Vega is variable?",
])
@pytest.mark.parametrize("prior", [False, True])
def test_grammatical_pronouns_do_not_acquire_source_constraints(question, prior):
    history = _history() if prior else Conversation(system="sys", messages=[])
    anchors = resolve_source_anchors(question, history)
    assert anchors.document_ids == ()
    assert anchors.unresolved_references == ()


def test_source_noun_and_ordinal_one_require_displayed_antecedents():
    question = "Compare that paper with the second one"
    anchors = resolve_source_anchors(question, _history())
    assert anchors.unresolved_references == ("that paper",)
    assert anchors.document_ids == ("paper-a",)
    missing = resolve_source_anchors(question, _history(answer=False))
    assert missing.document_ids == ()
    assert missing.unresolved_references == ("that paper", "second one")


@pytest.mark.asyncio
@pytest.mark.parametrize("question", [
    "What causes stars to lose their mass?",
    "Is it true that Vega is variable?",
])
async def test_fresh_conversation_grammar_reaches_retrieval(question, monkeypatch):
    from apps.chat.refs import CollectionsRef
    from apps.chat.services import rag_pipeline

    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source")
    monkeypatch.setenv("RAG_FOLLOWUP_EVIDENCE_ENABLED", "1")
    convo = Conversation(system="sys", messages=[UserMessage(content=question)])
    runtime = SourceRuntime(TurnBudget(TurnLimits()), _authorization(Policy()))
    consumer = SimpleNamespace(
        user=runtime.authorization.reauthorization_capability._principal,
        col_ref=CollectionsRef([1]), convo=convo,
    )
    queries = []

    def empty_search(_consumer, query, _top_k):
        queries.append(query)
        return {"result": []}

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", empty_search)
    with source_runtime_scope(runtime):
        assert await rag_pipeline.run_direct_rag_turn(
            consumer, SimpleNamespace(), convo
        ) == "handled"
    assert queries == [question]
    assert "title or citation" not in consumer.convo[-1].content


@pytest.mark.asyncio
async def test_default_mode_missing_source_antecedent_asks_focused_clarification(monkeypatch):
    from apps.chat.refs import CollectionsRef
    from apps.chat.services import rag_pipeline

    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "legacy")
    monkeypatch.setenv("RAG_FOLLOWUP_EVIDENCE_ENABLED", "0")
    convo = Conversation(system="sys", messages=[
        UserMessage(content="Compare that paper with the second one")
    ])
    runtime = SourceRuntime(TurnBudget(TurnLimits()), _authorization(Policy()))
    consumer = SimpleNamespace(user=runtime.authorization.reauthorization_capability._principal,
                               col_ref=CollectionsRef([1]), convo=convo)
    queried = []

    def search(*args):
        queried.append(args)
        return {"result": []}

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", search)
    with source_runtime_scope(runtime):
        assert await rag_pipeline.run_direct_rag_turn(consumer, SimpleNamespace(), convo) == "handled"
    assert queried == []
    assert "title or citation" in consumer.convo[-1].content
    assert runtime.budget.planner_calls == 0
