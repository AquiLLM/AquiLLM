"""Grammatical pronouns do not invent document antecedents."""

from types import SimpleNamespace

import pytest

from apps.chat.services.rag_source_continuity import resolve_source_anchors
from apps.chat.tests.test_rag_source_continuity import _history
from apps.chat.tests.persisted_direct_test_support import (
    close_direct_worker as close_direct_worker, persisted_direct_case,
)
from apps.chat.services.rag_turn import preservation_turn
from lib.llm.turn_context import current_turn
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import UserMessage


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


@pytest.mark.parametrize("label", ["Named", "named", "Called", "called", "Titled", "titled"])
@pytest.mark.parametrize("prefix", ["the paper ", ""])
@pytest.mark.parametrize("complete", [False, True])
def test_literal_label_prefix_is_not_removed(label, prefix, complete):
    requested = f"{label} Stars"
    history = _history(titles=(requested if complete else "Stars", "Other"))
    anchors = resolve_source_anchors(f"Summarize {prefix}{requested}", history)
    assert anchors.document_ids == (("paper-z",) if complete else ())


@pytest.mark.parametrize("label", ["Named", "named", "Called", "called", "Titled", "titled"])
@pytest.mark.parametrize("prefix", ["the paper ", ""])
def test_quoted_label_requires_source_noun_introduction(label, prefix):
    anchors = resolve_source_anchors(
        f'Summarize {prefix}{label} "Stars"', _history(titles=("Stars", "Other")))
    assert anchors.document_ids == (("paper-z",) if prefix else ())


def test_join_without_prior_source_reference_cannot_supply_name_start():
    anchors = resolve_source_anchors(
        "Investigate Red with Stars", _history(titles=("Stars", "Other")))
    assert anchors.document_ids == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("followup", [False, True])
@pytest.mark.parametrize("question", [
    "What causes stars to lose their mass?",
    "Is it true that Vega is variable?",
    "Summarize the paper Attention Is All You Need",
    'Summarize the paper "Attention Is All You Need"',
    "Summarize the report Cosmic Dust",
])
async def test_fresh_conversation_grammar_reaches_retrieval(question, followup, monkeypatch, close_direct_worker):
    from apps.chat.refs import CollectionsRef
    from apps.chat.services import rag_pipeline

    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source" if followup else "legacy")
    monkeypatch.setenv("RAG_FOLLOWUP_EVIDENCE_ENABLED", "1" if followup else "0")
    convo = Conversation(system="sys", messages=[UserMessage(content=question)])
    case = await persisted_direct_case([dict(
        doc_id="grammar", chunk_id=1, title="Available source",
        text="A real source establishes the original selected scope.",
        citation="[doc:grammar chunk:1]")])
    consumer = case.consumer(convo, (1,))
    queries = []

    def empty_search(_consumer, query, _top_k):
        from apps.chat.services.rag_action_tools import admit_tool_action
        assert admit_tool_action("vector", query)
        queries.append(query)
        return {"result": []}

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", empty_search)
    async with preservation_turn(consumer, max_func_calls=3) as runtime:
        if followup:
            assert runtime.authorization is not None
            assert runtime._frozen_retrieval_scope.representation is not None
        budget = current_turn().budget
        assert await rag_pipeline.run_direct_rag_turn(
            consumer, SimpleNamespace(), convo
        ) == "handled"
    assert queries == [question]
    assert budget.actions_used == 1 and budget.planner_calls == 0
    assert "title or citation" not in consumer.convo[-1].content


@pytest.mark.asyncio
@pytest.mark.parametrize("followup", [False, True])
@pytest.mark.parametrize("question", [
    "Compare that paper with the second one",
    "Summarize the paper Red and that paper",
    "Compare the paper Red with the second one",
])
async def test_missing_source_antecedent_asks_focused_clarification(question, followup, monkeypatch, close_direct_worker):
    from apps.chat.refs import CollectionsRef
    from apps.chat.services import rag_pipeline

    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source" if followup else "legacy")
    monkeypatch.setenv("RAG_FOLLOWUP_EVIDENCE_ENABLED", str(int(followup)))
    convo = Conversation(system="sys", messages=[
        UserMessage(content=question)
    ])
    case = await persisted_direct_case([dict(
        doc_id="grammar", chunk_id=1, title="Available source",
        text="A real source establishes the original selected scope.",
        citation="[doc:grammar chunk:1]")])
    consumer = case.consumer(convo, (1,))
    queried = []

    def search(*args):
        queried.append(args)
        return {"result": []}

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", search)
    async with preservation_turn(consumer, max_func_calls=3) as runtime:
        if followup:
            assert runtime.authorization is not None
            assert runtime._frozen_retrieval_scope.representation is not None
        budget = current_turn().budget
        assert await rag_pipeline.run_direct_rag_turn(consumer, SimpleNamespace(), convo) == "handled"
    assert queried == []
    assert "title or citation" in consumer.convo[-1].content
    assert budget.planner_calls == budget.actions_used == 0


@pytest.mark.parametrize("auxiliary", ["Is", "Was"])
@pytest.mark.parametrize("prior", ["empty", "single", "ambiguous"])
def test_true_that_expletive_never_acquires_historical_source(auxiliary, prior):
    from apps.chat.tests.auxiliary_followup_test_support import single_history

    history = (Conversation(system="sys", messages=[]) if prior == "empty"
               else single_history() if prior == "single" else _history())
    original = history.model_dump()
    anchors = resolve_source_anchors(f"{auxiliary} it true that Vega is variable?", history)
    assert anchors.document_ids == anchors.chunk_identities == ()
    assert anchors.unresolved_references == anchors.source_constraints == ()
    assert history.model_dump() == original


@pytest.mark.parametrize("placement", ["within", "later"])
@pytest.mark.parametrize("known", [False, True])
def test_expletive_keeps_explicit_citation_constraints(placement, known):
    from apps.chat.tests.auxiliary_followup_test_support import single_history

    citation = "[doc:paper-z chunk:7]" if known else "[doc:unknown chunk:99]"
    question = (f"Is it true that {citation} supports the claim?" if placement == "within"
                else f"Is it true that Vega is variable? Review {citation}.")
    history = single_history()
    original = history.model_dump()
    anchors = resolve_source_anchors(question, history)
    assert anchors.document_ids == (("paper-z",) if known else ())
    assert anchors.chunk_identities == (((7, "paper-z", 0),) if known else ())
    assert anchors.unresolved_references == (() if known else (citation,))
    assert anchors.source_constraints == ((citation,) if known else ())
    assert history.model_dump() == original


@pytest.mark.parametrize("question", ["Is it true?", "Was it true?"])
def test_copular_reference_without_complement_retains_single_source(question):
    from apps.chat.tests.auxiliary_followup_test_support import single_history

    history = single_history()
    original = history.model_dump()
    anchors = resolve_source_anchors(question, history)
    assert anchors.document_ids == ("paper-z",)
    assert anchors.chunk_identities == ((7, "paper-z", 0),)
    assert anchors.unresolved_references == ()
    assert anchors.source_constraints == ("it",)
    assert history.model_dump() == original
