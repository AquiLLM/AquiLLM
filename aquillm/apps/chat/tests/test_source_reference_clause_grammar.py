"""Source predicates and local possessors never mint title authority."""
import pytest

from apps.chat.services.rag_source_continuity import resolve_source_anchors
from apps.chat.tests.test_rag_source_continuity import _history
from apps.chat.tests.persisted_direct_test_support import close_direct_worker, persisted_direct_case
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import UserMessage


def single_history(title="Measured source"):
    history = _history(titles=(title, "Other"))
    history.messages[-1].content = "Displayed result [doc:paper-z chunk:7]."
    return history


@pytest.mark.parametrize("question,predicate", [
    ("What does this paper say?", "say?"),
    ("Does that report explain calibration?", "explain calibration?"),
    ("What did this source conclude?", "conclude?"),
])
@pytest.mark.parametrize("prior", [False, True])
def test_interrogative_source_subject_is_not_a_fresh_predicate_title(question, predicate, prior):
    history = _history(titles=(predicate, "Other")) if prior else Conversation(system="sys", messages=[])
    anchors = resolve_source_anchors(question, history)
    assert anchors.document_ids == ()
    assert anchors.unresolved_references
    assert anchors.source_constraints == ()


@pytest.mark.parametrize("predicate", ["say?", "explain calibration?", "conclude?"])
def test_explicit_quoted_title_matching_predicate_still_resolves(predicate):
    history = _history(titles=(predicate, "Other"))
    anchors = resolve_source_anchors(f'Summarize the paper "{predicate}"', history)
    assert anchors.document_ids == ("paper-z",)
    assert anchors.unresolved_references == ()


@pytest.mark.parametrize("possessor", ["Lumen unit", "star", "instrument"])
@pytest.mark.parametrize("prior", [False, True])
def test_explicit_local_nonsource_possessor_keeps_fresh_retrieval(possessor, prior):
    question = f"What is the {possessor}'s dry-mode capacity? What is its humid-mode capacity?"
    history = single_history() if prior else Conversation(system="sys", messages=[])
    anchors = resolve_source_anchors(question, history)
    assert anchors.document_ids == ()
    assert anchors.unresolved_references == ()


@pytest.mark.parametrize("possessor", ["paper", "report", "document", "source"])
def test_source_possessor_does_not_disable_genuine_historical_constraint(possessor):
    question = f"What is the {possessor}'s dry-mode result? What is its humid-mode result?"
    anchors = resolve_source_anchors(question, single_history())
    assert anchors.document_ids == ("paper-z",)
    assert "its" in anchors.source_constraints


def test_elliptical_historical_possessive_still_uses_displayed_source():
    anchors = resolve_source_anchors("What is its conclusion?", single_history())
    assert anchors.document_ids == ("paper-z",)
    assert anchors.unresolved_references == ()


@pytest.mark.parametrize("question,unresolved", [
    ("Search selected documents for controlled and field trial results; cite both sources.", False),
    ("How did the controlled and field trials differ? Cite both sources.", False),
    ("Cite both sources.", True),
    ("Compare both sources and cite both sources.", True),
])
def test_citation_output_directive_does_not_bind_an_antecedent(question, unresolved):
    anchors = resolve_source_anchors(question, Conversation(system="sys", messages=[]))
    assert anchors.document_ids == ()
    assert bool(anchors.unresolved_references) is unresolved


@pytest.mark.parametrize("question", [
    "Explain what each paper is about? What overlaps between them?",
    "Summarize each document. What differs between them?",
])
@pytest.mark.parametrize("prior", [False, True])
def test_local_quantified_sources_are_fresh_search_intent(question, prior):
    history = single_history() if prior else Conversation(system="sys", messages=[])
    anchors = resolve_source_anchors(question, history)
    assert anchors.document_ids == ()
    assert anchors.source_constraints == ()
    assert anchors.unresolved_references == ()


@pytest.mark.parametrize("question", [
    "Compare both sources.",
    "Compare that paper with the second one.",
    "Explain what each paper says. Compare that paper with the second one.",
])
def test_local_quantifier_never_suppresses_independent_definite_references(question):
    anchors = resolve_source_anchors(question, Conversation(system="sys", messages=[]))
    assert anchors.document_ids == ()
    assert anchors.unresolved_references


@pytest.mark.asyncio
@pytest.mark.parametrize("followup", [False, True])
@pytest.mark.parametrize("question,clarify", [
    ("What does this paper say?", True),
    ("Does that report explain calibration?", True),
    ("Explain what each paper is about? What overlaps between them?", False),
    ("Summarize each document. What differs between them?", False),
    ("Compare that paper with the second one.", True),
    ("Explain what each paper says. Compare that paper with the second one.", True),
    ("Search selected documents for controlled and field trial results; cite both sources.", False),
    ("Cite both sources.", True),
    ("Compare both sources and cite both sources.", True),
    ("What is the Lumen unit's dry-mode capacity? What is its humid-mode capacity?", False),
])
async def test_actual_turn_preserves_reference_or_fresh_local_grammar(
    question, clarify, followup, monkeypatch, close_direct_worker,
):
    from apps.chat.services import rag_pipeline
    from apps.chat.services.rag_action_tools import admit_tool_action
    from apps.chat.services.rag_turn import preservation_turn
    from apps.chat.tests.chat_message_test_support import _FakeLLMInterface
    from lib.llm.turn_context import current_turn

    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source" if followup else "legacy")
    monkeypatch.setenv("RAG_FOLLOWUP_EVIDENCE_ENABLED", str(int(followup)))
    case = await persisted_direct_case([dict(doc_id="grammar", chunk_id=1,
        title="Available source", text="A real source keeps original scope available.",
        citation="[doc:grammar chunk:1]")])
    convo = Conversation(system="sys", messages=[UserMessage(content=question)])
    original = convo.model_dump()
    consumer = case.consumer(convo, (1,))
    queries = []
    llm = _FakeLLMInterface([])

    def empty_search(_consumer, query, top_k):
        assert _consumer is consumer and admit_tool_action("vector", query)
        queries.append(query)
        return {"result": []}

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", empty_search)
    async with preservation_turn(consumer, max_func_calls=3) as runtime:
        budget = current_turn().budget
        if followup:
            assert runtime.authorization is not None
            assert runtime._frozen_retrieval_scope.representation is not None
        assert await rag_pipeline.run_direct_rag_turn(consumer, llm, convo) == "handled"
    assert queries == ([] if clarify else [question])
    assert budget.actions_used == int(not clarify) and budget.planner_calls == 0
    assert ("title or citation" in consumer.convo[-1].content) is clarify
    assert llm.calls == []
    assert convo.model_dump() == original
