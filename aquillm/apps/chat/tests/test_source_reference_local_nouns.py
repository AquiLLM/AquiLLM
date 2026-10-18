"""Explicit local entities must not borrow unrelated historical source identity."""
import pytest

from apps.chat.services.rag_source_anchor_resolution import _reference_events
from apps.chat.services.rag_source_continuity import resolve_source_anchors
from apps.chat.tests.persisted_direct_test_support import (
    close_direct_worker as close_direct_worker,
)
from apps.chat.tests.persisted_direct_test_support import persisted_direct_case
from apps.chat.tests.test_rag_source_continuity import _history
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import UserMessage

LOCAL_QUESTIONS = (
    "In the Aurora calibration report, which criterion is used "
    "for a sensor to qualify, "
    "and how is its gain refined afterwards?",
    "In the cooling study, what happens when a sample is chilled, "
    "and what is its final size?",
    "In the field notes, what happens when an actuator was tested; "
    "how is its response measured?",
    "In the workshop report, what is required for a valve to open, "
    "AND HOW IS ITS flow checked?",
    "What limitations do the authors of Measuring River Health Across Seasons "
    "identify in their comparison, and what additional checks do they use?",
    'What concerns do the authors of "Thermal Calibration Under Load" '
    "describe in their assessment?",
    "What uncertainties do the investigators of Harbor Water Quality "
    "identify in their estimates?",
)


def history_with_count(count):
    if count == 0:
        return Conversation(system="sys", messages=[])
    history = _history(titles=(
        "Unrelated annual totals", "Unrelated installation guide"))
    if count == 1:
        history.messages[1].result_dict["result"] = [
            row for row in history.messages[1].result_dict["result"]
            if row["doc_id"] == "paper-z"
        ]
        history.messages[-1].content = "Displayed result [doc:paper-z chunk:7]."
    return history


@pytest.mark.parametrize("question", LOCAL_QUESTIONS)
@pytest.mark.parametrize("count", [0, 1, 2])
def test_local_entity_does_not_acquire_historical_identity(question, count):
    history = history_with_count(count)
    original = history.model_dump()
    anchors = resolve_source_anchors(question, history)
    assert anchors.document_ids == ()
    assert anchors.chunk_identities == ()
    assert anchors.unresolved_references == ()
    assert anchors.source_constraints == ()
    assert history.model_dump() == original


@pytest.mark.parametrize("introduction,cue", [
    ("What is required for a sensor to qualify", "its"),
    ("What do the authors of Measuring River Health identify", "their"),
])
@pytest.mark.parametrize("barrier", [
    "the second paper", "that report", "both sources", '"Known title"',
    "[doc:paper-z chunk:7]", "6d54e8f0-45a1-4c00-a050-164df5703fb1",
    "a paper", "a report", "a document", "a source",
])
def test_intervening_source_reference_preserves_historical_pronoun(
    introduction, cue, barrier,
):
    question = f"{introduction}? Review {barrier}. What is in {cue} conclusion?"
    events = _reference_events(question, {"known title"})
    assert any(event[3] == cue for event in events)


@pytest.mark.parametrize("introduction,cue", [
    ("What is required for a sensor to qualify", "its"),
    ("What do the authors of Measuring River Health identify", "their"),
])
@pytest.mark.parametrize("barrier", [
    "it", "the second paper", '"Known title"', "[doc:paper-z chunk:7]",
    "6d54e8f0-45a1-4c00-a050-164df5703fb1", "a source",
])
def test_same_clause_source_reference_is_a_local_barrier(introduction, cue, barrier):
    continuation = (
        "what is its conclusion" if cue == "its" else "what is in their conclusion")
    question = f"{introduction} after we review {barrier}, and {continuation}?"
    events = _reference_events(question, {"known title"})
    assert any(event[3] == cue for event in events)
    if barrier == "it":
        assert any(event[3] == "it" for event in events)


@pytest.mark.parametrize("question,cue", [
    ("What is required for a paper to qualify, and what is its conclusion?", "its"),
    ("What is required for a report to qualify, and what is its conclusion?", "its"),
    ("What happens when a document is selected, and what is its conclusion?", "its"),
    ("What happens when a source is selected, and what is its conclusion?", "its"),
    ("What is required for a sensor and a detector to qualify, "
     "and what is its gain?", "its"),
    ("What is required for a sensor or a detector to qualify, "
     "and what is its gain?", "its"),
    ("What do the authors of the second paper identify in their comparison?", "their"),
    ("What do the authors of that report identify in their comparison?", "their"),
    ("What do the authors identify in their comparison?", "their"),
    ('Review "a sensor to qualify". What is its conclusion?', "its"),
])
def test_ambiguous_or_source_introduction_does_not_disable_reference(question, cue):
    anchors = resolve_source_anchors(question, history_with_count(0))
    assert cue in anchors.unresolved_references


@pytest.mark.parametrize("question,cue", [
    ("What is its conclusion?", "its"),
    ("What limitations are in their comparison?", "their"),
])
@pytest.mark.parametrize("count", [0, 1, 2])
def test_standalone_historical_reference_keeps_existing_resolution(
    question, cue, count,
):
    anchors = resolve_source_anchors(question, history_with_count(count))
    if (cue == "its" and count == 1) or (cue == "their" and count == 2):
        assert len(anchors.document_ids) == count
        assert anchors.unresolved_references == ()
    else:
        assert anchors.document_ids == ()
        assert anchors.unresolved_references == (cue,)


@pytest.mark.parametrize("question", (LOCAL_QUESTIONS[0], LOCAL_QUESTIONS[4]))
def test_local_reference_does_not_erase_independent_unresolved_reference(question):
    anchors = resolve_source_anchors(
        question + " Compare the second paper.", history_with_count(0))
    assert anchors.document_ids == ()
    assert anchors.unresolved_references == ("second paper",)


@pytest.mark.asyncio
@pytest.mark.parametrize("followup", [False, True])
@pytest.mark.parametrize("count", [0, 1, 2])
@pytest.mark.parametrize("question,clarify", [
    (LOCAL_QUESTIONS[0], False), (LOCAL_QUESTIONS[4], False),
    (LOCAL_QUESTIONS[0] + " Compare the third paper.", True),
])
async def test_actual_turn_keeps_original_query_for_local_entities(
    question, clarify, count, followup, monkeypatch, close_direct_worker,
):
    from apps.chat.services import rag_pipeline
    from apps.chat.services.rag_action_tools import admit_tool_action
    from apps.chat.services.rag_turn import preservation_turn
    from apps.chat.tests.chat_message_test_support import _FakeLLMInterface
    from lib.llm.turn_context import current_turn

    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source" if followup else "legacy")
    monkeypatch.setenv("RAG_FOLLOWUP_EVIDENCE_ENABLED", str(int(followup)))
    case = await persisted_direct_case([dict(doc_id="local", chunk_id=1,
        title="Available source", text="An authorized current source is available.",
        citation="[doc:local chunk:1]")])
    convo = history_with_count(count) + [UserMessage(content=question)]
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
        assert await rag_pipeline.run_direct_rag_turn(consumer, llm, convo) == "handled"
    assert queries == ([] if clarify else [question])
    assert budget.actions_used == int(not clarify)
    assert budget.planner_calls == 0
    assert ("title or citation" in consumer.convo[-1].content) is clarify
    assert llm.calls == []
    assert convo.model_dump() == original
