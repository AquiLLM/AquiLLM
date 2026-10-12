"""Explicit local action subjects never become historical source authority."""
import pytest

from apps.chat.services.rag_source_continuity import resolve_source_anchors
from apps.chat.tests.cross_document_acceptance_support import CASES, run_acceptance_case
from apps.chat.tests.persisted_direct_test_support import close_direct_worker
from apps.chat.tests.test_cross_document_acceptance import _probe
from apps.chat.tests.test_source_reference_clause_grammar import single_history
from lib.llm.types.conversation import Conversation


INTRO = "How did activating the Vela coating change the controlled response?"
FOLLOW = "How did it change the field response?"


@pytest.mark.parametrize("question", [
    CASES[1].question,
    INTRO + " " + FOLLOW,
    "HOW DID calibrating the adjustable blue light sensor coating layer CHANGE response? " + FOLLOW,
])
@pytest.mark.parametrize("prior", [False, True])
def test_explicit_local_action_uses_fresh_search_without_source_identity(question, prior):
    history = single_history() if prior else Conversation(system="sys", messages=[])
    anchors = resolve_source_anchors(question, history)
    assert anchors.document_ids == anchors.chunk_identities == ()
    assert anchors.source_constraints == anchors.unresolved_references == ()


@pytest.mark.parametrize("question", [
    FOLLOW,
    "How did the Vela coating change response? " + FOLLOW,
    "How did activating the Vela coating changes response? " + FOLLOW,
    "How did activating the Vela coating change-over response? " + FOLLOW,
    "How did activating the Vela coating reduce response? " + FOLLOW,
    "How did activating the Vela coating change response. " + FOLLOW,
    "How did activating the one two three four five six seven change response? " + FOLLOW,
    '"' + INTRO + '" ' + FOLLOW,
    'How did activating the "Vela coating" change response? ' + FOLLOW,
])
@pytest.mark.parametrize("prior", [False, True])
def test_missing_complete_bounded_matching_local_subject_retains_reference(question, prior):
    history = single_history() if prior else Conversation(system="sys", messages=[])
    anchors = resolve_source_anchors(question, history)
    if prior:
        assert anchors.document_ids == ("paper-z",)
        assert "it" in anchors.source_constraints
    else:
        assert anchors.document_ids == ()
        assert "it" in anchors.unresolved_references


@pytest.mark.parametrize("noun", ["paper", "REPORT", "document", "source", "papers", "Reports", "documents", "sources"])
def test_source_object_never_becomes_a_local_nonsource_antecedent(noun):
    anchors = resolve_source_anchors(
        f"How did activating the {noun} change response? {FOLLOW}", single_history())
    assert anchors.document_ids == ("paper-z",)
    assert "it" in anchors.source_constraints


@pytest.mark.parametrize("barrier", [
    'Review "Measured source".',
    "Review [doc:paper-z chunk:7].",
    "Review 11111111-1111-4111-8111-111111111111.",
    "Review the first paper.",
    "Compare both sources.",
    "Explain it.",
])
def test_independent_source_event_between_subject_and_pronoun_remains(barrier):
    anchors = resolve_source_anchors(f"{INTRO} {barrier} {FOLLOW}", single_history())
    assert "it" in (*anchors.source_constraints, *anchors.unresolved_references)


def test_source_event_inside_earlier_question_blocks_local_admission():
    question = INTRO.replace("the controlled response", "[doc:paper-z chunk:7] response")
    anchors = resolve_source_anchors(f"{question} {FOLLOW}", single_history())
    assert anchors.document_ids == ("paper-z",)
    assert "it" in anchors.source_constraints


def test_later_independent_ordinal_is_not_discarded_by_local_admission():
    anchors = resolve_source_anchors(
        f"{INTRO} {FOLLOW} Compare the second source.", Conversation(system="sys", messages=[]))
    assert "it" not in anchors.unresolved_references
    assert "second source" in anchors.unresolved_references
    assert anchors.document_ids == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("followup", [False, True])
@pytest.mark.parametrize("prior", [False, True])
async def test_original_conditioned_question_reaches_actual_synthesis_with_same_action(
    followup, prior, monkeypatch, close_direct_worker,
):
    from apps.chat.tests import direct_rag_test_support
    from lib.llm.turn_context import current_turn

    # The scripted provider declares a finite context; production arithmetic stays real.
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "8192")
    capacities = observe_context_budget(monkeypatch)
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source" if followup else "legacy")
    monkeypatch.setenv("RAG_FOLLOWUP_EVIDENCE_ENABLED", str(int(followup)))
    calls = []
    original_admit = direct_rag_test_support._admit_query

    def admit(query):
        result = original_admit(query)
        calls.append((query, current_turn().budget))
        return result

    monkeypatch.setattr(direct_rag_test_support, "_admit_query", admit)
    old = [dict(doc_id="old", chunk_id=99, title="Unrelated historical source",
                text="A different trial has no bearing on the Vela coating.",
                citation="[doc:old chunk:99]")] if prior else None
    case = CASES[1]
    llm = _probe(case)
    provider_budgets = []
    original_message = llm.get_message

    async def get_message(*args, **kwargs):
        provider_budgets.append(current_turn().budget)
        return await original_message(*args, **kwargs)

    monkeypatch.setattr(llm, "get_message", get_message)
    report = await run_acceptance_case(case, llm, prior_tool_rows=old, scripted_responses=True)
    assert report["outcome"] == "handled" and report["provider_call_count"] == 1
    assert report["evidence_coverage_passed"] and report["serialized_rows_match"]
    assert report["exact_allowlist_passed"]
    assert set(report["evidence_citations"]) == {report["citation_map"][c] for c in case.required_citations}
    assert [question for question, _ in calls] == [case.question]
    assert calls[0][1].actions_used == 1 and calls[0][1].planner_calls == 0
    assert len(provider_budgets) == 1 and provider_budgets[0] is calls[0][1]
    if prior:
        assert report["citation_map"][old[0]["citation"]] not in report["evidence_citations"]
    if followup:
        assert capacities == [(8192, 4096, 1000, 1000)]

def observe_context_budget(monkeypatch):
    """Observe the real one-call capacity calculation without changing its result."""
    from apps.chat.services import rag_selection_coordinator as coordinator
    from apps.chat.services.rag_context_budget import provider_context_capacity
    from apps.chat.services.rag_config import evidence_token_budget, synthesis_max_tokens

    observed = []
    original = coordinator.synthesis_evidence_budget

    def calculate(conversation, llm, rows, **kwargs):
        result = original(conversation, llm, rows, **kwargs)
        observed.append((provider_context_capacity(llm)[0], synthesis_max_tokens(),
                         evidence_token_budget(), result))
        return result

    monkeypatch.setattr(coordinator, "synthesis_evidence_budget", calculate)
    return observed


@pytest.mark.asyncio
@pytest.mark.parametrize("prior", [False, True])
async def test_original_conditioned_question_unknown_context_stays_limited_without_sdk(
    prior, monkeypatch, settings, close_direct_worker,
):
    from apps.chat.tests import direct_rag_test_support
    from apps.chat.services.rag_context_budget import provider_context_capacity
    from lib.llm.answer_publication import LIMITED_ANSWER
    from lib.llm.turn_context import current_turn

    for key in ("OPENAI_CONTEXT_LIMIT", "PROMPT_BUDGET_CONTEXT_LIMIT", "VLLM_MAX_MODEL_LEN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delattr(settings, "PROMPT_BUDGET_CONTEXT_LIMIT", raising=False)
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source")
    monkeypatch.setenv("RAG_FOLLOWUP_EVIDENCE_ENABLED", "1")
    capacities = observe_context_budget(monkeypatch)
    calls = []
    original_admit = direct_rag_test_support._admit_query

    def admit(query):
        result = original_admit(query)
        calls.append((query, current_turn().budget))
        return result

    monkeypatch.setattr(direct_rag_test_support, "_admit_query", admit)
    old = [dict(doc_id="old", chunk_id=99, title="Unrelated historical source",
                text="A different trial has no bearing on the Vela coating.",
                citation="[doc:old chunk:99]")] if prior else None
    case = CASES[1]
    llm = _probe(case)
    assert provider_context_capacity(llm)[0] == 0
    report = await run_acceptance_case(case, llm, prior_tool_rows=old, scripted_responses=True)
    assert capacities == [(0, 4096, 1000, 0)]
    assert report["outcome"] == "handled" and report["answer"] == LIMITED_ANSWER
    assert report["provider_call_count"] == 0 and llm.calls == []
    assert report["evidence_citations"] == []
    assert [question for question, _ in calls] == [case.question]
    assert calls[0][1].actions_used == 1 and calls[0][1].planner_calls == 0
