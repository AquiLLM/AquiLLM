"""Auxiliary-first historical references retain identity and local barriers."""
import pytest

from apps.chat.services.rag_source_continuity import resolve_source_anchors
from apps.chat.tests.test_rag_source_continuity import _history
from lib.llm.types.conversation import Conversation
from .auxiliary_followup_test_support import (
    EXISTING_LOCAL, FOLLOWUPS, PAIRED_LOCAL, single_history,
)


@pytest.mark.parametrize("question", FOLLOWUPS)
@pytest.mark.parametrize("compact", [False, True])
def test_auxiliary_first_it_resolves_single_validated_displayed_source(question, compact):
    history = single_history(compact=compact)
    original = history.model_dump()
    anchors = resolve_source_anchors(question, history)
    assert anchors.document_ids == ("paper-z",)
    assert anchors.chunk_identities == ((7, "paper-z", 0),)
    assert anchors.basis == "answer_singular"
    assert anchors.unresolved_references == ()
    assert anchors.source_constraints == ("it",)
    assert history.model_dump() == original


@pytest.mark.parametrize("question", FOLLOWUPS)
@pytest.mark.parametrize("history_kind", ["missing", "ambiguous", "unknown_citation"])
def test_auxiliary_reference_does_not_invent_missing_or_ambiguous_identity(question, history_kind):
    history = _history() if history_kind == "ambiguous" else single_history()
    if history_kind == "missing":
        history = Conversation(system="sys", messages=[])
    elif history_kind == "unknown_citation":
        history.messages[-1].content = "Displayed result [doc:unknown chunk:99]."
    original = history.model_dump()
    anchors = resolve_source_anchors(question, history)
    assert anchors.document_ids == anchors.chunk_identities == ()
    assert anchors.unresolved_references == ("it",)
    assert anchors.source_constraints == ()
    assert history.model_dump() == original


@pytest.mark.parametrize("question", (*EXISTING_LOCAL, *PAIRED_LOCAL))
@pytest.mark.parametrize("prior", [False, True])
def test_same_question_local_antecedent_never_acquires_unrelated_prior_source(question, prior):
    history = single_history() if prior else Conversation(system="sys", messages=[])
    original = history.model_dump()
    anchors = resolve_source_anchors(question, history)
    assert anchors.document_ids == anchors.chunk_identities == ()
    assert anchors.unresolved_references == anchors.source_constraints == ()
    assert history.model_dump() == original


@pytest.mark.parametrize("reference", ['[doc:paper-z chunk:7]', '"Z Paper"', "the first paper"])
def test_later_explicit_source_barrier_survives_auxiliary_local_introduction(reference):
    question = f"Does the instrument support the claim? Review {reference}. Does it support the claim?"
    history = single_history()
    original = history.model_dump()
    anchors = resolve_source_anchors(question, history)
    assert anchors.document_ids == ("paper-z",)
    assert anchors.chunk_identities == ((7, "paper-z", 0),)
    assert anchors.unresolved_references == ()
    assert "it" in anchors.source_constraints
    assert history.model_dump() == original
