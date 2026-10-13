"""Analogous auxiliary forms preserve exact history and local/quote boundaries."""
import pytest

from apps.chat.services.rag_source_continuity import resolve_source_anchors
from lib.llm.types.conversation import Conversation
from .auxiliary_followup_test_support import single_history

VARIANTS = (
    ("Did it support the claim?", "Did the instrument support the claim?"),
    ("Is it consistent with the result?", "Is the instrument consistent with the result?"),
    ("Was it evaluated?", "Was the instrument evaluated?"),
    ("Could it explain the result?", "Could cooling the sample explain the result?"),
    ("Will it support the claim?", "Will the instrument support the claim?"),
    ("Would it explain the result?", "Would cooling the sample explain the result?"),
)
QUOTE_PAIRS = ((chr(34), chr(34)), ("“", "”"), (chr(39), chr(39)))


@pytest.mark.parametrize("question,local", VARIANTS)
@pytest.mark.parametrize("compact", [False, True])
def test_auxiliary_variants_preserve_single_displayed_coordinates(question, local, compact):
    history = single_history(compact=compact)
    original = history.model_dump()
    anchors = resolve_source_anchors(question, history)
    assert anchors.document_ids == ("paper-z",)
    assert anchors.chunk_identities == ((7, "paper-z", 0),)
    assert anchors.basis == "answer_singular"
    assert anchors.unresolved_references == ()
    assert anchors.source_constraints == ("it",)
    assert history.model_dump() == original


@pytest.mark.parametrize("question,local", VARIANTS)
@pytest.mark.parametrize("prior", [False, True])
def test_auxiliary_variant_local_subject_does_not_acquire_history(question, local, prior):
    history = single_history() if prior else Conversation(system="sys", messages=[])
    original = history.model_dump()
    anchors = resolve_source_anchors(local + " " + question, history)
    assert anchors.document_ids == anchors.chunk_identities == ()
    assert anchors.unresolved_references == anchors.source_constraints == ()
    assert history.model_dump() == original


@pytest.mark.parametrize("left,right", QUOTE_PAIRS)
def test_quoted_auxiliary_question_does_not_create_source_reference(left, right):
    history = single_history()
    original = history.model_dump()
    question = f"The note says {left}Context. Does it support the claim?{right}"
    anchors = resolve_source_anchors(question, history)
    assert anchors.document_ids == anchors.chunk_identities == ()
    assert anchors.unresolved_references == anchors.source_constraints == ()
    assert history.model_dump() == original


@pytest.mark.parametrize("left,right", QUOTE_PAIRS)
def test_real_auxiliary_reference_outside_quoted_complement_survives(left, right):
    history = single_history()
    original = history.model_dump()
    anchors = resolve_source_anchors(f"Does it support {left}the claim{right}?", history)
    assert anchors.document_ids == ("paper-z",)
    assert anchors.chunk_identities == ((7, "paper-z", 0),)
    assert anchors.unresolved_references == ()
    assert anchors.source_constraints == ("it",)
    assert history.model_dump() == original


@pytest.mark.parametrize("left,right", QUOTE_PAIRS)
def test_quoted_local_introduction_cannot_suppress_later_source_reference(left, right):
    history = single_history()
    original = history.model_dump()
    question = f"Does the instrument support {left}the claim{right}? Does it support the claim?"
    anchors = resolve_source_anchors(question, history)
    assert anchors.document_ids == ("paper-z",)
    assert anchors.chunk_identities == ((7, "paper-z", 0),)
    assert anchors.unresolved_references == ()
    assert anchors.source_constraints == ("it",)
    assert history.model_dump() == original
