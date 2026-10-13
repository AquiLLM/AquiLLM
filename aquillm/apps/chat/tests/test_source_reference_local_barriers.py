"""Independent source references interrupt fresh local pronoun introductions."""
import pytest

from apps.chat.services.rag_source_anchor_resolution import _reference_events
from apps.chat.services.rag_source_continuity import resolve_source_anchors
from apps.chat.tests.test_source_reference_clause_grammar import single_history


@pytest.mark.parametrize("question,kind,value", [
    ('Explain what each paper says. Review "Known title". What differs between them?',
     "plural", "them"),
    ('What is the instrument\'s capacity? Review "Known title". What is its limit?',
     "singular", "its"),
    ("What is the instrument's capacity? Compare both sources. What is its limit?",
     "singular", "its"),
    ("Explain what each paper says. Compare both sources. What differs between them?",
     "plural", "them"),
])
def test_recognized_title_or_plural_blocks_local_pronoun_suppression(question, kind, value):
    events = _reference_events(question, {"known title"})
    assert any(event[2:] == (kind, value) for event in events)
    anchors = resolve_source_anchors(question, single_history("Known title"))
    if kind == "plural":
        assert value in anchors.unresolved_references
    else:
        assert value in anchors.source_constraints
        assert anchors.document_ids == ("paper-z",)


@pytest.mark.parametrize("barrier", [
    "[doc:paper-z chunk:7]",
    "6d54e8f0-45a1-4c00-a050-164df5703fb1",
    "the second paper",
])
def test_explicit_identity_barriers_remain_after_local_quantifier(barrier):
    question = f"Explain what each paper says. Review {barrier}. What differs between them?"
    events = _reference_events(question, {})
    assert any(event[2:] == ("plural", "them") for event in events)
    assert "them" in resolve_source_anchors(question, single_history()).unresolved_references
