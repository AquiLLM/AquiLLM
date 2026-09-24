"""Follow-up references preserve presented identities, then use current sources."""

import pytest

from apps.chat.services.rag_source_continuity import (
    resolve_source_anchors,
)
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage


def _history(kind="paper", *, compact=False, titles=None, answer=True):
    names = titles or (f"Z {kind.title()}", f"A {kind.title()}")
    rows = []
    for doc, chunk, name, number in (
        ("paper-z", 7, names[0], 91),
        ("paper-a", 8, names[1], 12),
    ):
        if compact:
            rows.append(
                {
                    "d": doc,
                    "i": chunk,
                    "c": 0,
                    "n": name,
                    "x": f"{number} units",
                    "ref": f"[doc:{doc} chunk:{chunk}]",
                }
            )
        else:
            rows.append(
                {
                    "doc_id": doc,
                    "chunk_id": chunk,
                    "chunk": 0,
                    "title": name,
                    "text": f"{number} units",
                    "citation": f"[doc:{doc} chunk:{chunk}]",
                }
            )
    messages = [
        UserMessage(content=f"Compare two {kind}s"),
        ToolMessage(
            tool_name="vector_search",
            for_whom="assistant",
            content="{}",
            result_dict={
                "result": list(reversed(rows)),
                "retrieved_documents": sorted(names),
            },
        ),
    ]
    if answer:
        messages.append(
            AssistantMessage(
                content=(
                    "First has 91 [doc:paper-z chunk:7], second has 12 "
                    "[doc:paper-a chunk:8]."
                ),
                stop_reason="end_turn",
            )
        )
    return Conversation(system="sys", messages=messages)


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize(
    "kind,plural,ordinal",
    [
        ("paper", "Compare their measurements", "Explain the second paper"),
        ("report", "Compare their totals", "Explain the second report"),
    ],
)
def test_answer_mention_order_drives_plural_and_ordinal(compact, kind, plural, ordinal):
    history = _history(kind, compact=compact)
    original = history.model_dump()
    assert resolve_source_anchors(plural, history).document_ids == (
        "paper-z",
        "paper-a",
    )
    assert resolve_source_anchors(ordinal, history).document_ids == ("paper-a",)
    assert resolve_source_anchors(f"both {kind}s", history).document_ids == (
        "paper-z",
        "paper-a",
    )
    assert history.model_dump() == original


def test_explicit_citation_and_unique_title_resolve_without_guessing():
    history = _history()
    assert resolve_source_anchors(
        "Check [doc:paper-z chunk:7]", history
    ).chunk_identities == ((7, "paper-z", 0),)
    assert resolve_source_anchors("Explain A Paper", history).document_ids == (
        "paper-a",
    )


def test_duplicate_title_conflicting_coordinate_and_unknown_citation_are_unresolved():
    history = _history(titles=("Annual Report", "Annual Report"))
    assert resolve_source_anchors(
        "Explain Annual Report", history
    ).unresolved_references
    history.messages[1].result_dict["result"][0]["citation"] = "[doc:paper-z chunk:7]"
    assert resolve_source_anchors(
        "Check [doc:paper-a chunk:8]", history
    ).unresolved_references
    assert resolve_source_anchors(
        "Check [doc:missing chunk:99]", history
    ).unresolved_references


def test_topic_switch_does_not_inherit_and_ordinal_without_answer_order_is_unresolved():
    assert resolve_source_anchors("What is dark matter?", _history()).document_ids == ()
    anchors = resolve_source_anchors("Explain the second paper", _history(answer=False))
    assert anchors.document_ids == ()
    assert anchors.unresolved_references
    switched = _history()
    switched.messages.extend(
        [
            UserMessage(content="What is dark matter?"),
            AssistantMessage(
                content="It is a cosmology topic.", stop_reason="end_turn"
            ),
        ]
    )
    assert (
        resolve_source_anchors("Compare their measurements", switched).document_ids
        == ()
    )


def test_conflicting_answer_presentation_order_is_unresolved():
    history = _history()
    history.messages.extend(
        [
            UserMessage(content="Compare again"),
            AssistantMessage(
                content=(
                    "A is first [doc:paper-a chunk:8], "
                    "Z is second [doc:paper-z chunk:7]."
                ),
                stop_reason="end_turn",
            ),
        ]
    )
    anchors = resolve_source_anchors("Explain the second paper", history)
    assert anchors.document_ids == ()
    assert anchors.unresolved_references
