"""Follow-up references preserve presented identities, then use current sources."""

from types import SimpleNamespace

import pytest

from apps.chat.services.rag_source_continuity import (
    resolve_source_anchors,
)
from apps.chat.tests.test_rag_selection_scoring import Policy, _authorization
from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


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


@pytest.mark.parametrize("compact", [False, True])
def test_mixed_explicit_and_contextual_references_follow_question_order(compact):
    history = _history("report", compact=compact)
    cited = "[doc:paper-z chunk:7]"
    assert resolve_source_anchors(
        f"Compare {cited} with the second report", history
    ).document_ids == ("paper-z", "paper-a")
    assert resolve_source_anchors(
        f"Compare the second report with {cited}", history
    ).document_ids == ("paper-a", "paper-z")
    assert resolve_source_anchors(
        f"Compare {cited} with both reports", history
    ).document_ids == ("paper-z", "paper-a")
    assert resolve_source_anchors(
        "Compare Z Report with the second report", history
    ).document_ids == ("paper-z", "paper-a")
    assert resolve_source_anchors(
        "Compare A Report with both reports", history
    ).document_ids == ("paper-a", "paper-z")
    assert resolve_source_anchors(
        "Compare both reports with A Report", history
    ).document_ids == ("paper-z", "paper-a")
    ambiguous = resolve_source_anchors(
        f"Compare {cited} with the second report",
        _history("report", compact=compact, answer=False),
    )
    assert ambiguous.unresolved_references
    assert resolve_source_anchors(
        "Compare A Report with both reports",
        _history("report", compact=compact, answer=False),
    ).unresolved_references
    assert resolve_source_anchors(
        "Compare Z Report with the second report",
        _history("report", compact=compact, answer=False),
    ).unresolved_references


@pytest.mark.parametrize("compact", [False, True])
def test_mixed_uuid_and_ordinal_keep_both_identities(compact):
    history = _history("report", compact=compact)
    ids = {
        "paper-z": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "paper-a": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
    }
    for row in history.messages[1].result_dict["result"]:
        old = row["d" if compact else "doc_id"]
        row["d" if compact else "doc_id"] = ids[old]
        row["ref" if compact else "citation"] = row[
            "ref" if compact else "citation"
        ].replace(old, ids[old])
    for old, new in ids.items():
        history.messages[2].content = history.messages[2].content.replace(old, new)
    anchors = resolve_source_anchors(
        f"Compare {ids['paper-z']} with the second report", history
    )
    assert anchors.document_ids == (ids["paper-z"], ids["paper-a"])
    assert resolve_source_anchors(
        f"Compare the second report with {ids['paper-z']}", history
    ).document_ids == (ids["paper-a"], ids["paper-z"])
    assert resolve_source_anchors(
        f"Compare {ids['paper-z']} with both reports", history
    ).document_ids == (ids["paper-z"], ids["paper-a"])


@pytest.mark.parametrize("compact", [False, True])
def test_named_document_uses_answer_cited_chunk_before_unrelated_old_row(compact):
    history = _history(compact=compact)
    stale = (
        {
            "d": "paper-z",
            "i": 9,
            "c": 1,
            "n": "Z Paper",
            "x": "old unrelated",
            "ref": "[doc:paper-z chunk:9]",
        }
        if compact
        else {
            "doc_id": "paper-z",
            "chunk_id": 9,
            "chunk": 1,
            "title": "Z Paper",
            "text": "old unrelated",
            "citation": "[doc:paper-z chunk:9]",
        }
    )
    history.messages[1].result_dict["result"].append(stale)
    assert resolve_source_anchors("Explain Z Paper", history).chunk_identities == (
        (7, "paper-z", 0),
    )
    assert resolve_source_anchors(
        "Check [doc:paper-z chunk:9]", history
    ).chunk_identities == ((9, "paper-z", 1),)


@pytest.mark.asyncio
async def test_ambiguous_title_asks_for_reference_without_guessing(monkeypatch):
    from apps.chat.refs import CollectionsRef
    from apps.chat.services import rag_pipeline

    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source")
    monkeypatch.setenv("RAG_FOLLOWUP_EVIDENCE_ENABLED", "1")
    convo = _history(titles=("Annual Report", "Annual Report")) + [
        UserMessage(content="Explain Annual Report")
    ]
    snapshot = convo.model_dump()
    runtime = SourceRuntime(TurnBudget(TurnLimits()), _authorization(Policy()))
    consumer = SimpleNamespace(
        user=runtime.authorization.reauthorization_capability._principal,
        col_ref=CollectionsRef([1]),
        convo=convo,
    )

    def unexpected_search(*_args):
        raise AssertionError("ambiguous reference reached retrieval")

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", unexpected_search)
    with source_runtime_scope(runtime):
        assert (
            await rag_pipeline.run_direct_rag_turn(consumer, SimpleNamespace(), convo)
            == "handled"
        )
    assert "title or citation" in consumer.convo[-1].content
    assert convo.model_dump() == snapshot
