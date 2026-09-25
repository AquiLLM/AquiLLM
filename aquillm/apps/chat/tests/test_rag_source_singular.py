"""Singular follow-up cues bind to the nearest unambiguous explicit source."""

import pytest

from apps.chat.services.rag_source_continuity import resolve_source_anchors
from apps.chat.tests.test_rag_source_continuity import _history


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("reference", ["citation", "title", "uuid"])
def test_singular_cue_binds_to_preceding_explicit_source(compact, reference):
    history = _history(compact=compact)
    history.messages[2].content = "B result [doc:paper-a chunk:8]."
    source = {
        "citation": "[doc:paper-z chunk:7]",
        "title": "Z Paper",
    }.get(reference)
    expected_doc = "paper-z"
    if reference == "uuid":
        expected_doc = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
        for row in history.messages[1].result_dict["result"]:
            key = "d" if compact else "doc_id"
            citation_key = "ref" if compact else "citation"
            if row[key] == "paper-z":
                row[key] = expected_doc
                row[citation_key] = row[citation_key].replace("paper-z", expected_doc)
        source = expected_doc
    anchors = resolve_source_anchors(f"Explain {source} and its methods", history)
    assert anchors.document_ids == (expected_doc,)
    assert anchors.unresolved_references == ()


@pytest.mark.parametrize("compact", [False, True])
def test_singular_cue_after_two_explicit_sources_is_ambiguous(compact):
    history = _history(compact=compact)
    history.messages[2].content = "B result [doc:paper-a chunk:8]."
    anchors = resolve_source_anchors(
        "Compare Z Paper and A Paper, then explain its methods", history
    )
    assert anchors.document_ids == ("paper-z", "paper-a")
    assert anchors.unresolved_references == ("its",)
