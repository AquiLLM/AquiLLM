"""Contextual graph identity regression controls extracted from legacy suites."""
from dataclasses import replace

import pytest

from apps.knowledge_graph.resolution.coreference import resolve_document_mentions
from test_coreference import (
    _mention, _ontology, _cluster_ids, _decision, OTHER_DOCUMENT_ID,
)


def test_contextual_repeated_names_are_candidates():
    result = resolve_document_mentions(
        (
            _mention("a", "Mercury", "model"),
            _mention("b", "Mercury", "model", start=20),
        ),
        _ontology(),
        identity_policy="contextual_v2",
    )
    assert _cluster_ids(result) == {frozenset({"a"}), frozenset({"b"})}
    assert _decision(result, "a", "b").outcome == "candidate"
    assert result.resolver_version == "document-coreference-v4-contextual"


def test_contextual_document_resolver_preserves_single_document_boundary():
    with pytest.raises(ValueError, match="single document"):
        resolve_document_mentions(
            (
                _mention("a", "Mercury", "model"),
                _mention(
                    "b", "Mercury", "model", start=20, document_id=OTHER_DOCUMENT_ID
                ),
            ),
            _ontology(),
            identity_policy="contextual_v2",
        )


def test_contextual_shared_anchor_links_and_conflicting_anchor_rejects():
    mentions = (
        _mention("a", "Mercury", "model", identifier="doi:10.1000/a"),
        _mention("b", "Mercury", "model", start=20, identifier="doi:10.1000/a"),
    )
    linked = resolve_document_mentions(
        mentions, _ontology(), identity_policy="contextual_v2"
    )
    assert _cluster_ids(linked) == {frozenset({"a", "b"})}
    conflict = resolve_document_mentions(
        (mentions[0], replace(mentions[1], identifier="doi:10.1000/b")),
        _ontology(),
        identity_policy="contextual_v2",
    )
    assert _decision(conflict, "a", "b").outcome == "rejected"
    assert len(conflict.clusters) == 2


@pytest.mark.parametrize("identity_policy", ["legacy", "contextual_v2"])
def test_parenthetical_full_form_and_all_later_acronyms_cluster(identity_policy):
    text = (
        "Retrieval-Augmented Generation (RAG) improves grounding. "
        "RAG retrieves evidence."
    )
    full = "Retrieval-Augmented Generation"
    first_acronym = text.index("RAG")
    later_acronym = text.rindex("RAG")
    result = resolve_document_mentions(
        (
            _mention(
                "full",
                full,
                start=0,
                source_text=text,
            ),
            _mention(
                "definition",
                "RAG",
                start=first_acronym,
                source_text=text,
            ),
            _mention(
                "later",
                "RAG",
                start=later_acronym,
                source_text=text,
            ),
        ),
        _ontology(),
        identity_policy=identity_policy,
    )

    assert _cluster_ids(result) == {frozenset(("full", "definition", "later"))}
    assert result.clusters[0].label == full
    assert _decision(result, "full", "definition").method == "defined_acronym"
    assert _decision(result, "full", "later").method == "defined_acronym"
    assert _decision(result, "definition", "later").method == "normalized_name"


@pytest.mark.parametrize("identity_policy", ["legacy", "contextual_v2"])
def test_identifier_conflict_cannot_be_bridged_by_a_defined_acronym(identity_policy):
    text = (
        "Retrieval-Augmented Generation (RAG) is introduced. RAG is referenced later."
    )
    full = "Retrieval-Augmented Generation"
    acronym_positions = [
        index for index in range(len(text)) if text.startswith("RAG", index)
    ]
    result = resolve_document_mentions(
        (
            _mention(
                "identifier-x",
                full,
                start=0,
                source_text=text,
                identifier="https://github.com/example/identity-x",
            ),
            _mention(
                "definition",
                "RAG",
                start=acronym_positions[0],
                source_text=text,
            ),
            _mention(
                "identifier-y",
                "RAG",
                start=acronym_positions[1],
                source_text=text,
                identifier="https://github.com/example/identity-y",
            ),
        ),
        _ontology(),
        identity_policy=identity_policy,
    )

    assert _cluster_ids(result) == {
        frozenset(("identifier-x", "definition")),
        frozenset(("identifier-y",)),
    }
    suppressed = _decision(result, "definition", "identifier-y")
    assert suppressed.accepted is False
    assert suppressed.method == "component_conflict"
    assert "conflicting_stable_identifiers" in suppressed.explanation
