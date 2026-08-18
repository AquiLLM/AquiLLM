from __future__ import annotations

import os
import socket
import uuid
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import pytest
from django.conf import settings
from django.db.models import UniqueConstraint

from apps.knowledge_graph.resolution import DOCUMENT_RESOLVER_VERSION
from apps.knowledge_graph.resolution.coreference import (
    MAX_DOCUMENT_MENTIONS,
    DocumentMention,
    PairDecision,
    ResolutionResult,
    resolution_input_fingerprint,
    resolution_result_checksum,
    resolve_document_mentions,
)
from apps.knowledge_graph.resolution.persistence import (
    ResolutionPersistenceError,
    _resolution_rows_match,
    _validate_destination,
    _validate_source_snapshot,
    persist_document_resolution,
    resolution_commit_is_valid,
    source_mention_fingerprint,
)

DOCUMENT_ID = uuid.UUID("11111111-1111-4111-8111-111111111111")
OTHER_DOCUMENT_ID = uuid.UUID("22222222-2222-4222-8222-222222222222")
CONTENT_OBJECT_ID = uuid.UUID("33333333-3333-4333-8333-333333333333")
RESOLVER_VERSION = DOCUMENT_RESOLVER_VERSION
MAX_DB_INTEGER = 2**63 - 1


class _UUIDLikeObject:
    def __str__(self):
        return str(DOCUMENT_ID)


class _DocumentUUIDMasquerade(str):
    def __str__(self):
        return str(DOCUMENT_ID)


class _ContentUUIDMasquerade(str):
    def __str__(self):
        return str(CONTENT_OBJECT_ID)


class _MasqueradingString(str):
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False

    __hash__ = str.__hash__


class _ExplosiveString(str):
    def strip(self, *args, **kwargs):
        raise AssertionError("untrusted string subclass method was invoked")


class _ExplosiveTuple(tuple):
    def __iter__(self):
        raise AssertionError("untrusted tuple subclass iterator was invoked")


class _CoordinateBasisMasquerade(_MasqueradingString):
    def __hash__(self):
        return hash("document_global")


def _ontology():
    return SimpleNamespace(
        checksum="b" * 64,
        entity_types=MappingProxyType(
            {
                "method": SimpleNamespace(name="method", aliases=("approach",)),
                "model": SimpleNamespace(name="model", aliases=("architecture",)),
                "dataset": SimpleNamespace(name="dataset", aliases=("benchmark",)),
                "paper": SimpleNamespace(name="paper", aliases=("publication",)),
            }
        ),
    )


def _mention(mention_id, raw_text, entity_type="method", start=0, **overrides):
    values = {
        "mention_id": mention_id,
        "raw_text": raw_text,
        "entity_type": entity_type,
        "start": start,
        "end": start + len(raw_text),
        "source_text": "",
        "source_offset": 0,
        "identifier": "",
        "confidence": 0.9,
        "document_id": DOCUMENT_ID,
        "chunk_id": 1,
    }
    values.update(overrides)
    return DocumentMention(**values)


def _mapping_mention(**overrides):
    values = {
        "mention_id": "mapping",
        "raw_text": "Orion",
        "entity_type": "model",
        "start": 0,
        "end": 5,
        "source_text": "Orion",
        "source_offset": 0,
        "confidence": 0.9,
        "document_id": str(DOCUMENT_ID),
        "chunk_id": 1,
        "position_basis": "document_global",
        "content_object_id": None,
    }
    values.update(overrides)
    return values


def _cluster_ids(result):
    return {frozenset(cluster.mention_ids) for cluster in result.clusters}


def _decision(result, left, right):
    pair = frozenset((str(left), str(right)))
    return next(
        decision
        for decision in result.decisions
        if frozenset((decision.left_mention_id, decision.right_mention_id)) == pair
    )


def test_identical_normalized_names_cluster_with_compatible_ontology_alias_types():
    result = resolve_document_mentions(
        (
            _mention(2, "Retrieval—Augmented Generation", "method", 20),
            _mention(1, "retrieval augmented generation", "approach", 0),
        ),
        _ontology(),
    )

    assert _cluster_ids(result) == {frozenset(("1", "2"))}
    decision = _decision(result, 1, 2)
    assert decision.accepted is True
    assert decision.method == "ontology_alias"
    assert result.clusters[0].label == "retrieval augmented generation"
    assert result.clusters[0].entity_type == "method"


def test_parenthetical_full_form_and_all_later_acronyms_cluster():
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
    )

    assert _cluster_ids(result) == {frozenset(("full", "definition", "later"))}
    assert result.clusters[0].label == full
    assert _decision(result, "full", "definition").method == "defined_acronym"
    assert _decision(result, "full", "later").method == "defined_acronym"
    assert _decision(result, "definition", "later").method == "normalized_name"


def test_acronym_mentions_before_the_definition_remain_singletons():
    text = (
        "RAG was mentioned early. Retrieval-Augmented Generation (RAG) is defined. "
        "RAG is now unambiguous."
    )
    positions = [index for index in range(len(text)) if text.startswith("RAG", index)]
    full = "Retrieval-Augmented Generation"
    result = resolve_document_mentions(
        (
            _mention("early", "RAG", start=positions[0], source_text=text),
            _mention("full", full, start=text.index(full), source_text=text),
            _mention("definition", "RAG", start=positions[1], source_text=text),
            _mention("later", "RAG", start=positions[2], source_text=text),
        ),
        _ontology(),
    )

    assert _cluster_ids(result) == {
        frozenset(("early",)),
        frozenset(("full", "definition", "later")),
    }
    assert _decision(result, "early", "definition").accepted is False
    assert _decision(result, "early", "definition").method == "pre_definition_acronym"


def test_undefined_acronym_occurrences_remain_separate_from_full_form_and_each_other():
    result = resolve_document_mentions(
        (
            _mention("full", "Retrieval-Augmented Generation", start=0),
            _mention("a", "RAG", start=40),
            _mention("b", "RAG", start=80),
        ),
        _ontology(),
    )

    assert _cluster_ids(result) == {
        frozenset(("full",)),
        frozenset(("a",)),
        frozenset(("b",)),
    }
    assert _decision(result, "a", "b").accepted is False
    assert _decision(result, "a", "b").method == "undefined_acronym"
    assert _decision(result, "full", "a").method == "undefined_acronym"


def test_acronym_with_two_explicit_expansions_does_not_join_either_expansion():
    text = (
        "Retrieval-Augmented Generation (RAG). "
        "Resource Allocation Graph (RAG). RAG remains ambiguous."
    )
    full_a = "Retrieval-Augmented Generation"
    full_b = "Resource Allocation Graph"
    acronym_positions = [
        index for index in range(len(text)) if text.startswith("RAG", index)
    ]
    result = resolve_document_mentions(
        (
            _mention("full-a", full_a, start=text.index(full_a), source_text=text),
            _mention("def-a", "RAG", start=acronym_positions[0], source_text=text),
            _mention("full-b", full_b, start=text.index(full_b), source_text=text),
            _mention("def-b", "RAG", start=acronym_positions[1], source_text=text),
            _mention("later", "RAG", start=acronym_positions[2], source_text=text),
        ),
        _ontology(),
    )

    assert _cluster_ids(result) == {
        frozenset(("full-a",)),
        frozenset(("full-b",)),
        frozenset(("def-a",)),
        frozenset(("def-b",)),
        frozenset(("later",)),
    }
    assert _decision(result, "full-a", "def-a").method == "ambiguous_acronym"
    assert _decision(result, "full-b", "def-b").method == "ambiguous_acronym"


def test_exact_stable_identifier_clusters_differently_formatted_mentions():
    result = resolve_document_mentions(
        (
            _mention(
                "url",
                "Attention Is All You Need",
                "paper",
                identifier="https://doi.org/10.5555/12345678",
            ),
            _mention(
                "prefixed",
                "Transformer paper",
                "publication",
                identifier="doi:10.5555/12345678",
                start=40,
            ),
        ),
        _ontology(),
    )

    assert _cluster_ids(result) == {frozenset(("url", "prefixed"))}
    assert result.clusters[0].identifier == "doi:10.5555/12345678"
    assert _decision(result, "url", "prefixed").method == "stable_identifier"


@pytest.mark.parametrize(
    "raw_text",
    [
        "https://github.com/example/orion\n",
        "https://doi.org/10.5555/12345678\r",
        "https://arxiv.org/abs/1706.03762\t",
        "https://orcid.org/0000-0002-1825-0097\n",
    ],
)
def test_control_tainted_raw_labels_are_not_auto_linked_as_identifiers(raw_text):
    result = resolve_document_mentions(
        (_mention("tainted", raw_text, "paper"),),
        _ontology(),
    )

    assert result.clusters[0].identifier == ""


def test_exact_stable_identifier_precedes_pronoun_only_resolution_rejection():
    identifier = "https://github.com/example/orion"
    result = resolve_document_mentions(
        (
            _mention("pronoun-one", "it", "model", identifier=identifier),
            _mention(
                "named",
                "Orion",
                "model",
                start=20,
                identifier=identifier,
            ),
            _mention(
                "pronoun-two",
                "they",
                "model",
                start=40,
                identifier=identifier,
            ),
        ),
        _ontology(),
    )

    assert _cluster_ids(result) == {frozenset(("pronoun-one", "named", "pronoun-two"))}
    assert {decision.method for decision in result.decisions} == {"stable_identifier"}
    assert all(decision.accepted for decision in result.decisions)
    memberships = {
        membership.mention_id: membership
        for membership in result.clusters[0].memberships
    }
    assert memberships["named"].method == "root"
    assert memberships["named"].parent_mention_id is None
    assert memberships["pronoun-one"].method == "stable_identifier"
    assert memberships["pronoun-one"].parent_mention_id == "named"
    assert memberships["pronoun-two"].method == "stable_identifier"
    assert memberships["pronoun-two"].parent_mention_id == "pronoun-one"


def test_pronoun_only_mentions_without_an_authoritative_identifier_stay_separate():
    result = resolve_document_mentions(
        (
            _mention("one", "it", "model"),
            _mention("two", "they", "model", start=20),
        ),
        _ontology(),
    )

    assert _cluster_ids(result) == {frozenset(("one",)), frozenset(("two",))}
    assert _decision(result, "one", "two").method == "pronoun_only"


@pytest.mark.parametrize(
    ("left", "right", "reason"),
    [
        (
            _mention("left", "Orion", "model"),
            _mention("right", "Orion", "dataset", start=20),
            "incompatible_entity_types",
        ),
        (
            _mention("left", "Orion v1", "model"),
            _mention("right", "Orion v2", "model", start=20),
            "version_mismatch",
        ),
        (
            _mention("left", "it", "model"),
            _mention("right", "Orion", "model", start=20),
            "pronoun_only",
        ),
        (
            _mention("left", "this method", "method"),
            _mention("right", "this method", "method", start=20),
            "pronoun_only",
        ),
    ],
)
def test_incompatible_versions_and_pronouns_remain_separate(left, right, reason):
    result = resolve_document_mentions((left, right), _ontology())

    assert len(result.clusters) == 2
    decision = _decision(result, "left", "right")
    assert decision.accepted is False
    assert decision.method == reason


def test_conflicting_identifiers_prevent_same_name_and_unidentified_bridge_merges():
    result = resolve_document_mentions(
        (
            _mention("one", "Orion", "model", identifier="arxiv:1706.03762"),
            _mention("bridge", "Orion", "model", start=20),
            _mention(
                "two",
                "Orion",
                "model",
                start=40,
                identifier="arxiv:1706.03763",
            ),
        ),
        _ontology(),
    )

    assert len(result.clusters) == 3
    assert all(not decision.accepted for decision in result.decisions)
    assert {decision.method for decision in result.decisions} == {
        "conflicting_stable_identifiers"
    }


def test_version_signature_difference_blocks_even_an_equal_stable_identifier():
    repository = "https://github.com/example/orion"
    result = resolve_document_mentions(
        (
            _mention("base", "Orion", "model", identifier=repository),
            _mention(
                "v2",
                "Orion v2",
                "model",
                start=20,
                identifier=repository,
            ),
        ),
        _ontology(),
    )

    assert len(result.clusters) == 2
    assert _decision(result, "base", "v2").method == "version_mismatch"


def test_identifier_conflict_cannot_be_bridged_by_a_defined_acronym():
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
    )

    assert _cluster_ids(result) == {
        frozenset(("identifier-x", "definition")),
        frozenset(("identifier-y",)),
    }
    suppressed = _decision(result, "definition", "identifier-y")
    assert suppressed.accepted is False
    assert suppressed.method == "component_conflict"
    assert "conflicting_stable_identifiers" in suppressed.explanation


def test_shared_identifier_does_not_bridge_versioned_or_versionless_aliases():
    repository = "https://github.com/example/orion"
    result = resolve_document_mentions(
        (
            _mention("v1", "Orion/v1", "model", start=0, identifier=repository),
            _mention(
                "alias",
                "Project Orion",
                "model",
                start=20,
                identifier=repository,
            ),
            _mention("v2", "Orion/v2", "model", start=40, identifier=repository),
        ),
        _ontology(),
    )

    assert _cluster_ids(result) == {
        frozenset(("v1",)),
        frozenset(("alias",)),
        frozenset(("v2",)),
    }
    assert {_decision(result, "v1", "alias").method} == {"version_mismatch"}
    assert {_decision(result, "alias", "v2").method} == {"version_mismatch"}
    assert _decision(result, "v1", "v2").method == "version_mismatch"


def test_nfkc_equivalent_undefined_acronyms_cannot_form_a_name_bridge():
    result = resolve_document_mentions(
        (
            _mention("ascii-one", "RAG", start=0),
            _mention("fullwidth", "ＲＡＧ", start=20),
            _mention("ascii-two", "RAG", start=40),
        ),
        _ontology(),
    )

    assert _cluster_ids(result) == {
        frozenset(("ascii-one",)),
        frozenset(("fullwidth",)),
        frozenset(("ascii-two",)),
    }
    assert all(not decision.accepted for decision in result.decisions)
    assert {decision.method for decision in result.decisions} == {"undefined_acronym"}


def test_nfkc_acronym_definition_uses_the_same_shape_key_as_ascii():
    text = (
        "Retrieval-Augmented Generation (ＲＡＧ) is introduced. "
        "ＲＡＧ is referenced later."
    )
    full = "Retrieval-Augmented Generation"
    positions = [
        index for index in range(len(text)) if text.startswith("ＲＡＧ", index)
    ]

    result = resolve_document_mentions(
        (
            _mention("full", full, start=0, source_text=text),
            _mention("definition", "ＲＡＧ", start=positions[0], source_text=text),
            _mention("later", "ＲＡＧ", start=positions[1], source_text=text),
        ),
        _ontology(),
    )

    assert _cluster_ids(result) == {frozenset(("full", "definition", "later"))}


def test_nfkc_full_form_words_define_an_ascii_acronym():
    full = "Ｒｅｔｒｉｅｖａｌ Ａｕｇｍｅｎｔｅｄ Ｇｅｎｅｒａｔｉｏｎ"
    text = f"{full} (RAG) is introduced. RAG is referenced later."
    positions = [index for index in range(len(text)) if text.startswith("RAG", index)]

    result = resolve_document_mentions(
        (
            _mention("full", full, start=0, source_text=text),
            _mention("definition", "RAG", start=positions[0], source_text=text),
            _mention("later", "RAG", start=positions[1], source_text=text),
        ),
        _ontology(),
    )

    assert _cluster_ids(result) == {frozenset(("full", "definition", "later"))}


def test_cluster_memberships_preserve_their_actual_parent_edge_provenance():
    repository = "https://github.com/example/orion"
    result = resolve_document_mentions(
        (
            _mention(
                "identifier-label",
                "Project Orion",
                "model",
                start=0,
                identifier=repository,
            ),
            _mention(
                "identifier-name",
                "Orion",
                "model",
                start=20,
                identifier=repository,
            ),
            _mention("name-only", "Orion", "model", start=40),
        ),
        _ontology(),
    )

    assert len(result.clusters) == 1
    memberships = {
        membership.mention_id: membership
        for membership in result.clusters[0].memberships
    }
    assert memberships["identifier-label"].method == "root"
    assert memberships["identifier-label"].parent_mention_id is None
    assert memberships["identifier-name"].method == "stable_identifier"
    assert memberships["identifier-name"].parent_mention_id == "identifier-label"
    assert memberships["name-only"].method == "normalized_name"
    assert memberships["name-only"].parent_mention_id == "identifier-name"
    assert all(membership.reason for membership in memberships.values())


def test_singleton_cluster_membership_records_singleton_not_structural_root():
    result = resolve_document_mentions(
        (_mention("only", "Orion", "model"),), _ontology()
    )

    membership = result.clusters[0].memberships[0]
    assert membership.mention_id == "only"
    assert membership.method == "singleton"
    assert membership.parent_mention_id is None
    assert membership.reason == "Singleton cluster."


def test_unknown_entity_types_are_rejected_instead_of_silently_accepted():
    with pytest.raises(ValueError, match="unknown ontology entity type"):
        resolve_document_mentions(
            (_mention("unknown", "RAG", "evaluation_metric"),), _ontology()
        )


def test_acronym_definitions_do_not_propagate_to_another_source_coordinate_space():
    text = "Retrieval-Augmented Generation (RAG) is defined."
    full = "Retrieval-Augmented Generation"
    definition = text.index("RAG")
    result = resolve_document_mentions(
        (
            _mention(
                "full",
                full,
                start=0,
                source_text=text,
                source_key="document:text",
            ),
            _mention(
                "definition",
                "RAG",
                start=definition,
                source_text=text,
                source_key="document:text",
            ),
            _mention(
                "figure",
                "RAG",
                start=0,
                source_text="RAG",
                source_key="figure:one",
                position_basis="chunk_content",
                content_object_id=CONTENT_OBJECT_ID,
            ),
        ),
        _ontology(),
    )

    assert _cluster_ids(result) == {
        frozenset(("full", "definition")),
        frozenset(("figure",)),
    }
    assert _decision(result, "definition", "figure").method == "source_mismatch"
    assert _decision(result, "full", "figure").method == "source_mismatch"


def test_document_global_acronym_definition_applies_across_text_chunks():
    definition_text = "Retrieval-Augmented Generation (RAG) is defined."
    full_text = "Retrieval-Augmented Generation"
    definition_offset = 100
    acronym_start = definition_offset + definition_text.index("RAG")
    later_text = "Later, RAG is evaluated."
    later_offset = 500
    later_start = later_offset + later_text.index("RAG")
    result = resolve_document_mentions(
        (
            _mention(
                "full",
                full_text,
                start=definition_offset,
                source_text=definition_text,
                source_offset=definition_offset,
                source_key="text-chunk:definition",
                chunk_id=1,
            ),
            _mention(
                "definition",
                "RAG",
                start=acronym_start,
                source_text=definition_text,
                source_offset=definition_offset,
                source_key="text-chunk:definition",
                chunk_id=1,
            ),
            _mention(
                "later",
                "RAG",
                start=later_start,
                source_text=later_text,
                source_offset=later_offset,
                source_key="text-chunk:later",
                chunk_id=2,
            ),
        ),
        _ontology(),
    )

    assert _cluster_ids(result) == {frozenset(("full", "definition", "later"))}
    assert _decision(result, "full", "later").accepted is True
    assert _decision(result, "full", "later").method == "defined_acronym"


def test_document_global_definition_can_use_overlapping_context_windows():
    document_text = "Retrieval-Augmented Generation (RAG) is defined."
    full_text = "Retrieval-Augmented Generation"
    document_offset = 100
    acronym_start = document_offset + document_text.index("RAG")
    acronym_window_start = document_text.index("Generation")
    result = resolve_document_mentions(
        (
            _mention(
                "full",
                full_text,
                start=document_offset,
                source_text=document_text,
                source_offset=document_offset,
                source_key="window:full",
                chunk_id=1,
            ),
            _mention(
                "definition",
                "RAG",
                start=acronym_start,
                source_text=document_text[acronym_window_start:],
                source_offset=document_offset + acronym_window_start,
                source_key="window:acronym",
                chunk_id=2,
            ),
        ),
        _ontology(),
    )

    assert _cluster_ids(result) == {frozenset(("full", "definition"))}
    assert _decision(result, "full", "definition").method == "defined_acronym"


def test_chunk_content_acronym_definition_stays_with_its_content_object():
    definition_text = "Retrieval-Augmented Generation (RAG) is defined."
    full_text = "Retrieval-Augmented Generation"
    acronym_start = definition_text.index("RAG")
    result = resolve_document_mentions(
        (
            _mention(
                "full",
                full_text,
                start=0,
                source_text=definition_text,
                source_key="figure:first",
                position_basis="chunk_content",
                content_object_id=CONTENT_OBJECT_ID,
            ),
            _mention(
                "definition",
                "RAG",
                start=acronym_start,
                source_text=definition_text,
                source_key="figure:first",
                position_basis="chunk_content",
                content_object_id=CONTENT_OBJECT_ID,
            ),
            _mention(
                "other-figure",
                "RAG",
                start=0,
                source_text="RAG",
                source_key="figure:second",
                position_basis="chunk_content",
                content_object_id=OTHER_DOCUMENT_ID,
            ),
        ),
        _ontology(),
    )

    assert _cluster_ids(result) == {
        frozenset(("full", "definition")),
        frozenset(("other-figure",)),
    }
    assert _decision(result, "definition", "other-figure").method == "source_mismatch"


def test_chunk_content_acronym_ambiguity_is_scoped_to_each_content_object():
    first_text = "Retrieval-Augmented Generation (RAG). Later RAG."
    second_text = "Red Amber Green (RAG). Later RAG."
    first_full = "Retrieval-Augmented Generation"
    second_full = "Red Amber Green"
    first_positions = [
        index for index in range(len(first_text)) if first_text.startswith("RAG", index)
    ]
    second_positions = [
        index
        for index in range(len(second_text))
        if second_text.startswith("RAG", index)
    ]
    common = {"position_basis": "chunk_content", "source_offset": 0}
    result = resolve_document_mentions(
        (
            _mention(
                "first-full",
                first_full,
                start=0,
                source_text=first_text,
                source_key="figure:first",
                content_object_id=CONTENT_OBJECT_ID,
                **common,
            ),
            _mention(
                "first-definition",
                "RAG",
                start=first_positions[0],
                source_text=first_text,
                source_key="figure:first",
                content_object_id=CONTENT_OBJECT_ID,
                **common,
            ),
            _mention(
                "first-later",
                "RAG",
                start=first_positions[1],
                source_text=first_text,
                source_key="figure:first",
                content_object_id=CONTENT_OBJECT_ID,
                **common,
            ),
            _mention(
                "second-full",
                second_full,
                start=0,
                source_text=second_text,
                source_key="figure:second",
                content_object_id=OTHER_DOCUMENT_ID,
                **common,
            ),
            _mention(
                "second-definition",
                "RAG",
                start=second_positions[0],
                source_text=second_text,
                source_key="figure:second",
                content_object_id=OTHER_DOCUMENT_ID,
                **common,
            ),
            _mention(
                "second-later",
                "RAG",
                start=second_positions[1],
                source_text=second_text,
                source_key="figure:second",
                content_object_id=OTHER_DOCUMENT_ID,
                **common,
            ),
        ),
        _ontology(),
    )

    assert _cluster_ids(result) == {
        frozenset(("first-full", "first-definition", "first-later")),
        frozenset(("second-full", "second-definition", "second-later")),
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"position_basis": "document_global", "content_object_id": CONTENT_OBJECT_ID},
        {"position_basis": "chunk_content", "content_object_id": None},
        {"position_basis": "chunk_content", "content_object_id": ""},
    ],
)
def test_document_mentions_require_coordinate_basis_provenance_pairing(overrides):
    with pytest.raises(ValueError, match="content_object_id|coordinate basis"):
        _mention("bad-basis", "Orion", "model", **overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"position_basis": "document_global", "content_object_id": CONTENT_OBJECT_ID},
        {"position_basis": "chunk_content", "content_object_id": None},
    ],
)
def test_mapping_inputs_cannot_bypass_coordinate_basis_pairing(overrides):
    with pytest.raises(ValueError, match="content_object_id|coordinate basis"):
        resolution_input_fingerprint((_mapping_mention(**overrides),))


@pytest.mark.parametrize("position_basis", [None, False, 0, [], {}])
def test_coordinate_basis_rejects_non_string_scalars(position_basis):
    with pytest.raises(ValueError, match="position_basis"):
        resolution_input_fingerprint((_mapping_mention(position_basis=position_basis),))


def test_chunk_content_cluster_identity_never_depends_on_chunk_database_pk():
    common = {
        "position_basis": "chunk_content",
        "content_object_id": CONTENT_OBJECT_ID,
        "source_text": "Orion",
    }
    first = resolve_document_mentions(
        (_mention("mention", "Orion", "model", chunk_id=1, **common),),
        _ontology(),
    )
    reloaded = resolve_document_mentions(
        (_mention("mention", "Orion", "model", chunk_id=999, **common),),
        _ontology(),
    )

    assert first.clusters[0].cluster_key == reloaded.clusters[0].cluster_key
    assert first.input_fingerprint != reloaded.input_fingerprint


def test_document_global_cluster_identity_ignores_representative_context_window():
    first = resolve_document_mentions(
        (
            _mention(
                "mention",
                "Orion",
                "model",
                start=105,
                source_text="xxxxxOrion",
                source_offset=100,
                source_key="text-chunk:first",
                chunk_id=1,
            ),
        ),
        _ontology(),
    )
    reloaded = resolve_document_mentions(
        (
            _mention(
                "mention",
                "Orion",
                "model",
                start=105,
                source_text="............Orion",
                source_offset=93,
                source_key="text-chunk:replacement",
                chunk_id=999,
            ),
        ),
        _ontology(),
    )

    assert first.clusters[0].cluster_key == reloaded.clusters[0].cluster_key
    assert first.input_fingerprint != reloaded.input_fingerprint


def test_result_is_deterministic_immutable_and_audits_every_pair():
    mentions = (
        _mention("c", "Orion v2", "model", start=40),
        _mention("a", "Orion v1", "model", start=0),
        _mention("b", "Orion v1", "architecture", start=20),
    )

    forward = resolve_document_mentions(mentions, _ontology())
    reverse = resolve_document_mentions(tuple(reversed(mentions)), _ontology())

    assert forward == reverse
    assert isinstance(forward.clusters, tuple)
    assert isinstance(forward.decisions, tuple)
    assert forward.resolver_version == DOCUMENT_RESOLVER_VERSION
    assert len(forward.checksum) == 64
    assert all(len(cluster.cluster_key) == 64 for cluster in forward.clusters)
    assert len(forward.decisions) == 3
    assert all(decision.explanation for decision in forward.decisions)
    with pytest.raises((AttributeError, TypeError)):
        forward.clusters += ()


def test_cluster_identity_uses_source_coordinates_not_database_ids_or_confidence():
    first = resolve_document_mentions(
        (
            _mention(
                10,
                "Orion",
                "model",
                start=5,
                confidence=0.51,
                document_id=DOCUMENT_ID,
            ),
            _mention(
                11,
                "Orion",
                "model",
                start=30,
                confidence=0.61,
                document_id=DOCUMENT_ID,
            ),
        ),
        _ontology(),
    )
    reloaded = resolve_document_mentions(
        (
            _mention(
                900,
                "Orion",
                "model",
                start=5,
                confidence=0.91,
                document_id=DOCUMENT_ID,
            ),
            _mention(
                901,
                "Orion",
                "model",
                start=30,
                confidence=0.99,
                document_id=DOCUMENT_ID,
            ),
        ),
        _ontology(),
    )

    assert first.ontology_checksum == reloaded.ontology_checksum
    assert first.clusters[0].cluster_key == reloaded.clusters[0].cluster_key
    assert len(first.input_fingerprint) == 64
    assert first.input_fingerprint != reloaded.input_fingerprint


def test_document_resolver_rejects_mentions_from_mixed_document_uuids():
    with pytest.raises(ValueError, match="single document"):
        resolve_document_mentions(
            (
                _mention("first", "Orion", "model", document_id=DOCUMENT_ID),
                _mention(
                    "second",
                    "Orion",
                    "model",
                    start=20,
                    document_id=OTHER_DOCUMENT_ID,
                ),
            ),
            _ontology(),
        )


def test_resolution_input_fingerprint_binds_source_context():
    original = resolve_document_mentions(
        (
            _mention(
                "one",
                "Orion",
                "model",
                source_text="Orion is evaluated.",
                document_id=DOCUMENT_ID,
            ),
        ),
        _ontology(),
    )
    changed = resolve_document_mentions(
        (
            _mention(
                "one",
                "Orion",
                "model",
                source_text="Orion was changed.",
                document_id=DOCUMENT_ID,
            ),
        ),
        _ontology(),
    )

    assert original.input_fingerprint != changed.input_fingerprint


def test_resolution_input_fingerprint_hashes_repeated_source_context_once(monkeypatch):
    import apps.knowledge_graph.resolution.coreference as coreference

    source_text = "Repeated source context. " * 128
    mentions = tuple(
        _mention(
            f"mention-{index}",
            f"Entity {index}",
            "model",
            start=index * 20,
            source_text=source_text,
        )
        for index in range(MAX_DOCUMENT_MENTIONS)
    )
    original_dumps = coreference.json.dumps
    original_context_digest = coreference._source_context_digest
    occurrences: list[int] = []
    digest_calls = 0

    def recording_dumps(value, *args, **kwargs):
        encoded = original_dumps(value, *args, **kwargs)
        occurrences.append(encoded.count(source_text))
        return encoded

    def recording_context_digest(**kwargs):
        nonlocal digest_calls
        digest_calls += 1
        return original_context_digest(**kwargs)

    monkeypatch.setattr(coreference.json, "dumps", recording_dumps)
    monkeypatch.setattr(
        coreference,
        "_source_context_digest",
        recording_context_digest,
    )

    fingerprint = resolution_input_fingerprint(mentions)

    assert len(fingerprint) == 64
    assert max(occurrences, default=0) <= 1
    assert digest_calls == 1


def test_resolution_input_fingerprint_rejects_excess_unique_source_context():
    mentions = tuple(
        _mention(
            f"mention-{index}",
            f"Entity {index}",
            "model",
            start=index * 20,
            source_text=(chr(ord("a") + index) * 700_001),
            source_key=f"unique-context:{index}",
        )
        for index in range(3)
    )

    with pytest.raises(ValueError, match="aggregate.*source context"):
        resolution_input_fingerprint(mentions)


def test_same_source_key_and_coordinate_rejects_mismatched_source_text():
    first = _mapping_mention(
        mention_id="first",
        source_key="shared-source",
        source_text="Orion first context.",
    )
    second = _mapping_mention(
        mention_id="second",
        start=20,
        end=25,
        source_key="shared-source",
        source_text="Orion changed context.",
    )

    with pytest.raises(
        ValueError, match="source key.*mismatched|source context.*mismatch"
    ):
        resolution_input_fingerprint((first, second))


def test_repeated_source_context_cannot_bypass_string_validation_with_equality():
    class PretendsToBeCachedText:
        def __eq__(self, other):
            return other == "Orion"

    first = _mapping_mention(
        mention_id="first",
        source_key="shared-source",
        source_text="Orion",
    )
    second = _mapping_mention(
        mention_id="second",
        source_key="shared-source",
        source_text=PretendsToBeCachedText(),
    )

    with pytest.raises(ValueError, match="source text.*string"):
        resolution_input_fingerprint((first, second))


def test_resolver_validates_and_hashes_one_shared_large_context_once(monkeypatch):
    import apps.knowledge_graph.resolution.coreference as coreference

    source_text = "x" * 1_000_000
    mentions = tuple(
        _mapping_mention(
            mention_id=f"mention-{index}",
            raw_text=f"Entity {index}",
            start=index * 10,
            end=index * 10 + len(f"Entity {index}"),
            source_text=source_text,
            source_key="shared-large-context",
        )
        for index in range(MAX_DOCUMENT_MENTIONS)
    )
    original_validate = coreference._validated_source_text
    original_digest = coreference._source_context_digest
    validation_calls = 0
    digest_calls = 0

    def recording_validate(value):
        nonlocal validation_calls
        validation_calls += 1
        return original_validate(value) if validation_calls == 1 else value

    def recording_digest(**kwargs):
        nonlocal digest_calls
        digest_calls += 1
        return original_digest(**kwargs)

    monkeypatch.setattr(coreference, "_validated_source_text", recording_validate)
    monkeypatch.setattr(coreference, "_source_context_digest", recording_digest)

    result = resolve_document_mentions(mentions, _ontology())

    assert len(result.mention_ids) == MAX_DOCUMENT_MENTIONS
    assert validation_calls == 1
    assert digest_calls == 1


def test_invalid_confidence_duplicate_ids_and_unbounded_documents_are_rejected():
    with pytest.raises(ValueError, match="finite confidence"):
        resolve_document_mentions(
            (_mention("bad", "Orion", confidence=float("nan")),), _ontology()
        )
    with pytest.raises(ValueError, match="unique"):
        resolve_document_mentions(
            (_mention("same", "Orion"), _mention("same", "MMLU", start=20)),
            _ontology(),
        )
    over_cap = tuple(
        _mention(index, f"entity {index}", start=index * 20)
        for index in range(MAX_DOCUMENT_MENTIONS + 1)
    )
    with pytest.raises(ValueError, match="mention cap"):
        resolve_document_mentions(over_cap, _ontology())


def test_adversarial_scalars_are_rejected_as_bounded_validation_errors():
    with pytest.raises(ValueError, match="finite confidence"):
        _mention("huge-confidence", "Orion", confidence=10**10_000)
    with pytest.raises(ValueError, match="source text.*limit"):
        _mention("huge-source", "Orion", source_text="x" * 1_000_001)
    with pytest.raises(ValueError, match="identifier.*limit"):
        _mention("huge-identifier", "Orion", identifier="x" * 2_049)
    with pytest.raises(ValueError, match="source text.*control"):
        _mention("nul-source", "Orion", source_text="Orion\x00")
    with pytest.raises(ValueError, match="source key.*limit"):
        _mention("huge-source-key", "Orion", source_key="x" * 513)
    with pytest.raises(ValueError, match="mention_id.*limit"):
        _mention("x" * 129, "Orion")
    with pytest.raises(ValueError, match="entity_type.*128"):
        _mention("huge-type", "Orion", entity_type="x" * 129)


def test_mapping_inputs_cannot_bypass_source_bounds():
    mention = {
        "mention_id": "mapping",
        "raw_text": "Orion",
        "entity_type": "model",
        "start": 0,
        "end": 5,
        "source_text": "x" * 1_000_001,
        "source_offset": 0,
        "confidence": 0.9,
        "document_id": str(DOCUMENT_ID),
        "chunk_id": 1,
    }

    with pytest.raises(ValueError, match="source text.*limit"):
        resolve_document_mentions((mention,), _ontology())

    invalid_empty_source = {
        **mention,
        "source_text": "",
        "source_offset": -1,
    }
    with pytest.raises(ValueError, match="source_offset"):
        resolve_document_mentions((invalid_empty_source,), _ontology())

    oversized_label = {
        **mention,
        "raw_text": "x" * 4_097,
        "end": 4_097,
        "source_text": "",
    }
    with pytest.raises(ValueError, match="entity label.*limit"):
        resolution_input_fingerprint((oversized_label,))


@pytest.mark.parametrize(
    ("overrides", "field_name"),
    [
        ({"chunk_id": MAX_DB_INTEGER + 1}, "chunk_id"),
        ({"start": MAX_DB_INTEGER + 1, "end": MAX_DB_INTEGER + 2}, "start"),
        ({"end": MAX_DB_INTEGER + 1}, "end"),
        ({"source_offset": MAX_DB_INTEGER + 1}, "source_offset"),
    ],
)
def test_document_mention_database_integers_are_bounded(overrides, field_name):
    with pytest.raises(ValueError, match=field_name):
        _mention("bad-range", "Orion", "model", **overrides)


@pytest.mark.parametrize(
    ("overrides", "field_name"),
    [
        ({"chunk_id": MAX_DB_INTEGER + 1}, "chunk_id"),
        ({"chunk_id": 1.0}, "chunk_id"),
        ({"start": True}, "start"),
        ({"start": -1}, "start"),
        ({"start": MAX_DB_INTEGER + 1}, "start"),
        ({"end": "5"}, "end"),
        ({"end": 0}, "end"),
        ({"end": MAX_DB_INTEGER + 1}, "end"),
        ({"source_offset": False}, "source_offset"),
        ({"source_offset": MAX_DB_INTEGER + 1}, "source_offset"),
    ],
)
def test_public_fingerprint_strictly_validates_database_integer_ranges(
    overrides, field_name
):
    with pytest.raises(ValueError, match=field_name):
        resolution_input_fingerprint((_mapping_mention(**overrides),))


@pytest.mark.parametrize("source_offset", [True, "0", -1, MAX_DB_INTEGER + 1])
@pytest.mark.parametrize(
    "chunk",
    [None, SimpleNamespace(content="Orion", start_position=40)],
    ids=["without-chunk", "with-chunk"],
)
def test_public_fingerprint_validates_source_offset_without_source_text(
    source_offset,
    chunk,
):
    mention = _mapping_mention(source_offset=source_offset)
    mention.pop("source_text")
    if chunk is not None:
        mention["chunk"] = chunk

    with pytest.raises(ValueError, match="source_offset"):
        resolution_input_fingerprint((mention,))


def test_public_fingerprint_binds_source_offset_without_source_text():
    first = _mapping_mention(source_offset=3)
    second = _mapping_mention(source_offset=4)
    first.pop("source_text")
    second.pop("source_text")

    assert resolution_input_fingerprint((first,)) != resolution_input_fingerprint(
        (second,)
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"document_id": "not-a-uuid"}, "document_id.*UUID"),
        ({"document_id": _UUIDLikeObject()}, "document_id.*UUID"),
        ({"chunk_id": 0}, "chunk_id"),
        ({"chunk_id": True}, "chunk_id"),
        ({"chunk_id": "1"}, "chunk_id"),
        (
            {
                "position_basis": "chunk_content",
                "content_object_id": "not-a-uuid",
            },
            "content_object_id.*UUID",
        ),
        (
            {
                "position_basis": "chunk_content",
                "content_object_id": _UUIDLikeObject(),
            },
            "content_object_id.*UUID",
        ),
        ({"source_key": 0}, "source key.*string"),
        ({"source_key": "   "}, "source key.*nonempty"),
        ({"source_key": "bad\x01key"}, "source key.*control"),
        ({"source_key": "bad\u202ekey"}, "source key.*control"),
        ({"source_key": "x" * 513}, "source key.*limit"),
    ],
)
def test_mapping_inputs_cannot_bypass_source_identity_scalar_validation(
    overrides, message
):
    mention = {
        "mention_id": "mapping",
        "raw_text": "Orion",
        "entity_type": "model",
        "start": 0,
        "end": 5,
        "source_text": "Orion",
        "source_offset": 0,
        "confidence": 0.9,
        "document_id": str(DOCUMENT_ID),
        "chunk_id": 1,
        "source_key": "provided-source-key",
        **overrides,
    }

    with pytest.raises(ValueError, match=message):
        resolution_input_fingerprint((mention,))


def test_explicit_identifier_and_source_key_cannot_masquerade_as_empty_strings():
    class PretendsToBeEmpty:
        def __eq__(self, other):
            return other == ""

    with pytest.raises(ValueError, match="identifier.*string"):
        resolution_input_fingerprint(
            (_mapping_mention(identifier=PretendsToBeEmpty()),)
        )
    with pytest.raises(ValueError, match="source key.*string"):
        resolution_input_fingerprint(
            (_mapping_mention(source_key=PretendsToBeEmpty()),)
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"identifier": _MasqueradingString("doi:10.5555/12345678")}, "identifier"),
        ({"source_key": _MasqueradingString("source:key")}, "source key"),
        ({"source_text": _MasqueradingString("Orion")}, "source text"),
        ({"position_basis": _CoordinateBasisMasquerade("evil")}, "position_basis"),
        ({"document_id": _DocumentUUIDMasquerade("evil")}, "document_id"),
        (
            {
                "position_basis": "chunk_content",
                "content_object_id": _ContentUUIDMasquerade("evil"),
            },
            "content_object_id",
        ),
    ],
)
def test_public_resolution_strings_must_be_exact_builtin_strings(overrides, message):
    with pytest.raises(ValueError, match=message):
        resolution_input_fingerprint((_mapping_mention(**overrides),))


def test_document_mention_and_metadata_reject_string_subclass_boundaries():
    with pytest.raises(ValueError, match="source key"):
        _mention(
            "mention",
            "Orion",
            source_key=_MasqueradingString("source:key"),
        )
    with pytest.raises(ValueError, match="identifier"):
        resolution_input_fingerprint(
            (
                _mapping_mention(
                    metadata={
                        "stable_identifier": _ExplosiveString("doi:10.5555/12345678")
                    }
                ),
            )
        )


def test_repeated_source_context_rejects_string_subclass_equality_masquerade():
    first = _mapping_mention(
        mention_id="first",
        source_key="shared-source",
        source_text="Orion",
    )
    second = _mapping_mention(
        mention_id="second",
        source_key="shared-source",
        source_text=_MasqueradingString("Changed context"),
    )

    with pytest.raises(ValueError, match="source text"):
        resolution_input_fingerprint((first, second))


@pytest.mark.parametrize(
    "metadata",
    [
        {"stable_identifier": False},
        {"stable_identifier": 0},
        {"stable_identifier": ""},
        {"stable_identifier": "   "},
        {"stable_identifier": None},
        {"identifier": False},
        {"identifier": ""},
        {
            "stable_identifier": "doi:10.5555/12345678",
            "identifier": "doi:10.5555/87654321",
        },
    ],
)
def test_metadata_identifier_keys_are_validated_fail_closed(metadata):
    with pytest.raises(ValueError, match="identifier"):
        resolution_input_fingerprint((_mapping_mention(metadata=metadata),))


@pytest.mark.parametrize(
    ("map_name", "name", "aliases", "message"),
    [
        ("x" * 129, "model", (), "ontology type map key.*128"),
        ("model", "x" * 129, (), "ontology type name.*128"),
        ("model", "model", ("x" * 129,), "ontology type alias.*128"),
        ("bad\x01key", "model", (), "ontology type map key.*control"),
        ("model", "bad\x00name", (), "ontology type name.*control"),
        ("model", "model", ("bad\x01alias",), "ontology type alias.*control"),
        ("model", "model", ("bad\u202ealias",), "ontology type alias.*control"),
    ],
)
def test_ontology_type_names_keys_and_aliases_obey_persistence_bounds(
    map_name, name, aliases, message
):
    ontology = SimpleNamespace(
        checksum="b" * 64,
        entity_types=MappingProxyType(
            {map_name: SimpleNamespace(name=name, aliases=aliases)}
        ),
    )

    with pytest.raises(ValueError, match=message):
        resolve_document_mentions((_mention("mention", "Orion", "model"),), ontology)


def test_fallback_ontology_checksum_binds_type_map_keys_used_as_aliases():
    canonical_key = SimpleNamespace(
        checksum="",
        entity_types=MappingProxyType(
            {"model": SimpleNamespace(name="model", aliases=())}
        ),
    )
    alias_key = SimpleNamespace(
        checksum="",
        entity_types=MappingProxyType(
            {"architecture": SimpleNamespace(name="model", aliases=())}
        ),
    )
    mention = _mention("mention", "Orion", "model")

    canonical = resolve_document_mentions((mention,), canonical_key)
    alias = resolve_document_mentions((mention,), alias_key)

    assert canonical.ontology_checksum != alias.ontology_checksum


@pytest.mark.parametrize("checksum", [None, False, 0, (), []])
def test_ontology_checksum_fallback_requires_an_exact_empty_string(checksum):
    ontology = SimpleNamespace(
        checksum=checksum,
        entity_types=_ontology().entity_types,
    )

    with pytest.raises(ValueError, match="ontology checksum"):
        resolve_document_mentions((_mention("mention", "Orion", "model"),), ontology)


def test_ontology_checksum_cannot_masquerade_as_the_empty_string():
    class PretendsToBeEmpty:
        def __eq__(self, other):
            return other == ""

    ontology = SimpleNamespace(
        checksum=PretendsToBeEmpty(),
        entity_types=_ontology().entity_types,
    )

    with pytest.raises(ValueError, match="ontology checksum"):
        resolve_document_mentions((_mention("mention", "Orion", "model"),), ontology)


def test_ontology_checksum_must_be_an_exact_builtin_string():
    ontology = SimpleNamespace(
        checksum=_MasqueradingString("b" * 64),
        entity_types=_ontology().entity_types,
    )

    with pytest.raises(ValueError, match="ontology checksum"):
        resolve_document_mentions((_mention("mention", "Orion", "model"),), ontology)

    result = resolve_document_mentions(
        (_mention("mention", "Orion", "model"),), _ontology()
    )
    with pytest.raises(ValueError, match="ontology_checksum"):
        replace(result, ontology_checksum=_MasqueradingString("b" * 64))


@pytest.mark.parametrize(
    "field_name",
    ("resolver_version", "ontology_checksum", "input_fingerprint", "checksum"),
)
def test_resolution_result_identity_strings_require_exact_builtin_strings(field_name):
    result = resolve_document_mentions(
        (_mention("mention", "Orion", "model"),), _ontology()
    )

    with pytest.raises(ValueError, match=field_name):
        replace(
            result,
            **{field_name: _MasqueradingString(getattr(result, field_name))},
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "cluster_key",
        "label",
        "normalized_label",
        "version_signature",
        "entity_type",
        "identifier",
        "method",
    ),
)
def test_resolved_cluster_identity_strings_require_exact_builtin_strings(field_name):
    cluster = resolve_document_mentions(
        (_mention("mention", "Orion", "model"),), _ontology()
    ).clusters[0]

    with pytest.raises(ValueError, match=field_name):
        replace(
            cluster,
            **{field_name: _MasqueradingString(getattr(cluster, field_name))},
        )


def test_result_cluster_and_membership_mention_ids_require_exact_builtin_strings():
    result = resolve_document_mentions(
        (_mention("mention", "Orion", "model"),), _ontology()
    )
    cluster = result.clusters[0]
    membership = cluster.memberships[0]
    subclass_id = _MasqueradingString("mention")

    with pytest.raises(ValueError, match="mention_id"):
        replace(membership, mention_id=subclass_id)
    with pytest.raises(ValueError, match="mention_ids"):
        replace(cluster, mention_ids=(subclass_id,))
    with pytest.raises(ValueError, match="mention_ids"):
        replace(result, mention_ids=(subclass_id,))


@pytest.mark.parametrize(
    ("owner_name", "field_name"),
    (
        ("cluster", "mention_ids"),
        ("cluster", "memberships"),
        ("result", "mention_ids"),
        ("result", "clusters"),
        ("result", "decisions"),
    ),
)
def test_resolution_identity_containers_require_exact_tuples(owner_name, field_name):
    result = resolve_document_mentions(
        (_mention("mention", "Orion", "model"),), _ontology()
    )
    owner = result.clusters[0] if owner_name == "cluster" else result

    with pytest.raises(ValueError, match=field_name):
        replace(
            owner,
            **{field_name: _ExplosiveTuple(getattr(owner, field_name))},
        )


@pytest.mark.parametrize("field_name", ("method", "reason"))
def test_persisted_membership_strings_require_exact_builtin_strings(field_name):
    membership = (
        resolve_document_mentions((_mention("mention", "Orion", "model"),), _ontology())
        .clusters[0]
        .memberships[0]
    )

    with pytest.raises(ValueError, match=field_name):
        replace(
            membership,
            **{field_name: _MasqueradingString(getattr(membership, field_name))},
        )


def test_persisted_membership_parent_requires_an_exact_builtin_string():
    cluster = resolve_document_mentions(
        (
            _mention("first", "Orion", "model", start=0),
            _mention("second", "Orion", "model", start=20),
        ),
        _ontology(),
    ).clusters[0]
    child = next(item for item in cluster.memberships if item.parent_mention_id)

    with pytest.raises(ValueError, match="parent_mention_id"):
        replace(
            child,
            parent_mention_id=_MasqueradingString(child.parent_mention_id),
        )


def test_source_snapshot_rejects_forged_string_subclass_fingerprint():
    mention = _mapping_mention(id="mapping")
    result = resolve_document_mentions((mention,), _ontology())
    forged = replace(result)
    object.__setattr__(
        forged,
        "input_fingerprint",
        _MasqueradingString("f" * 64),
    )
    object.__setattr__(forged, "checksum", resolution_result_checksum(forged))

    with pytest.raises(ResolutionPersistenceError, match="source context|fingerprint"):
        _validate_source_snapshot(
            SimpleNamespace(scope_id=DOCUMENT_ID),
            forged,
            (mention,),
        )


def _destination_objects(result):
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.models.artifacts import graph_identity_checksum

    identity = {
        "scope_type": GraphArtifact.ScopeType.DOCUMENT,
        "scope_id": DOCUMENT_ID,
        "source_hash": "a" * 64,
        "ontology_version": "1.0.0",
        "extractor_version": "extractor-v1",
        "resolver_version": result.resolver_version,
        "filter_policy_version": "pending-v1",
        "ontology_checksum": result.ontology_checksum,
        "filter_policy_checksum": graph_identity_checksum(
            "document-filter-policy", "pending-v1"
        ),
        "resolution_config_checksum": graph_identity_checksum(
            "document-resolver", result.resolver_version
        ),
    }
    artifact = SimpleNamespace(
        pk=7,
        status=GraphArtifact.Status.BUILDING,
        **identity,
    )
    run = SimpleNamespace(
        artifact_id=7,
        status=GraphBuildRun.Status.RUNNING,
        stage=GraphBuildRun.Stage.RESOLUTION,
        **identity,
    )
    return artifact, run


@pytest.mark.parametrize("target", ("cluster", "membership", "decision"))
def test_destination_recursively_rejects_forged_nested_identity_strings(target):
    result = resolve_document_mentions(
        (
            _mention("first", "Orion", "model", start=0),
            _mention("second", "Orion", "model", start=20),
        ),
        _ontology(),
    )
    cluster = result.clusters[0]
    if target == "cluster":
        object.__setattr__(
            cluster,
            "cluster_key",
            _MasqueradingString(cluster.cluster_key),
        )
    elif target == "membership":
        object.__setattr__(
            cluster.memberships[0],
            "method",
            _MasqueradingString(cluster.memberships[0].method),
        )
    else:
        object.__setattr__(
            result.decisions[0],
            "method",
            _MasqueradingString(result.decisions[0].method),
        )
    object.__setattr__(result, "checksum", resolution_result_checksum(result))
    artifact, run = _destination_objects(result)

    with pytest.raises(ResolutionPersistenceError, match="result"):
        _validate_destination(artifact, run, result)


def test_source_snapshot_rejects_forged_result_mention_ids():
    mention = _mapping_mention(id="mapping")
    result = resolve_document_mentions((mention,), _ontology())
    object.__setattr__(
        result,
        "mention_ids",
        (_MasqueradingString("mapping"),),
    )
    object.__setattr__(result, "checksum", resolution_result_checksum(result))

    with pytest.raises(ResolutionPersistenceError, match="mention IDs"):
        _validate_source_snapshot(
            SimpleNamespace(scope_id=DOCUMENT_ID),
            result,
            (mention,),
        )


def test_destination_rejects_forged_cluster_mention_id_container_before_iteration():
    result = resolve_document_mentions(
        (_mention("mention", "Orion", "model"),), _ontology()
    )
    object.__setattr__(
        result.clusters[0],
        "mention_ids",
        _ExplosiveTuple(("mention",)),
    )
    artifact, run = _destination_objects(result)

    with pytest.raises(ResolutionPersistenceError, match="result"):
        _validate_destination(artifact, run, result)


@pytest.mark.parametrize(
    "identifier",
    [
        " repository:github.com/example/orion ",
        "repository:github.com/example/orion/issues/1",
        "DOI:10.5555/12345678",
        "x" * 256,
    ],
)
def test_resolved_clusters_reject_noncanonical_or_unpersistable_identifiers(identifier):
    result = resolve_document_mentions(
        (_mention("only", "Orion", "model"),), _ontology()
    )

    with pytest.raises(ValueError, match="identifier.*canonical|persistence"):
        replace(result.clusters[0], identifier=identifier)


def test_duplicate_source_coordinate_member_identities_are_rejected():
    with pytest.raises(ValueError, match="source-coordinate member identities"):
        resolve_document_mentions(
            (
                _mention("first-row", "Orion", "model", start=0),
                _mention("duplicate-row", "Orion", "model", start=0),
            ),
            _ontology(),
        )


def test_resolution_result_rejects_duplicate_cluster_keys():
    result = resolve_document_mentions(
        (_mention("one", "Orion", "model", start=0),), _ontology()
    )
    first = result.clusters[0]
    second = replace(
        first,
        mention_ids=("two",),
        memberships=(replace(first.memberships[0], mention_id="two"),),
    )
    decision = PairDecision(
        left_mention_id="one",
        right_mention_id="two",
        accepted=False,
        method="normalized_name_mismatch",
        confidence=0.0,
        explanation="Rejected: no conservative identity rule matched.",
    )

    with pytest.raises(ValueError, match="cluster keys must be unique"):
        ResolutionResult(
            resolver_version=result.resolver_version,
            ontology_checksum=result.ontology_checksum,
            input_fingerprint=result.input_fingerprint,
            mention_ids=("one", "two"),
            clusters=(first, second),
            decisions=(decision,),
            checksum="a" * 64,
        )


def test_document_entity_schema_supports_distinct_deterministic_singleton_clusters():
    from apps.knowledge_graph.models import DocumentEntity

    field = DocumentEntity._meta.get_field("cluster_key")
    version_field = DocumentEntity._meta.get_field("version_signature")
    assert field.max_length == 64
    assert field.editable is False
    assert version_field.max_length == 128
    assert version_field.editable is False
    assert "kg_document_version_signature_valid" in {
        constraint.name for constraint in DocumentEntity._meta.constraints
    }
    unique_fields = {
        tuple(constraint.fields)
        for constraint in DocumentEntity._meta.constraints
        if isinstance(constraint, UniqueConstraint)
    }
    assert ("artifact", "cluster_key") in unique_fields
    assert (
        "artifact",
        "document_id",
        "entity_type",
        "normalized_label",
    ) not in unique_fields
    assert (
        "artifact",
        "entity_type",
        "identifier",
        "version_signature",
    ) in unique_fields
    assert ("artifact", "entity_type", "identifier") not in unique_fields
    assert {
        "cluster_key",
        "label",
        "version_signature",
        "metadata",
        "created_at",
    }.issubset(DocumentEntity._IMMUTABLE_FIELDS)


def test_document_mention_links_store_immutable_typed_resolution_audit_fields():
    from apps.knowledge_graph.models import DocumentEntityMention

    method = DocumentEntityMention._meta.get_field("method")
    resolver_version = DocumentEntityMention._meta.get_field("resolver_version")
    parent_mention_id = DocumentEntityMention._meta.get_field("parent_mention_id")
    assert method.max_length == 64
    assert resolver_version.max_length == 128
    assert parent_mention_id.max_length == 128
    assert "root" in DocumentEntityMention.Method.values
    assert "kg_document_mention_parent_valid" in {
        constraint.name for constraint in DocumentEntityMention._meta.constraints
    }
    assert "method" in DocumentEntityMention._IMMUTABLE_FIELDS
    assert "resolver_version" in DocumentEntityMention._IMMUTABLE_FIELDS
    assert "parent_mention_id" in DocumentEntityMention._IMMUTABLE_FIELDS


def test_task7_artifact_identity_uses_the_shared_concrete_resolver_version():
    from apps.knowledge_graph.extraction.pipeline import _artifact_identity_values

    extraction_settings = SimpleNamespace(
        provider="gliner2_local",
        model_id="fastino/gliner2-base-v1",
        model_revision="a" * 40,
    )

    identity = _artifact_identity_values(
        DOCUMENT_ID,
        "b" * 64,
        "1.0.0",
        "c" * 64,
        settings=extraction_settings,
    )

    assert identity["resolver_version"] == DOCUMENT_RESOLVER_VERSION


def test_resolution_commit_validator_requires_exact_typed_counts_and_hashes():
    values = {
        "version": 1,
        "resolver_version": DOCUMENT_RESOLVER_VERSION,
        "ontology_checksum": "a" * 64,
        "source_mention_count": 2,
        "source_mention_fingerprint": "b" * 64,
        "document_entity_count": 1,
        "membership_count": 2,
        "result_checksum": "c" * 64,
    }

    assert resolution_commit_is_valid(
        values,
        resolver_version=DOCUMENT_RESOLVER_VERSION,
        ontology_checksum="a" * 64,
        source_mention_count=2,
        source_mention_fingerprint="b" * 64,
        document_entity_count=1,
        membership_count=2,
        result_checksum="c" * 64,
    )
    for field, invalid in (
        ("version", True),
        ("resolver_version", "other"),
        ("ontology_checksum", "A" * 64),
        ("source_mention_count", 2.0),
        ("source_mention_fingerprint", "short"),
        ("document_entity_count", -1),
        ("membership_count", 1),
        ("result_checksum", "g" * 64),
    ):
        changed = {**values, field: invalid}
        assert not resolution_commit_is_valid(
            changed,
            resolver_version=DOCUMENT_RESOLVER_VERSION,
            ontology_checksum="a" * 64,
            source_mention_count=2,
            source_mention_fingerprint="b" * 64,
            document_entity_count=1,
            membership_count=2,
            result_checksum="c" * 64,
        )

    subclass_marker = {
        **values,
        "ontology_checksum": _MasqueradingString("f" * 64),
    }
    assert not resolution_commit_is_valid(
        subclass_marker,
        resolver_version=DOCUMENT_RESOLVER_VERSION,
        ontology_checksum="a" * 64,
        source_mention_count=2,
        source_mention_fingerprint="b" * 64,
        document_entity_count=1,
        membership_count=2,
        result_checksum="c" * 64,
    )
    assert not resolution_commit_is_valid(
        values,
        resolver_version=DOCUMENT_RESOLVER_VERSION,
        ontology_checksum=_MasqueradingString("f" * 64),
        source_mention_count=2,
        source_mention_fingerprint="b" * 64,
        document_entity_count=1,
        membership_count=2,
        result_checksum="c" * 64,
    )


def test_source_mention_fingerprint_is_order_independent_and_evidence_sensitive():
    common = {
        "artifact_id": 9,
        "document_id": DOCUMENT_ID,
        "chunk_id": 11,
        "position_basis": "document_global",
        "normalized_text": "orion",
        "entity_type": "model",
        "extraction_confidence": 0.9,
        "content_object_type_id": None,
        "content_object_id": None,
        "metadata": {"observations": [{"chunk_id": 11, "start": 0, "end": 5}]},
    }
    first = {**common, "id": 1, "start": 0, "end": 5, "raw_text": "Orion"}
    second = {**common, "id": 2, "start": 10, "end": 15, "raw_text": "Orion"}

    forward = source_mention_fingerprint((first, second))
    reverse = source_mention_fingerprint((second, first))
    changed = source_mention_fingerprint(
        (first, {**second, "raw_text": "Altair", "normalized_text": "altair"})
    )

    assert forward == reverse
    assert len(forward) == 64
    assert changed != forward


def test_committed_resolution_state_rejects_extra_inactive_rows():
    result = resolve_document_mentions(
        (
            _mention("one", "Orion", "model", start=0),
            _mention("two", "Orion", "model", start=20),
        ),
        _ontology(),
    )
    cluster = result.clusters[0]
    status = SimpleNamespace(ACTIVE="active")
    entity = SimpleNamespace(
        cluster_key=cluster.cluster_key,
        label=cluster.label,
        normalized_label=cluster.normalized_label,
        version_signature=cluster.version_signature,
        entity_type=cluster.entity_type,
        identifier=cluster.identifier,
        resolution_confidence=cluster.confidence,
        metadata={
            "resolver_version": result.resolver_version,
            "methods": sorted(
                {
                    membership.method
                    for membership in cluster.memberships
                    if membership.method != "root"
                }
                or {"root"}
            ),
            "resolution_confidence": cluster.confidence,
            "result_checksum": result.checksum,
        },
        status="active",
        Status=status,
    )
    links = tuple(
        SimpleNamespace(
            document_entity=entity,
            mention_id=membership.mention_id,
            method=membership.method,
            resolver_version=result.resolver_version,
            parent_mention_id=membership.parent_mention_id or "",
            reason=membership.reason,
            metadata={"result_checksum": result.checksum},
            status="active",
            Status=status,
        )
        for membership in cluster.memberships
    )
    extra = SimpleNamespace(
        **{
            **vars(entity),
            "cluster_key": "f" * 64,
            "status": "suppressed",
        }
    )

    assert _resolution_rows_match(result, (entity,), links)
    assert not _resolution_rows_match(result, (entity, extra), links)
    tampered = SimpleNamespace(
        **{
            **vars(entity),
            "metadata": {**entity.metadata, "result_checksum": "0" * 64},
        }
    )
    assert not _resolution_rows_match(result, (tampered,), links)
    for field_name in ("resolver_version", "result_checksum"):
        subclass_metadata = {
            **entity.metadata,
            field_name: _MasqueradingString(entity.metadata[field_name]),
        }
        subclass_row = SimpleNamespace(
            **{**vars(entity), "metadata": subclass_metadata}
        )
        assert not _resolution_rows_match(result, (subclass_row,), links)
    object.__setattr__(cluster, "label", _MasqueradingString("wrong"))
    object.__setattr__(result, "checksum", resolution_result_checksum(result))
    entity.metadata["result_checksum"] = result.checksum
    for link in links:
        link.metadata["result_checksum"] = result.checksum
    assert not _resolution_rows_match(result, (entity,), links)


def test_singleton_resolution_rows_match_exact_link_provenance_idempotently():
    result = resolve_document_mentions(
        (_mention("only", "Orion", "model"),), _ontology()
    )
    cluster = result.clusters[0]
    membership = cluster.memberships[0]
    status = SimpleNamespace(ACTIVE="active")
    entity = SimpleNamespace(
        cluster_key=cluster.cluster_key,
        label=cluster.label,
        normalized_label=cluster.normalized_label,
        version_signature=cluster.version_signature,
        entity_type=cluster.entity_type,
        identifier=cluster.identifier,
        resolution_confidence=cluster.confidence,
        metadata={
            "resolver_version": result.resolver_version,
            "methods": ["singleton"],
            "resolution_confidence": cluster.confidence,
            "result_checksum": result.checksum,
        },
        status="active",
        Status=status,
    )
    link = SimpleNamespace(
        document_entity=entity,
        mention_id="only",
        method="singleton",
        resolver_version=result.resolver_version,
        parent_mention_id="",
        reason="Singleton cluster.",
        metadata={"result_checksum": result.checksum},
        status="active",
        Status=status,
    )

    assert membership.method == "singleton"
    assert _resolution_rows_match(result, (entity,), (link,))
    assert not _resolution_rows_match(
        result,
        (entity,),
        (SimpleNamespace(**{**vars(link), "method": "root"}),),
    )
    for field_name in ("method", "resolver_version", "parent_mention_id", "reason"):
        subclass_link = SimpleNamespace(
            **{
                **vars(link),
                field_name: _MasqueradingString(getattr(link, field_name)),
            }
        )
        assert not _resolution_rows_match(result, (entity,), (subclass_link,))
    subclass_checksum_link = SimpleNamespace(
        **{
            **vars(link),
            "metadata": {
                "result_checksum": _MasqueradingString(result.checksum),
            },
        }
    )
    assert not _resolution_rows_match(
        result,
        (entity,),
        (subclass_checksum_link,),
    )


def _database_is_reachable():
    database = settings.DATABASES["default"]
    try:
        with socket.create_connection(
            (database["HOST"], int(database.get("PORT") or 5432)), timeout=0.2
        ):
            return True
    except OSError:
        return False


_POSTGRES_AVAILABLE = _database_is_reachable()
_POSTGRES_REQUIRED = os.environ.get(
    "KG_REQUIRE_POSTGRES_TESTS", ""
).strip().lower() in {"1", "true", "yes", "on"}
database_required = pytest.mark.skipif(
    not _POSTGRES_AVAILABLE and not _POSTGRES_REQUIRED,
    reason="configured PostgreSQL database is not reachable",
)


@pytest.mark.django_db(transaction=True)
@database_required
def test_persistence_links_mentions_audibly_without_owning_build_lifecycle():
    from apps.documents.models import TextChunk
    from apps.knowledge_graph.models import (
        DocumentEntity,
        DocumentEntityMention,
        EntityMention,
        GraphArtifact,
        GraphBuildRun,
    )

    artifact = GraphArtifact.objects.create(
        scope_type=GraphArtifact.ScopeType.DOCUMENT,
        scope_id=DOCUMENT_ID,
        status=GraphArtifact.Status.BUILDING,
        source_hash="a" * 64,
        ontology_version="1.0.0",
        extractor_version="extractor-v1",
        resolver_version=RESOLVER_VERSION,
        filter_policy_version="pending-v1",
    )
    chunk = TextChunk.objects.create(
        doc_id=DOCUMENT_ID,
        content="Orion and Orion",
        start_position=0,
        end_position=15,
        chunk_number=0,
        modality=TextChunk.Modality.TEXT,
        embedding=[0.0] * 1024,
    )
    mentions = [
        EntityMention.objects.create(
            artifact=artifact,
            document_id=DOCUMENT_ID,
            chunk=chunk,
            start=start,
            end=start + 5,
            position_basis=EntityMention.PositionBasis.DOCUMENT_GLOBAL,
            raw_text="Orion",
            normalized_text="Orion",
            entity_type="model",
            extraction_confidence=confidence,
            metadata={"observations": []},
        )
        for start, confidence in ((0, 0.8), (10, 0.9))
    ]
    original_stats = {
        "entity_mention_count": 2,
        "relation_mention_count": 0,
        "ontology_checksum": "b" * 64,
        "extraction_commit": {
            "version": 1,
            "entity_mention_count": 2,
            "relation_mention_count": 0,
        },
        "provider": "gliner2_local",
    }
    run = GraphBuildRun.objects.create(
        artifact=artifact,
        stage=GraphBuildRun.Stage.EXTRACTION,
        status=GraphBuildRun.Status.RUNNING,
        stats=original_stats,
    )
    result = resolve_document_mentions(tuple(mentions), _ontology())

    persisted = persist_document_resolution(
        artifact.pk,
        run.pk,
        result,
    )

    assert len(persisted) == 1
    entity = DocumentEntity.objects.get(artifact=artifact)
    assert entity.label == "Orion"
    assert entity.version_signature == ""
    assert entity.cluster_key == result.clusters[0].cluster_key
    assert entity.metadata["resolver_version"] == RESOLVER_VERSION
    assert entity.metadata["methods"] == ["normalized_name"]
    links = list(DocumentEntityMention.objects.filter(document_entity=entity))
    assert {link.mention_id for link in links} == {mention.pk for mention in mentions}
    assert {link.resolver_version for link in links} == {RESOLVER_VERSION}
    links_by_mention = {link.mention_id: link for link in links}
    assert links_by_mention[mentions[0].pk].method == "root"
    assert links_by_mention[mentions[0].pk].parent_mention_id == ""
    assert links_by_mention[mentions[1].pk].method == "normalized_name"
    assert links_by_mention[mentions[1].pk].parent_mention_id == str(mentions[0].pk)
    assert EntityMention.objects.filter(artifact=artifact).count() == 2
    run.refresh_from_db()
    artifact.refresh_from_db()
    assert run.stage == GraphBuildRun.Stage.EXTRACTION
    assert run.status == GraphBuildRun.Status.RUNNING
    assert artifact.status == GraphArtifact.Status.BUILDING
    assert run.stats["extraction_commit"] == original_stats["extraction_commit"]
    assert run.stats["provider"] == "gliner2_local"
    assert run.stats["resolution_commit"]["resolver_version"] == RESOLVER_VERSION
    assert run.stats["resolution_commit"]["ontology_checksum"] == "b" * 64
    assert run.stats["resolution_commit"]["source_mention_count"] == 2
    assert run.stats["resolution_commit"]["membership_count"] == 2
    assert run.stats["resolution_commit"]["document_entity_count"] == 1
    assert run.stats["resolution_commit"]["result_checksum"] == result.checksum

    repeated = persist_document_resolution(artifact.pk, run.pk, result)
    assert [row.pk for row in repeated] == [entity.pk]
    assert DocumentEntity.objects.filter(artifact=artifact).count() == 1
    assert DocumentEntityMention.objects.filter(mention__artifact=artifact).count() == 2


def test_coreference_module_has_no_provider_embedding_or_llm_imports():
    import apps.knowledge_graph.resolution.coreference as module

    source = open(module.__file__, encoding="utf-8").read()
    forbidden = ("gliner2", "openai", "anthropic", "embedding", "lib.llm")
    assert not any(name in source.lower() for name in forbidden)
