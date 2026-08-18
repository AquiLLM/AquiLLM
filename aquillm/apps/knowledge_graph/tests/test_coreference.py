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
    resolve_document_mentions,
)
from apps.knowledge_graph.resolution.persistence import (
    _resolution_rows_match,
    persist_document_resolution,
    resolution_commit_is_valid,
    source_mention_fingerprint,
)

DOCUMENT_ID = uuid.UUID("11111111-1111-4111-8111-111111111111")
RESOLVER_VERSION = DOCUMENT_RESOLVER_VERSION


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
    }
    values.update(overrides)
    return DocumentMention(**values)


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
                content_object_id="figure-one",
            ),
        ),
        _ontology(),
    )

    assert _cluster_ids(result) == {
        frozenset(("full", "definition")),
        frozenset(("figure",)),
    }
    assert _decision(result, "definition", "figure").method == "source_mismatch"


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
    }

    with pytest.raises(ValueError, match="source text.*limit"):
        resolve_document_mentions((mention,), _ontology())

    invalid_empty_source = {
        **mention,
        "source_text": "",
        "source_offset": -1,
    }
    with pytest.raises(ValueError, match="source context"):
        resolve_document_mentions((invalid_empty_source,), _ontology())

    oversized_label = {
        **mention,
        "raw_text": "x" * 4_097,
        "end": 4_097,
        "source_text": "",
    }
    with pytest.raises(ValueError, match="entity label.*limit"):
        resolution_input_fingerprint((oversized_label,))


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
