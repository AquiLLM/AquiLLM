from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
import os
import socket

import pytest
from django.conf import settings
from django.core.exceptions import ValidationError
from django.db.models import UniqueConstraint

from apps.knowledge_graph.models import CanonicalEntity, CanonicalEntityLink


def _database_is_reachable() -> bool:
    database = settings.DATABASES["default"]
    try:
        with socket.create_connection(
            (database["HOST"], int(database.get("PORT") or 5432)), timeout=0.2
        ):
            return True
    except OSError:
        return False


database_required = pytest.mark.skipif(
    not _database_is_reachable() and os.environ.get("KG_REQUIRE_POSTGRES_TESTS") != "1",
    reason="configured PostgreSQL database is not reachable",
)


def _entity(
    entity_id: int,
    collection_id: int,
    label: str,
    *,
    entity_type: str = "model",
    identifier: str = "",
    version_signature: str = "",
    aliases: tuple[str, ...] = (),
    acronym_expansions: tuple[tuple[str, str], ...] = (),
):
    from apps.knowledge_graph.resolution.canonical import (
        CanonicalAcronymExpansion,
        CanonicalAliasEvidence,
        CanonicalEntityInput,
    )

    return CanonicalEntityInput(
        entity_id=entity_id,
        collection_id=collection_id,
        artifact_id=collection_id * 100,
        cluster_key=f"{entity_id:064x}",
        label=label,
        normalized_label=label.casefold(),
        entity_type=entity_type,
        identifier=identifier,
        version_signature=version_signature,
        aliases=tuple(
            CanonicalAliasEvidence(
                alias=value,
                method="ontology_alias",
                document_entity_id=entity_id * 100 + index,
                mention_id=entity_id * 1_000 + index,
                parent_mention_id=entity_id * 10_000 + index,
            )
            for index, value in enumerate(aliases, start=1)
        ),
        acronym_expansions=tuple(
            CanonicalAcronymExpansion(
                acronym=acronym.upper(),
                full_form=full_form,
                document_entity_id=entity_id * 100 + index,
                acronym_mention_id=entity_id * 1_000 + index,
                full_form_mention_id=entity_id * 10_000 + index,
            )
            for index, (acronym, full_form) in enumerate(acronym_expansions, start=1)
        ),
    )


def _component_members(result) -> tuple[tuple[int, ...], ...]:
    return tuple(component.entity_ids for component in result.components)


def test_identical_stable_identifiers_merge_only_for_compatible_type_and_version():
    from apps.knowledge_graph.resolution.canonical import (
        CanonicalOutcome,
        resolve_canonical_entities,
    )

    compatible = resolve_canonical_entities(
        (
            _entity(1, 10, "Aquila", identifier="repository:github.com/acme/aquila"),
            _entity(2, 20, "Night Sky", identifier="repository:github.com/acme/aquila"),
        )
    )
    wrong_type = resolve_canonical_entities(
        (
            _entity(1, 10, "Aquila", identifier="repository:github.com/acme/aquila"),
            _entity(
                2,
                20,
                "Aquila team",
                entity_type="organization",
                identifier="repository:github.com/acme/aquila",
            ),
        )
    )
    wrong_version = resolve_canonical_entities(
        (
            _entity(
                1,
                10,
                "Aquila v1",
                identifier="repository:github.com/acme/aquila",
                version_signature="v1",
            ),
            _entity(
                2,
                20,
                "Aquila",
                identifier="repository:github.com/acme/aquila",
            ),
        )
    )

    assert _component_members(compatible) == ((1, 2),)
    assert compatible.decisions[0].method == "stable_identifier"
    assert compatible.decisions[0].outcome is CanonicalOutcome.AUTOMATIC
    assert _component_members(wrong_type) == ((1,), (2,))
    assert wrong_type.decisions[0].reason == "ontology_type_conflict"
    assert _component_members(wrong_version) == ((1,), (2,))
    assert wrong_version.decisions[0].reason == "version_signature_conflict"


def test_exact_names_and_declared_aliases_merge_without_conflicting_identifiers():
    from apps.knowledge_graph.resolution.canonical import (
        CanonicalOutcome,
        resolve_canonical_entities,
    )

    result = resolve_canonical_entities(
        (
            _entity(1, 10, "Retrieval Augmented Generation"),
            _entity(
                2,
                20,
                "Grounded generation",
                aliases=("Retrieval-Augmented Generation",),
            ),
        )
    )
    conflict = resolve_canonical_entities(
        (
            _entity(1, 10, "Atlas", identifier="doi:10.1000/one"),
            _entity(2, 20, "Atlas", identifier="doi:10.1000/two"),
        )
    )

    assert _component_members(result) == ((1, 2),)
    assert result.decisions[0].method == "exact_name_or_alias"
    assert result.decisions[0].outcome is CanonicalOutcome.AUTOMATIC
    assert _component_members(conflict) == ((1,), (2,))
    assert conflict.decisions[0].reason == "component_stable_identifier_conflict"


def test_whole_component_conflicts_block_transitive_identifier_bridges():
    from apps.knowledge_graph.resolution.canonical import resolve_canonical_entities

    result = resolve_canonical_entities(
        (
            _entity(
                1,
                10,
                "Anchor one",
                identifier="doi:10.1000/one",
                aliases=("shared one",),
            ),
            _entity(
                2,
                20,
                "Bridge",
                aliases=("shared one", "shared two"),
            ),
            _entity(
                3,
                30,
                "Anchor two",
                identifier="doi:10.1000/two",
                aliases=("shared two",),
            ),
        )
    )

    assert _component_members(result) == ((1,), (2,), (3,))
    assert {decision.reason for decision in result.decisions} == {
        "component_stable_identifier_conflict"
    }


def test_only_identical_singleton_defined_acronym_expansions_merge():
    from apps.knowledge_graph.resolution.canonical import resolve_canonical_entities

    accepted = resolve_canonical_entities(
        (
            _entity(
                1,
                10,
                "RAG",
                acronym_expansions=(("rag", "retrieval augmented generation"),),
            ),
            _entity(
                2,
                20,
                "RAG",
                acronym_expansions=(("rag", "retrieval augmented generation"),),
            ),
        )
    )
    ambiguous = resolve_canonical_entities(
        (
            _entity(
                1,
                10,
                "RAG",
                acronym_expansions=(
                    ("rag", "retrieval augmented generation"),
                    ("rag", "really awesome graph"),
                ),
            ),
            _entity(
                2,
                20,
                "RAG",
                acronym_expansions=(("rag", "retrieval augmented generation"),),
            ),
        )
    )

    assert _component_members(accepted) == ((1, 2),)
    assert accepted.decisions[0].method == "defined_acronym"
    assert _component_members(ambiguous) == ((1,), (2,))
    assert ambiguous.decisions[0].reason == "ambiguous_acronym"


def test_embedding_similarity_is_candidate_only_and_never_changes_components():
    from apps.knowledge_graph.resolution.canonical import (
        CanonicalEmbeddingCandidate,
        CanonicalOutcome,
        resolve_canonical_entities,
    )

    result = resolve_canonical_entities(
        (_entity(1, 10, "Aquila"), _entity(2, 20, "Night sky")),
        embedding_candidates=(
            CanonicalEmbeddingCandidate(
                left_entity_id=1,
                right_entity_id=2,
                similarity=0.997,
                embedding_model_signature="local:model@locked",
                left_input_hash="a" * 64,
                right_input_hash="b" * 64,
            ),
        ),
    )

    assert _component_members(result) == ((1,), (2,))
    assert len(result.decisions) == 1
    assert result.decisions[0].outcome is CanonicalOutcome.CANDIDATE
    assert result.decisions[0].method == "embedding_similarity"


def test_embedding_candidate_is_rejected_for_known_identifier_conflict():
    from apps.knowledge_graph.resolution.canonical import (
        CanonicalEmbeddingCandidate,
        CanonicalOutcome,
        resolve_canonical_entities,
    )

    result = resolve_canonical_entities(
        (
            _entity(1, 10, "One", identifier="doi:10.1000/one"),
            _entity(2, 20, "Two", identifier="doi:10.1000/two"),
        ),
        embedding_candidates=(
            CanonicalEmbeddingCandidate(
                1, 2, 0.99, "local:model@locked", "a" * 64, "b" * 64
            ),
        ),
    )

    assert result.decisions[0].outcome is CanonicalOutcome.REJECTED
    assert result.decisions[0].reason == "conflicting_stable_identifiers"


def test_resolution_is_deterministic_under_reversed_input_and_candidate_order():
    from apps.knowledge_graph.resolution.canonical import (
        CanonicalEmbeddingCandidate,
        resolve_canonical_entities,
    )

    entities = (
        _entity(1, 30, "Aquila", aliases=("Night sky",)),
        _entity(2, 10, "Night sky"),
        _entity(3, 20, "Other"),
    )
    candidates = (
        CanonicalEmbeddingCandidate(
            2, 3, 0.9, "local:model@locked", "a" * 64, "b" * 64
        ),
    )

    forward = resolve_canonical_entities(entities, embedding_candidates=candidates)
    reverse = resolve_canonical_entities(
        tuple(reversed(entities)), embedding_candidates=tuple(reversed(candidates))
    )

    assert forward == reverse
    assert forward.checksum == reverse.checksum


def test_component_identity_survives_addition_of_an_identical_copy():
    from apps.knowledge_graph.resolution.canonical import resolve_canonical_entities

    original = resolve_canonical_entities(
        (
            _entity(1, 10, "Aquila", identifier="repository:github.com/acme/aquila"),
            _entity(2, 20, "Aquila", identifier="repository:github.com/acme/aquila"),
        )
    )
    extended = resolve_canonical_entities(
        (
            _entity(3, 30, "Aquila", identifier="repository:github.com/acme/aquila"),
            _entity(2, 20, "Aquila", identifier="repository:github.com/acme/aquila"),
            _entity(1, 10, "Aquila", identifier="repository:github.com/acme/aquila"),
        )
    )

    assert original.components[0].identity_key == extended.components[0].identity_key


def test_same_collection_and_undefined_acronym_are_never_automatic():
    from apps.knowledge_graph.resolution.canonical import resolve_canonical_entities

    same_collection = resolve_canonical_entities(
        (_entity(1, 10, "Atlas"), _entity(2, 10, "Atlas"))
    )
    undefined_acronym = resolve_canonical_entities(
        (_entity(1, 10, "RAG"), _entity(2, 20, "RAG"))
    )

    assert _component_members(same_collection) == ((1,), (2,))
    assert same_collection.decisions[0].reason == "same_collection_cannot_link"
    assert _component_members(undefined_acronym) == ((1,), (2,))
    assert undefined_acronym.decisions[0].reason == "undefined_acronym"


def test_inputs_require_canonical_stable_identifier_and_label_audit():
    from apps.knowledge_graph.resolution.canonical import CanonicalEntityInput

    values = {
        "entity_id": 1,
        "collection_id": 10,
        "artifact_id": 100,
        "cluster_key": "a" * 64,
        "label": "Aquila",
        "normalized_label": "aquila",
        "entity_type": "model",
    }
    with pytest.raises(ValueError, match="stable identifier"):
        CanonicalEntityInput(**values, identifier="not-a-stable-id")
    with pytest.raises(ValueError, match="normalized_label"):
        CanonicalEntityInput(**{**values, "normalized_label": "fabricated"})


def test_duplicate_embedding_candidate_pairs_are_rejected():
    from apps.knowledge_graph.resolution.canonical import (
        CanonicalEmbeddingCandidate,
        resolve_canonical_entities,
    )

    candidate = CanonicalEmbeddingCandidate(
        1, 2, 0.9, "local:model@locked", "a" * 64, "b" * 64
    )
    with pytest.raises(ValueError, match="duplicate"):
        resolve_canonical_entities(
            (_entity(1, 10, "One"), _entity(2, 20, "Two")),
            embedding_candidates=(candidate, candidate),
        )


def test_pair_budget_is_preflighted_before_materializing_a_large_block(monkeypatch):
    from apps.knowledge_graph.resolution import canonical

    monkeypatch.setattr(canonical, "MAX_CANONICAL_DECISIONS", 2)
    with pytest.raises(ValueError, match="decision cap"):
        canonical.resolve_canonical_entities(
            (
                _entity(1, 10, "Atlas"),
                _entity(2, 20, "Atlas"),
                _entity(3, 30, "Atlas"),
            )
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"embedding_model_signature": ""},
        {"left_input_hash": "bad"},
        {"right_input_hash": "bad"},
        {"similarity": float("nan")},
    ],
)
def test_embedding_candidate_requires_complete_finite_same_model_audit(overrides):
    from apps.knowledge_graph.resolution.canonical import CanonicalEmbeddingCandidate

    values = {
        "left_entity_id": 1,
        "right_entity_id": 2,
        "similarity": 0.9,
        "embedding_model_signature": "local:model@locked",
        "left_input_hash": "a" * 64,
        "right_input_hash": "b" * 64,
    }
    values.update(overrides)
    with pytest.raises(ValueError):
        CanonicalEmbeddingCandidate(**values)


def test_acronym_evidence_requires_uppercase_surface_and_nonacronym_full_form():
    from apps.knowledge_graph.resolution.canonical import CanonicalAcronymExpansion

    values = {
        "document_entity_id": 1,
        "acronym_mention_id": 2,
        "full_form_mention_id": 3,
    }
    with pytest.raises(ValueError, match="acronym surface"):
        CanonicalAcronymExpansion(
            acronym="rag", full_form="Retrieval Augmented Generation", **values
        )
    with pytest.raises(ValueError, match="full form"):
        CanonicalAcronymExpansion(acronym="RAG", full_form="RAG", **values)


def test_provenance_projection_ignores_unbound_metadata_and_rejects_foreign_evidence():
    from types import SimpleNamespace

    from apps.knowledge_graph.resolution.canonical import (
        CanonicalProvenanceRow,
        CanonicalSourceMembership,
        build_canonical_inputs_from_provenance,
    )

    entity = SimpleNamespace(
        pk=11,
        collection_id=1,
        artifact_id=101,
        cluster_key="a" * 64,
        label="Retrieval Augmented Generation",
        normalized_label="retrieval augmented generation",
        entity_type="concept",
        identifier="",
        version_signature="",
        metadata={"aliases": ["unbound poison"]},
    )
    without_bound_rows = build_canonical_inputs_from_provenance((entity,), (), ())
    assert without_bound_rows[0].aliases == ()
    assert without_bound_rows[0].acronym_expansions == ()

    valid = CanonicalProvenanceRow(
        collection_entity_id=11,
        document_entity_id=21,
        document_artifact_id=201,
        mention_id=31,
        parent_mention_id=32,
        method="defined_acronym",
        surface="RAG",
        parent_surface="Retrieval Augmented Generation",
        source_collection_entity_id=11,
    )
    membership = CanonicalSourceMembership(11, 21, 201)
    projected = build_canonical_inputs_from_provenance(
        (entity,), (valid,), (membership,)
    )
    assert projected[0].acronym_expansions[0].acronym == "rag"
    assert projected[0].acronym_expansions[0].full_form == (
        "retrieval augmented generation"
    )

    foreign = dataclasses.replace(valid, source_collection_entity_id=99)
    with pytest.raises(ValueError, match="exact collection entity"):
        build_canonical_inputs_from_provenance((entity,), (foreign,), (membership,))
    cross_artifact = dataclasses.replace(valid, document_artifact_id=999)
    with pytest.raises(ValueError, match="active source membership"):
        build_canonical_inputs_from_provenance(
            (entity,), (cross_artifact,), (membership,)
        )

    acronym_representative = SimpleNamespace(
        **{
            **vars(entity),
            "pk": 12,
            "cluster_key": "b" * 64,
            "label": "RAG",
            "normalized_label": "rag",
        }
    )
    acronym_row = dataclasses.replace(
        valid,
        collection_entity_id=12,
        source_collection_entity_id=12,
    )
    acronym_membership = CanonicalSourceMembership(12, 21, 201)
    projected_acronym = build_canonical_inputs_from_provenance(
        (acronym_representative,),
        (acronym_row,),
        (acronym_membership,),
    )
    assert projected_acronym[0].acronym_expansions[0].full_form == (
        "retrieval augmented generation"
    )


def test_pair_audits_collapse_deterministically_without_promoting_candidates():
    from apps.knowledge_graph.resolution.canonical import (
        CanonicalEmbeddingCandidate,
        CanonicalOutcome,
        project_canonical_link_decisions,
        resolve_canonical_entities,
    )

    entities = (
        _entity(1, 10, "Possibly related"),
        _entity(2, 20, "Aquila", identifier="repository:github.com/acme/aquila"),
        _entity(3, 30, "Aquila", identifier="repository:github.com/acme/aquila"),
    )
    candidates = (
        CanonicalEmbeddingCandidate(
            1, 2, 0.91, "local:model@locked", "a" * 64, "b" * 64
        ),
        CanonicalEmbeddingCandidate(
            1, 3, 0.93, "local:model@locked", "a" * 64, "c" * 64
        ),
    )
    forward = project_canonical_link_decisions(
        resolve_canonical_entities(entities, embedding_candidates=candidates)
    )
    reverse = project_canonical_link_decisions(
        resolve_canonical_entities(
            tuple(reversed(entities)),
            embedding_candidates=tuple(reversed(candidates)),
        )
    )

    assert forward == reverse
    automatic = tuple(
        row for row in forward if row.outcome is CanonicalOutcome.AUTOMATIC
    )
    candidate = tuple(
        row for row in forward if row.outcome is CanonicalOutcome.CANDIDATE
    )
    assert {row.source_entity_id for row in automatic} == {2, 3}
    assert all(row.status == "active" for row in automatic)
    assert len(candidate) == 1
    assert candidate[0].source_entity_id == 1
    assert candidate[0].status == "suppressed"
    audit = json.loads(candidate[0].metadata_json)
    assert len(audit["pair_decisions"]) == 2
    assert {
        item["metadata"]["right_input_hash"] for item in audit["pair_decisions"]
    } == {"b" * 64, "c" * 64}
    assert not any(
        row.source_entity_id == 1 and row.status == "active" for row in forward
    )


def test_conflicting_pair_audits_collapse_to_rejected_not_candidate():
    from apps.knowledge_graph.resolution.canonical import (
        CanonicalEmbeddingCandidate,
        CanonicalOutcome,
        project_canonical_link_decisions,
        resolve_canonical_entities,
    )

    result = resolve_canonical_entities(
        (
            _entity(1, 10, "One", identifier="doi:10.1000/one"),
            _entity(2, 20, "Two", identifier="doi:10.1000/two"),
            _entity(3, 30, "Two", identifier="doi:10.1000/two"),
        ),
        embedding_candidates=(
            CanonicalEmbeddingCandidate(
                1, 2, 0.99, "local:model@locked", "a" * 64, "b" * 64
            ),
            CanonicalEmbeddingCandidate(
                1, 3, 0.98, "local:model@locked", "a" * 64, "c" * 64
            ),
        ),
    )
    projected = project_canonical_link_decisions(result)
    cross_component = tuple(row for row in projected if row.source_entity_id == 1)

    assert len(cross_component) == 1
    assert cross_component[0].outcome is CanonicalOutcome.REJECTED
    assert cross_component[0].status == "rejected"
    assert len(json.loads(cross_component[0].metadata_json)["pair_decisions"]) == 2


def test_automatic_projection_uses_source_specific_evidence_without_member_ids():
    from apps.knowledge_graph.resolution.canonical import (
        project_canonical_link_decisions,
        resolve_canonical_entities,
    )

    result = resolve_canonical_entities(
        (
            _entity(
                1,
                10,
                "Aquila",
                identifier="repository:github.com/acme/aquila",
            ),
            _entity(
                2,
                20,
                "Aquila",
                identifier="repository:github.com/acme/aquila",
                aliases=("Night sky",),
            ),
            _entity(3, 30, "Night sky"),
        )
    )
    projected = project_canonical_link_decisions(result)
    by_source = {row.source_entity_id: row for row in projected}

    assert by_source[1].method == "stable_identifier"
    assert by_source[1].score == 1.0
    assert by_source[3].method == "exact_name_or_alias"
    assert by_source[3].score == 0.99
    for row in projected:
        metadata = json.loads(row.metadata_json)
        assert "component_entity_ids" not in metadata
        assert "component_collection_ids" not in metadata
        assert "resolution_checksum" not in metadata
        assert "left_entity_id" not in row.metadata_json
        assert "right_entity_id" not in row.metadata_json


def test_embedding_candidates_are_derived_only_from_exact_locked_endpoint_audit():
    from types import SimpleNamespace

    from apps.knowledge_graph.resolution import canonical

    def row(pk, collection_id, *, signature="local:model@locked", digest=None):
        return SimpleNamespace(
            pk=pk,
            artifact_id=collection_id * 100,
            collection_id=collection_id,
            entity_type="model",
            version_signature="",
            embedding_model_signature=signature,
            embedding_input_hash=digest or f"{pk:064x}",
            embedding=[1.0] * 1024,
        )

    candidates = canonical._derive_locked_embedding_candidates(
        (row(1, 10), row(2, 20)),
        artifact_embedding_signatures={
            1_000: "local:model@locked",
            2_000: "local:model@locked",
        },
    )
    mismatched_model = canonical._derive_locked_embedding_candidates(
        (row(1, 10), row(2, 20, signature="local:other@locked")),
        artifact_embedding_signatures={
            1_000: "local:model@locked",
            2_000: "local:other@locked",
        },
    )

    assert len(candidates) == 1
    assert candidates[0].embedding_model_signature == "local:model@locked"
    assert candidates[0].left_input_hash == f"{1:064x}"
    assert candidates[0].right_input_hash == f"{2:064x}"
    assert mismatched_model == ()
    with pytest.raises(RuntimeError, match="audit is incomplete"):
        canonical._derive_locked_embedding_candidates(
            (row(1, 10), row(2, 20, digest="poison")),
            artifact_embedding_signatures={
                1_000: "local:model@locked",
                2_000: "local:model@locked",
            },
        )
    assert set(inspect.signature(canonical.rebuild_canonical_registry).parameters) == {
        "using"
    }


def test_rejected_embedding_audit_is_valid_but_never_automatic():
    from apps.knowledge_graph.models import CollectionEntity

    source = CollectionEntity(pk=1, entity_type="model", version_signature="")
    target = CanonicalEntity(
        pk=2,
        entity_type="model",
        version_signature="",
        resolver_version="canonical-resolution-v1",
    )
    rejected = CanonicalEntityLink(
        collection_entity=source,
        canonical_entity=target,
        score=0.0,
        method="embedding_similarity",
        resolver_version="canonical-resolution-v1",
        outcome=CanonicalEntityLink.Outcome.REJECTED,
        status=CanonicalEntityLink.Status.REJECTED,
        reason="conflicting_stable_identifiers",
        metadata={},
    )
    rejected.prepare_for_persistence()
    rejected.clean()

    automatic = CanonicalEntityLink(
        collection_entity=source,
        canonical_entity=target,
        score=0.9,
        method="embedding_similarity",
        resolver_version="canonical-resolution-v1",
        outcome=CanonicalEntityLink.Outcome.AUTOMATIC,
        status=CanonicalEntityLink.Status.ACTIVE,
        reason="forbidden",
        metadata={},
    )
    automatic.prepare_for_persistence()
    with pytest.raises(ValidationError, match="never be automatic"):
        automatic.clean()


def test_canonical_schema_has_stable_registry_identity_and_one_current_target():
    field_names = {field.name for field in CanonicalEntity._meta.fields}
    link_field_names = {field.name for field in CanonicalEntityLink._meta.fields}
    current_constraints = {
        constraint.name: constraint
        for constraint in CanonicalEntityLink._meta.constraints
        if isinstance(constraint, UniqueConstraint)
    }

    assert {"identity_key", "resolver_version", "version_signature"} <= field_names
    assert {"outcome", "decision_checksum"} <= link_field_names
    assert "kg_one_active_canonical_target" in current_constraints
    assert current_constraints["kg_one_active_canonical_target"].condition is not None
    assert {
        "collection_entity",
        "collection_entity_id",
        "canonical_entity",
        "canonical_entity_id",
        "score",
        "method",
        "resolver_version",
        "outcome",
        "decision_checksum",
        "status",
        "reason",
        "metadata",
    } <= set(CanonicalEntityLink._IMMUTABLE_FIELDS)
    canonical_constraints = {
        constraint.name: constraint for constraint in CanonicalEntity._meta.constraints
    }
    assert "kg_canonical_identity_key_valid" in canonical_constraints
    assert "kg_canonical_identity_unique" in canonical_constraints
    assert {
        "identity_key",
        "resolver_version",
        "version_signature",
        "label",
        "normalized_label",
        "entity_type",
    } <= set(CanonicalEntity._IMMUTABLE_FIELDS)


def test_canonical_link_constraints_pin_outcome_status_checksum_and_method():
    constraint_names = {
        constraint.name for constraint in CanonicalEntityLink._meta.constraints
    }

    assert {
        "kg_canonical_link_outcome_status",
        "kg_canonical_link_decision_hash",
        "kg_canonical_link_method_outcome",
        "kg_canonical_embedding_candidate_only",
    } <= constraint_names


def test_canonical_migration_stages_bounded_irreversible_legacy_isolation():
    from django.db import migrations

    migration = importlib.import_module(
        "apps.knowledge_graph.migrations.0004_canonical_identity_audit"
    )
    operations = migration.Migration.operations
    run_index = next(
        index
        for index, operation in enumerate(operations)
        if isinstance(operation, migrations.RunPython)
    )
    added_identity_fields = {
        operation.name: operation
        for operation in operations[:run_index]
        if isinstance(operation, migrations.AddField)
    }

    assert {
        "identity_key",
        "resolver_version",
        "version_signature",
        "decision_checksum",
        "outcome",
    } <= set(added_identity_fields)
    assert all(operation.field.null for operation in added_identity_fields.values())
    assert operations[run_index].reversible is False
    source = inspect.getsource(migration.isolate_legacy_canonical_registry)
    helper_source = inspect.getsource(migration._isolate_legacy_batch)
    assert "_MAX_LEGACY_ROWS" in source
    assert "canonical_entity_id__in=canonical_ids" in helper_source
    assert "canonical.embedding = None" not in helper_source


def test_canonical_link_audit_fields_reject_queryset_and_instance_rewrites():
    with pytest.raises(ValidationError, match="immutable"):
        CanonicalEntityLink.objects.filter(pk=1).update(score=0.25)
    with pytest.raises(ValidationError, match="immutable"):
        CanonicalEntityLink.objects.filter(pk=1).update(canonical_entity_id=2)

    link = CanonicalEntityLink(pk=1, score=0.5)
    source = inspect.getsource(type(link).clean)
    assert "canonical_entity" in source
    assert "resolver_version" in source
    assert "decision_checksum" in source


def _embedding_signature() -> str:
    return (
        f"test-local:model@rev:endpoint={'e' * 64}:dims=1024:"
        "prep=kg-entity-v1:max_chars=8192:batch=64"
    )


def _create_active_collection_entity(
    collection,
    *,
    label: str,
    cluster_digit: str,
    identifier: str = "",
    embedding=None,
    embedding_hash: str = "",
):
    from apps.knowledge_graph.models import CollectionEntity, GraphArtifact

    artifact = GraphArtifact.objects.create(
        scope_type=GraphArtifact.ScopeType.COLLECTION,
        scope_id=collection.pk,
        status=GraphArtifact.Status.BUILDING,
        source_hash=cluster_digit * 64,
        ontology_version="ontology-v1",
        extractor_version="extractor-v1",
        resolver_version="collection-resolution-v1",
        filter_policy_version="filter-v1",
        embedding_model_signature=_embedding_signature(),
    )
    entity = CollectionEntity.objects.create(
        artifact=artifact,
        collection=collection,
        cluster_key=cluster_digit * 64,
        label=label,
        normalized_label=label.casefold(),
        entity_type="model",
        identifier=identifier,
        extraction_confidence=1.0,
        resolution_confidence=1.0,
        retrieval_utility=1.0,
        promotion_confidence=1.0,
        embedding_model_signature=(
            artifact.embedding_model_signature if embedding is not None else ""
        ),
        embedding_input_hash=embedding_hash,
        embedding=embedding,
    )
    GraphArtifact.objects.filter(pk=artifact.pk).update(
        status=GraphArtifact.Status.ACTIVE
    )
    return artifact, entity


@pytest.mark.django_db(transaction=True)
@database_required
def test_registry_rebuild_is_idempotent_preserves_pk_and_splits_after_bridge_unlink():
    from apps.collections.models import Collection
    from apps.knowledge_graph.models import GraphArtifact
    from apps.knowledge_graph.resolution.canonical import (
        CANONICAL_RESOLVER_VERSION,
        rebuild_canonical_registry,
    )

    collections = [
        Collection.objects.create(name=f"canonical {index}") for index in range(5)
    ]
    stable_id = "repository:github.com/acme/aquila"
    rows = (
        _create_active_collection_entity(
            collections[0],
            label="Alpha one",
            cluster_digit="1",
            identifier=stable_id,
        ),
        _create_active_collection_entity(
            collections[1],
            label="Alpha two",
            cluster_digit="2",
            identifier=stable_id,
        ),
        _create_active_collection_entity(
            collections[2],
            label="Bridge",
            cluster_digit="3",
            identifier=stable_id,
        ),
        _create_active_collection_entity(
            collections[3], label="Bridge", cluster_digit="4"
        ),
        _create_active_collection_entity(
            collections[4], label="Bridge", cluster_digit="5"
        ),
    )
    artifacts = tuple(row[0] for row in rows)
    entities = tuple(row[1] for row in rows)

    first = rebuild_canonical_registry()
    first_links = tuple(
        CanonicalEntityLink.objects.current(
            resolver_version=CANONICAL_RESOLVER_VERSION
        ).order_by("collection_entity_id")
    )
    original_canonical_id = first_links[0].canonical_entity_id
    assert len(first_links) == 5
    assert {row.canonical_entity_id for row in first_links} == {original_canonical_id}

    second = rebuild_canonical_registry()
    assert second.created_entity_count == 0
    assert second.created_link_count == 0
    assert second.canonical_entity_ids == first.canonical_entity_ids

    GraphArtifact.objects.filter(pk=artifacts[2].pk).update(
        status=GraphArtifact.Status.SUPERSEDED
    )
    split = rebuild_canonical_registry()
    current_links = {
        row.collection_entity_id: row.canonical_entity_id
        for row in CanonicalEntityLink.objects.current(
            resolver_version=CANONICAL_RESOLVER_VERSION
        )
    }

    assert split.active_link_count == 4
    assert current_links[entities[0].pk] == original_canonical_id
    assert current_links[entities[1].pk] == original_canonical_id
    assert current_links[entities[3].pk] == current_links[entities[4].pk]
    assert current_links[entities[3].pk] != original_canonical_id
    assert entities[2].pk not in current_links


@pytest.mark.django_db(transaction=True)
@database_required
def test_singleton_embedding_candidate_is_suppressed_rebuilt_and_superseded():
    from apps.collections.models import Collection
    from apps.knowledge_graph.models import GraphArtifact
    from apps.knowledge_graph.resolution.canonical import rebuild_canonical_registry

    left_collection = Collection.objects.create(name="candidate left")
    right_collection = Collection.objects.create(name="candidate right")
    left_artifact, left = _create_active_collection_entity(
        left_collection,
        label="Left singleton",
        cluster_digit="a",
        embedding=[1.0] * 1024,
        embedding_hash="1" * 64,
    )
    right_artifact, right = _create_active_collection_entity(
        right_collection,
        label="Right singleton",
        cluster_digit="b",
        embedding=[1.0] * 1024,
        embedding_hash="2" * 64,
    )

    first = rebuild_canonical_registry()
    candidate = CanonicalEntityLink.objects.get(
        resolver_version=first.resolver_version,
        outcome=CanonicalEntityLink.Outcome.CANDIDATE,
    )
    assert candidate.status == CanonicalEntityLink.Status.SUPPRESSED
    assert candidate.collection_entity_id == min(left.pk, right.pk)
    assert not CanonicalEntityLink.objects.current(
        resolver_version=first.resolver_version
    ).exists()

    second = rebuild_canonical_registry()
    assert second.created_entity_count == 0
    assert second.created_link_count == 0
    assert CanonicalEntityLink.objects.get(pk=candidate.pk).status == (
        CanonicalEntityLink.Status.SUPPRESSED
    )

    removed_artifact = (
        left_artifact if candidate.collection_entity_id == right.pk else right_artifact
    )
    GraphArtifact.objects.filter(pk=removed_artifact.pk).update(
        status=GraphArtifact.Status.SUPERSEDED
    )
    rebuild_canonical_registry()
    candidate.refresh_from_db()
    assert candidate.status == CanonicalEntityLink.Status.SUPERSEDED


@pytest.mark.django_db(transaction=True)
@database_required
def test_conflicting_locked_embeddings_persist_rejected_never_automatic_audit():
    from apps.collections.models import Collection
    from apps.knowledge_graph.resolution.canonical import rebuild_canonical_registry

    left_collection = Collection.objects.create(name="rejected embedding left")
    right_collection = Collection.objects.create(name="rejected embedding right")
    _create_active_collection_entity(
        left_collection,
        label="Left identifier",
        cluster_digit="c",
        identifier="doi:10.1000/left",
        embedding=[1.0] * 1024,
        embedding_hash="3" * 64,
    )
    _create_active_collection_entity(
        right_collection,
        label="Right identifier",
        cluster_digit="d",
        identifier="doi:10.1000/right",
        embedding=[1.0] * 1024,
        embedding_hash="4" * 64,
    )

    result = rebuild_canonical_registry()
    rejected = CanonicalEntityLink.objects.get(
        resolver_version=result.resolver_version,
        method="embedding_similarity",
    )

    assert rejected.outcome == CanonicalEntityLink.Outcome.REJECTED
    assert rejected.status == CanonicalEntityLink.Status.REJECTED
    assert rejected.reason == "conflicting_stable_identifiers"
    assert not CanonicalEntityLink.objects.current(
        resolver_version=result.resolver_version
    ).exists()


def _legacy_canonical_graph_row(apps, *, index: int, link_status: str, method: str):
    Collection = apps.get_model("apps_collections", "Collection")
    GraphArtifact = apps.get_model("apps_knowledge_graph", "GraphArtifact")
    CollectionEntity = apps.get_model("apps_knowledge_graph", "CollectionEntity")
    LegacyCanonical = apps.get_model("apps_knowledge_graph", "CanonicalEntity")
    LegacyLink = apps.get_model("apps_knowledge_graph", "CanonicalEntityLink")

    collection = Collection.objects.create(name=f"legacy canonical {index}")
    digit = format(index, "x")[-1]
    artifact = GraphArtifact.objects.create(
        scope_type="collection",
        scope_id=str(collection.pk),
        collection_scope_id=collection.pk,
        build_key=digit * 64,
        build_generation=1,
        orchestration_version=0,
        status="active",
        source_hash=digit * 64,
        ontology_version="ontology-v1",
        extractor_version="extractor-v1",
        resolver_version="collection-resolution-v1",
        filter_policy_version="filter-v1",
        embedding_model_signature=_embedding_signature(),
        ontology_checksum="a" * 64,
        filter_policy_checksum="b" * 64,
        resolution_config_checksum="c" * 64,
        assembly_version="collection-assembly-v1",
        assembly_config_checksum="d" * 64,
        metadata={},
    )
    source = CollectionEntity.objects.create(
        artifact_id=artifact.pk,
        collection_id=collection.pk,
        cluster_key=digit * 64,
        label=f"Legacy {index}",
        normalized_label=f"legacy {index}",
        version_signature="",
        entity_type="model",
        identifier="",
        status="active",
        extraction_confidence=1.0,
        resolution_confidence=1.0,
        retrieval_utility=1.0,
        promotion_confidence=1.0,
        filter_reason="",
        embedding_model_signature="",
        embedding_input_hash="",
        embedding=None,
        metadata={},
    )
    canonical = LegacyCanonical.objects.create(
        label=f"Legacy {index}",
        normalized_label=f"legacy {index}",
        entity_type="model",
        status="active",
        embedding=[float(index)] * 1024,
        metadata={"legacy": index},
    )
    link = LegacyLink.objects.create(
        collection_entity_id=source.pk,
        canonical_entity_id=canonical.pk,
        score=0.9,
        method=method,
        resolver_version="legacy-resolver-v0",
        status=link_status,
        reason="legacy audit",
        metadata={"legacy": index},
    )
    return source.pk, canonical.pk, link.pk


def _migrate_empty_registry_to_0003(connection):
    """Create the legacy test boundary without weakening production rollback."""
    from django.db import migrations
    from django.db.migrations.executor import MigrationExecutor

    from apps.knowledge_graph.models import CanonicalEntity, CanonicalEntityLink

    assert not CanonicalEntity.objects.exists()
    assert not CanonicalEntityLink.objects.exists()

    before = [("apps_knowledge_graph", "0003_graph_lifecycle_ownership")]
    executor = MigrationExecutor(connection)
    migration = executor.loader.get_migration(
        "apps_knowledge_graph", "0004_canonical_identity_audit"
    )
    data_operation = next(
        operation
        for operation in migration.operations
        if isinstance(operation, migrations.RunPython)
    )
    assert data_operation.reverse_code is None

    # The production migration is deliberately irreversible because historical
    # audit rows cannot be losslessly collapsed into the old unique shape. A
    # pristine test database contains no such audit, so temporarily making only
    # the data operation a no-op in reverse lets MigrationExecutor build the
    # real 0003 schema. The operation is restored before any legacy row exists.
    data_operation.reverse_code = migrations.RunPython.noop
    try:
        executor.migrate(before)
    finally:
        data_operation.reverse_code = None
    return executor.loader.project_state(before).apps


@pytest.fixture
def _restore_latest_knowledge_graph_schema(request):
    from django.db import connection
    from django.db.migrations.executor import MigrationExecutor

    def restore_latest() -> None:
        executor = MigrationExecutor(connection)
        executor.migrate(executor.loader.graph.leaf_nodes("apps_knowledge_graph"))
        from apps.knowledge_graph.models import GraphRebuildRequest

        GraphRebuildRequest.objects.exists()

    request.addfinalizer(restore_latest)


@pytest.mark.django_db(transaction=True)
@database_required
def test_0004_migrates_legacy_audits_with_checksum_parity_and_is_irreversible(
    _restore_latest_knowledge_graph_schema,
):
    from django.db import connection
    from django.db.migrations.exceptions import IrreversibleError
    from django.db.migrations.executor import MigrationExecutor

    from apps.knowledge_graph.models.associations import (
        canonical_link_decision_checksum,
    )

    before = [("apps_knowledge_graph", "0003_graph_lifecycle_ownership")]
    after = [("apps_knowledge_graph", "0004_canonical_identity_audit")]
    old_apps = _migrate_empty_registry_to_0003(connection)
    rows = (
        _legacy_canonical_graph_row(
            old_apps, index=1, link_status="active", method="stable_identifier"
        ),
        _legacy_canonical_graph_row(
            old_apps,
            index=2,
            link_status="suppressed",
            method="embedding_similarity",
        ),
        _legacy_canonical_graph_row(
            old_apps,
            index=3,
            link_status="rejected",
            method="embedding_similarity",
        ),
    )
    executor = MigrationExecutor(connection)
    executor.migrate(after)

    migrated = MigrationExecutor(connection).loader.project_state(after).apps
    MigratedCanonical = migrated.get_model("apps_knowledge_graph", "CanonicalEntity")
    MigratedLink = migrated.get_model("apps_knowledge_graph", "CanonicalEntityLink")
    expected_outcomes = ("automatic", "candidate", "rejected")
    for (_source_id, canonical_id, link_id), expected_outcome in zip(
        rows, expected_outcomes, strict=True
    ):
        canonical = MigratedCanonical.objects.get(pk=canonical_id)
        link = MigratedLink.objects.get(pk=link_id)
        assert canonical.status == "superseded"
        assert canonical.resolver_version.startswith("legacy-canonical-v0:")
        assert len(canonical.identity_key) == 64
        assert (
            tuple(float(value) for value in canonical.embedding)
            == (float(canonical.metadata["legacy"]),) * 1024
        )
        assert link.status == "superseded"
        assert link.outcome == expected_outcome
        assert link.resolver_version == canonical.resolver_version
        assert link.decision_checksum == canonical_link_decision_checksum(link)

    with pytest.raises(IrreversibleError):
        MigrationExecutor(connection).migrate(before)
    MigrationExecutor(connection).migrate(after)


@pytest.mark.django_db(transaction=True)
@database_required
def test_0004_fails_closed_for_multiple_legacy_active_targets(
    request,
    _restore_latest_knowledge_graph_schema,
):
    from django.db import connection
    from django.db.migrations.executor import MigrationExecutor

    after = [("apps_knowledge_graph", "0004_canonical_identity_audit")]
    old_apps = _migrate_empty_registry_to_0003(connection)
    source_id, _canonical_id, _link_id = _legacy_canonical_graph_row(
        old_apps, index=4, link_status="active", method="stable_identifier"
    )
    LegacyCanonical = old_apps.get_model("apps_knowledge_graph", "CanonicalEntity")
    LegacyLink = old_apps.get_model("apps_knowledge_graph", "CanonicalEntityLink")
    second = LegacyCanonical.objects.create(
        label="Conflicting legacy target",
        normalized_label="conflicting legacy target",
        entity_type="model",
        status="active",
        metadata={},
    )
    conflicting = LegacyLink.objects.create(
        collection_entity_id=source_id,
        canonical_entity_id=second.pk,
        score=0.8,
        method="exact_name_or_alias",
        resolver_version="legacy-resolver-v0",
        status="active",
        reason="legacy conflict",
        metadata={},
    )
    request.addfinalizer(lambda: LegacyLink.objects.filter(pk=conflicting.pk).delete())

    with pytest.raises(RuntimeError, match="multiple active canonical targets"):
        MigrationExecutor(connection).migrate(after)
    LegacyLink.objects.filter(pk=conflicting.pk).delete()
    MigrationExecutor(connection).migrate(after)
