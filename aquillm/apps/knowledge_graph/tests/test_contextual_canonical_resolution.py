"""Contextual graph identity regression controls extracted from legacy suites."""
import pytest
from django.core.exceptions import ValidationError

from apps.knowledge_graph.models import CanonicalEntity, CanonicalEntityLink
from test_canonical_resolution import (
    _entity, _component_members, _create_active_collection_entity, database_required,
)


@pytest.mark.parametrize(
    "right_identifier,outcome",
    [
        ("doi:10.1000/a", "automatic"),
        ("doi:10.1000/b", "rejected"),
    ],
)
def test_contextual_canonical_anchors_control_identity(right_identifier, outcome):
    from apps.knowledge_graph.resolution.canonical import resolve_canonical_entities

    result = resolve_canonical_entities(
        (
            _entity(1, 1, "Mercury", identifier="doi:10.1000/a"),
            _entity(2, 2, "Mercury", identifier=right_identifier),
        ),
        identity_policy="contextual_v2",
    )
    assert result.decisions[0].outcome.value == outcome
    assert len(result.components) == (1 if outcome == "automatic" else 2)


def test_contextual_defined_acronyms_across_collections_are_candidates():
    from apps.knowledge_graph.resolution.canonical import resolve_canonical_entities

    result = resolve_canonical_entities(
        tuple(
            _entity(
                i,
                i,
                "RAG",
                acronym_expansions=(("RAG", "Retrieval Augmented Generation"),),
            )
            for i in (1, 2)
        ),
        identity_policy="contextual_v2",
    )
    assert _component_members(result) == ((1,), (2,))
    assert result.decisions[0].outcome.value == "candidate"


@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize("transport", ["orm", "sql"])
@pytest.mark.parametrize("method", ["exact_name_or_alias", "defined_acronym"])
@pytest.mark.parametrize(
    "resolver_version,allowed",
    [
        ("canonical-resolution-v2-contextual", True),
        ("canonical-resolution-v1", False),
        ("canonical-resolution-v2-other", False),
    ],
)
def test_contextual_candidate_methods_are_namespace_fenced_in_orm_and_sql(
    method, resolver_version, allowed, transport
):
    from django.db import IntegrityError, transaction
    from django.db.models import QuerySet

    from apps.collections.models import Collection

    _artifact, source = _create_active_collection_entity(
        Collection.objects.create(name="Synthetic candidate constraint"),
        label="Mercury",
        cluster_digit="a",
    )
    target = CanonicalEntity.objects.create(
        identity_key="b" * 64,
        label="Mercury",
        normalized_label="mercury",
        entity_type="model",
        resolver_version=resolver_version,
    )
    values = dict(
        collection_entity=source,
        canonical_entity=target,
        score=0.5,
        method=method,
        outcome="candidate",
        reason="identity_anchor_required",
        resolver_version=resolver_version,
        status="suppressed",
    )

    def write():
        if transport == "orm":
            return CanonicalEntityLink.objects.create(**values)
        row = CanonicalEntityLink(**values)
        row.prepare_for_persistence()
        with transaction.atomic():
            QuerySet.bulk_create(CanonicalEntityLink.objects.all(), [row])
        return row

    if allowed:
        row = write()
        assert row.outcome == "candidate"
        assert not CanonicalEntityLink.objects.current(
            resolver_version=resolver_version
        ).exists()
    else:
        with pytest.raises(ValidationError if transport == "orm" else IntegrityError):
            write()


@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize("legacy_row", [False, True])
def test_contextual_schema_reverse_is_safe_before_use(legacy_row):
    from django.db import connection
    from django.db.migrations.executor import MigrationExecutor

    retained = None
    if legacy_row:
        from apps.collections.models import Collection

        _artifact, source = _create_active_collection_entity(
            Collection.objects.create(name="Retained legacy audit"),
            label="Mercury",
            cluster_digit="a",
        )
        target = CanonicalEntity.objects.create(
            identity_key="b" * 64,
            label="Mercury",
            normalized_label="mercury",
            entity_type="model",
            resolver_version="canonical-resolution-v1",
        )
        retained = CanonicalEntityLink.objects.create(
            collection_entity=source,
            canonical_entity=target,
            score=1.0,
            method="stable_identifier",
            outcome="automatic",
            reason="legacy",
            resolver_version=target.resolver_version,
        ).pk
    current = MigrationExecutor(connection).loader.graph.leaf_nodes()
    try:
        MigrationExecutor(connection).migrate(
            [("apps_knowledge_graph", "0011_projection_bulk_chunk_fence")]
        )
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
                "WHERE conname = %s",
                ["kg_canonical_link_method_outcome"],
            )
            assert "canonical-resolution-v2-contextual" not in cursor.fetchone()[0]
        if retained is not None:
            assert CanonicalEntityLink.objects.filter(pk=retained).exists()
    finally:
        MigrationExecutor(connection).migrate(current)


@pytest.mark.django_db(transaction=True)
def test_contextual_schema_reverse_refuses_used_candidate_rows():
    from django.db import connection
    from django.db.migrations.exceptions import IrreversibleError
    from django.db.migrations.executor import MigrationExecutor

    from apps.collections.models import Collection

    _artifact, source = _create_active_collection_entity(
        Collection.objects.create(name="Retained V2 audit"),
        label="Mercury",
        cluster_digit="a",
    )
    target = CanonicalEntity.objects.create(
        identity_key="b" * 64,
        label="Mercury",
        normalized_label="mercury",
        entity_type="model",
        resolver_version="canonical-resolution-v2-contextual",
    )
    candidate = CanonicalEntityLink.objects.create(
        collection_entity=source,
        canonical_entity=target,
        score=0.5,
        method="exact_name_or_alias",
        outcome="candidate",
        reason="identity_anchor_required",
        resolver_version=target.resolver_version,
        status="suppressed",
    )
    current = MigrationExecutor(connection).loader.graph.leaf_nodes()
    try:
        with pytest.raises(IrreversibleError, match="candidate"):
            MigrationExecutor(connection).migrate(
                [("apps_knowledge_graph", "0011_projection_bulk_chunk_fence")]
            )
    finally:
        MigrationExecutor(connection).migrate(current)
    assert CanonicalEntityLink.objects.get(pk=candidate.pk).outcome == "candidate"


@pytest.mark.django_db(transaction=True)
@database_required
def test_contextual_registry_candidates_replay_in_distinct_namespace(monkeypatch):
    from apps.collections.models import Collection
    from apps.knowledge_graph.resolution.canonical import rebuild_canonical_registry

    for digit in ("a", "b"):
        _create_active_collection_entity(
            Collection.objects.create(name=f"Synthetic {digit}"),
            label="Mercury",
            cluster_digit=digit,
            resolver_version="collection-resolution-v2-contextual",
        )
    monkeypatch.setenv("KG_IDENTITY_POLICY", "contextual_v2")
    result = rebuild_canonical_registry()
    assert result.resolver_version == "canonical-resolution-v2-contextual"
    candidate = CanonicalEntityLink.objects.get(
        resolver_version=result.resolver_version
    )
    assert candidate.outcome == CanonicalEntityLink.Outcome.CANDIDATE
    assert candidate.status == CanonicalEntityLink.Status.SUPPRESSED
    assert not CanonicalEntityLink.objects.current(
        resolver_version=result.resolver_version
    ).exists()
    repeated = rebuild_canonical_registry()
    assert repeated.created_entity_count == 0
    assert repeated.created_link_count == 0
    assert (
        CanonicalEntityLink.objects.get(pk=candidate.pk).decision_checksum
        == candidate.decision_checksum
    )
    monkeypatch.setenv("KG_IDENTITY_POLICY", "legacy")
    legacy = rebuild_canonical_registry()
    assert legacy.resolver_version == "canonical-resolution-v1"
    assert legacy.created_link_count == 0
    assert (
        CanonicalEntityLink.objects.get(pk=candidate.pk).status
        == CanonicalEntityLink.Status.SUPPRESSED
    )
