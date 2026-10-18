"""Actual SQL observation at both existing whole-authority boundaries."""

from time import monotonic

import pytest
from django.db import connection
from django.test.utils import CaptureQueriesContext

from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from apps.knowledge_graph.models import CollectionGraphMembershipCertificate
from apps.knowledge_graph.retrieval.ready_materialization_v2 import (
    validate_captured_selection,
)
from apps.knowledge_graph.retrieval.ready_scope_repository_v2 import (
    select_ready_scope_v2,
)
from apps.knowledge_graph.tests.membership_certificate_serving_support import (
    certified_fixture,
    fingerprint_queries,
    full_membership_queries,
    lifecycle_values,
    publish_certificates,
)
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

pytestmark = pytest.mark.django_db


def test_valid_hit_one_fresh_query_per_authority_boundary_no_full_derivation(
    monkeypatch,
):
    f, _, settings, _ = certified_fixture(monkeypatch)
    with CaptureQueriesContext(connection) as initial:
        selected = select_ready_scope_v2(
            authorization=f.authorization,
            settings=settings,
            representation_snapshot=f.snapshot,
        )
    assert len(selected.groups) == 2
    assert len(fingerprint_queries(initial)) == 1
    assert not full_membership_queries(initial)
    with CaptureQueriesContext(connection) as final:
        validate_captured_selection(
            selected=selected,
            authorization=f.authorization,
            settings=settings,
            codec=f.codec,
        )
    assert len(fingerprint_queries(final)) == 1
    assert not full_membership_queries(final)


@pytest.mark.parametrize(
    "mutation",
    ["source", "generation", "authorization", "vocabulary", "representation"],
)
def test_final_boundary_rejects_changed_current_authority(monkeypatch, mutation):
    from uuid import uuid4

    from apps.knowledge_graph.models import (
        CollectionGraphProjection,
    )

    f, _, settings, _ = certified_fixture(monkeypatch)
    selected = select_ready_scope_v2(
        authorization=f.authorization,
        settings=settings,
        representation_snapshot=f.snapshot,
    )
    if mutation == "source":
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE apps_knowledge_graph_collectionentity "
                "SET status='superseded' WHERE collection_id=7"
            )
    elif mutation == "generation":
        CollectionGraphProjection.objects.filter(pk=f.projections["A"].pk).update(
            generation_key=uuid4()
        )
    elif mutation == "authorization":
        f.policy.rows = ()
    elif mutation == "vocabulary":
        # A newly required vocabulary context cannot inherit the earlier proof.
        monkeypatch.setenv("KG_VOCABULARY_VETO_ENABLED", "1")
    else:
        from apps.documents.services.representation_history import reference
        from apps.documents.services.representation_snapshot_build import (
            build_canonical_snapshot_representation,
        )

        assert (
            build_canonical_snapshot_representation(
                reference(f.documents["A"]), using="default"
            )
            == "activated"
        )
    with pytest.raises((ValueError, RuntimeError)):
        validate_captured_selection(
            selected=selected,
            authorization=f.authorization,
            settings=settings,
            codec=f.codec,
        )


@pytest.mark.parametrize("mode", ["off", "issuance-only", "missing", "bad-signature"])
def test_legacy_or_certificate_miss_retains_full_derivation(monkeypatch, mode):
    f, _, settings, _ = certified_fixture(monkeypatch)
    if mode in {"off", "issuance-only"}:
        settings.membership_certificate_enabled = False
        settings.membership_certificate_issuance_enabled = mode == "issuance-only"
    elif mode == "missing":
        CollectionGraphMembershipCertificate.objects.all().delete()
    else:
        CollectionGraphMembershipCertificate.objects.all().update(signature="0" * 64)
    with CaptureQueriesContext(connection) as queries:
        selected = select_ready_scope_v2(
            authorization=f.authorization,
            settings=settings,
            representation_snapshot=f.snapshot,
        )
    assert len(selected.groups) == 2
    assert not fingerprint_queries(queries)
    assert len(full_membership_queries(queries)) == 2


def test_native_change_after_proof_before_acceptance_rejects(monkeypatch):
    from apps.knowledge_graph.projection import memberships
    from apps.knowledge_graph.retrieval import ready_scope_repository_v2 as repository

    f, _, settings, _ = certified_fixture(monkeypatch)
    original = repository.validate_record_binding

    def replace(record):
        bound = original(record)
        monkeypatch.setattr(
            memberships, "membership_decision_checksum", lambda *a, **kw: "a" * 64
        )
        return bound

    monkeypatch.setattr(repository, "validate_record_binding", replace)
    with pytest.raises(ValueError):
        select_ready_scope_v2(
            authorization=f.authorization,
            settings=settings,
            representation_snapshot=f.snapshot,
        )


def test_real_vocabulary_detachment_revokes_certificate_hit(monkeypatch):
    from apps.knowledge_graph.tests.test_vocabulary_ready_authority import prepared

    f, runtime, pin = prepared(monkeypatch, bound=True)
    f.settings.membership_certificate_enabled = True
    publish_certificates(f)
    with source_runtime_scope(runtime):
        with CaptureQueriesContext(connection) as initial:
            selected = f.select_v2(ready=())
        assert len(fingerprint_queries(initial)) == 1
        assert not full_membership_queries(initial)
        pin.delete()
        with pytest.raises(ValueError):
            validate_captured_selection(
                selected=selected,
                authorization=f.authorization,
                settings=f.settings,
                codec=f.codec,
            )


def test_original_source_deadline_replacement_during_fingerprint_rejects(monkeypatch):
    from apps.documents.services.source_deadline import (
        _DEADLINE,
        source_preparation_scope,
    )

    f, _, settings, _ = certified_fixture(monkeypatch)
    tokens = []

    def replace(execute, sql, params, many, context):
        result = execute(sql, params, many, context)
        if "kg-membership-source-sql-v1:" in sql:
            tokens.append(_DEADLINE.set((monotonic() + 100, monotonic, None)))
        return result

    with source_preparation_scope(monotonic() + 5, monotonic):
        try:
            with connection.execute_wrapper(replace), pytest.raises(TimeoutError):
                select_ready_scope_v2(
                    authorization=f.authorization,
                    settings=settings,
                    representation_snapshot=f.snapshot,
                )
        finally:
            for token in reversed(tokens):
                _DEADLINE.reset(token)


@pytest.mark.parametrize("state", ["ready", "pending", "building"])
def test_present_bad_authority_is_shared_failure_even_with_certificate(
    monkeypatch, state
):
    from apps.knowledge_graph.models import CollectionGraphProjection

    f, _, settings, _ = certified_fixture(monkeypatch)
    values = lifecycle_values(state)
    CollectionGraphProjection.objects.filter(pk=f.projections["A"].pk).update(
        **values, schema_version="unsupported-v99"
    )
    with pytest.raises(ValueError):
        select_ready_scope_v2(
            authorization=f.authorization,
            settings=settings,
            representation_snapshot=f.snapshot,
        )


@pytest.mark.parametrize("state", ["pending", "building"])
def test_nonready_certificate_miss_still_checks_full_membership(monkeypatch, state):
    from apps.knowledge_graph.models import CollectionGraphProjection

    f, _, settings, _ = certified_fixture(monkeypatch)
    CollectionGraphProjection.objects.filter(pk=f.projections["A"].pk).update(
        **lifecycle_values(state)
    )
    with CaptureQueriesContext(connection) as queries:
        selected = select_ready_scope_v2(
            authorization=f.authorization,
            settings=settings,
            representation_snapshot=f.snapshot,
        )
    assert len(selected.groups) == 1
    assert len(fingerprint_queries(queries)) == 1
    assert len(full_membership_queries(queries)) == 1


def test_certified_runtime_publishes_real_graph_candidates(monkeypatch):
    from apps.knowledge_graph.retrieval.ready_materialization_v2 import (
        materialize_graph_candidates_v2,
    )

    f, runtime, settings, _ = certified_fixture(monkeypatch)
    budget = TurnBudget(TurnLimits())
    with source_runtime_scope(SourceRuntime(budget, f.authorization)):
        result = runtime.run_v2(
            query="A",
            authorization=f.authorization,
            representation_snapshot=f.snapshot,
            budget=budget,
            settings=settings,
            deadline=monotonic() + 5,
        )
        assert result.handoff.candidates
        with CaptureQueriesContext(connection) as queries:
            rows = materialize_graph_candidates_v2(
                selected=result.selection,
                refs=result.handoff.candidates,
                authorization=f.authorization,
            )
    assert len(rows) == 2
    assert len(fingerprint_queries(queries)) == 2  # Existing pre/post body boundaries.
    assert not full_membership_queries(queries)
