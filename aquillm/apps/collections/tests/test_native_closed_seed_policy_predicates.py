"""Effective ordinary policy predicates bound the private SQL shortcut."""

from contextlib import contextmanager

import pytest
from django.db import connection, models
from django.test.utils import CaptureQueriesContext

from apps.collections.models import CollectionPermission
from apps.collections.services.bounded_retrieval_authorization import (
    bounded_reauthorization_reads,
)
from apps.knowledge_graph.retrieval.extended_seed_repository import (
    ExtendedSeedRepository,
)

from .native_closed_seed_policy_fixtures import (
    native_frame,
    policy_data_counts,
    prepare,
)

pytestmark = pytest.mark.django_db(transaction=True, databases="__all__")


class FalsePredicate(models.Lookup):
    lookup_name = "in"
    prepare_rhs = False

    def as_sql(self, compiler, connection):
        return "FALSE", []


class FalseTransform(models.Transform):
    lookup_name = "in"
    output_field = models.CharField()
    as_sql = FalsePredicate.as_sql

    def get_lookup(self, name):
        return FalsePredicate


@contextmanager
def changed_predicate(monkeypatch, field, name, change):
    target, registry = field, "instance_lookups"
    if change in {"class", "inherited"}:
        target = type(field) if change == "class" else models.Field
        registry = "class_lookups"
    try:
        with monkeypatch.context() as patch:
            patch.setattr(
                target, registry, dict(target.__dict__.get(registry, {})), raising=False
            )
            if change == "implementation":
                patch.setattr(field.get_lookup(name), "as_sql", FalsePredicate.as_sql)
            elif change == "configuration":
                patch.setattr(field.get_lookup(name), "prepare_rhs", False)
            elif change in {"get_lookup", "get_transform", "get_lookups"}:
                patch.setattr(
                    field, change,
                    lambda _: pytest.fail("replacement policy resolver executed"),
                )
            else:
                target.register_lookup(
                    FalseTransform if change == "transform" else FalsePredicate,
                    lookup_name="unused_policy_fix1" if change == "unrelated" else name,
                )
            yield
    finally:
        if registry == "class_lookups":
            target._clear_cached_class_lookups()


def select(frame):
    return frame[1]._native_policy.select(
        authorization=frame[0][3].authorization,
        repository=ExtendedSeedRepository(),
        read_budget=frame[4],
    )


def ordinary_permission_query(frame):
    authorization = frame[0][3].authorization
    return CollectionPermission.objects.filter(
        user_id=authorization.reauthorization_capability._principal.pk,
        user__is_active=True,
        collection_id__in=authorization.selected_collection_ids,
        permission__in=("VIEW", "EDIT", "MANAGE"),
    ).values_list("collection_id", flat=True)


@pytest.mark.parametrize(
    "change",
    ["instance", "class", "inherited", "transform", "implementation", "unrelated"],
)
@pytest.mark.parametrize("selected", [False, True])
def test_effective_permission_predicate_controls_native_policy_selection(
    monkeypatch, change, selected
):
    with native_frame(monkeypatch) as frame:
        authorization = frame[0][3].authorization
        proof = frame[1]._native_policy
        field = CollectionPermission._meta.get_field("permission")
        original_sql, _ = ordinary_permission_query(frame).query.sql_with_params()
        with bounded_reauthorization_reads(authorization, frame[4]):
            if selected:
                assert select(frame) is proof
                proof.require_current()
            before = (frame[4].queries, frame[4].source_rows)
            with changed_predicate(monkeypatch, field, "in", change):
                changed_query = ordinary_permission_query(frame)
                changed_sql, _ = changed_query.query.sql_with_params()
                if change == "unrelated":
                    assert changed_sql == original_sql
                else:
                    assert changed_sql != original_sql and "FALSE" in changed_sql
                with CaptureQueriesContext(connection) as queries:
                    if change == "unrelated":
                        assert select(frame) is proof
                    elif selected:
                        with pytest.raises(ValueError):
                            select(frame)
                        assert proof.failed
                    else:
                        assert select(frame) is None
                assert len(queries) == 0
                assert (frame[4].queries, frame[4].source_rows) == before
                if not selected and change in {"instance", "class", "transform"}:
                    with policy_data_counts() as counts, pytest.raises(ValueError):
                        prepare(frame)
                    assert counts == [0] and frame[1].mapping is None
            if selected and change != "unrelated":
                # Restoring the predicate cannot revive a selected failed proof.
                with CaptureQueriesContext(connection) as queries:
                    with pytest.raises(ValueError):
                        proof.require_current()
                    with pytest.raises(ValueError):
                        select(frame)
                assert not queries
                assert (frame[4].queries, frame[4].source_rows) == before


@pytest.mark.parametrize(
    "change", ["configuration", "get_lookup", "get_transform", "get_lookups"]
)
@pytest.mark.parametrize("selected", [False, True])
def test_policy_predicate_drift_rejects_before_resolver_or_sql(
    monkeypatch, change, selected
):
    with native_frame(monkeypatch) as frame:
        proof = frame[1]._native_policy
        with bounded_reauthorization_reads(frame[0][3].authorization, frame[4]):
            if selected:
                assert select(frame) is proof
            with changed_predicate(
                monkeypatch, CollectionPermission._meta.get_field("permission"),
                "in", change,
            ), CaptureQueriesContext(connection) as queries:
                if selected:
                    with pytest.raises(ValueError):
                        proof.require_current()
                    assert proof.failed
                else:
                    assert select(frame) is None
            assert not queries and frame[4].queries == frame[4].source_rows == 0


def test_every_bypassed_policy_lookup_has_its_own_effective_field_guard(monkeypatch):
    from django.contrib.auth.models import User

    from apps.collections.services.native_closed_seed_policy_dependencies import _MODELS

    predicates = tuple((m, "collection", "in", (7, 9)) for m in _MODELS) + (
        (CollectionPermission, "collection", "in", (7, 9)),
        (CollectionPermission, "user", "exact", 1),
        (CollectionPermission, "permission", "in", ("VIEW",)),
        (User, "is_active", "exact", True),
    )
    with native_frame(monkeypatch) as frame:
        with bounded_reauthorization_reads(frame[0][3].authorization, frame[4]):
            for model, name, lookup, rhs in predicates:
                field = model._meta.get_field(name)
                with changed_predicate(monkeypatch, field, lookup, "instance"):
                    query = model._base_manager.filter(**{f"{name}__{lookup}": rhs})
                    sql, _ = query.query.sql_with_params()
                    assert "FALSE" in sql
                    with CaptureQueriesContext(connection) as queries:
                        assert select(frame) is None
                    assert not queries
            assert select(frame) is frame[1]._native_policy
            frame[1]._native_policy.require_current()
        assert (frame[4].queries, frame[4].source_rows) == (1, 1)
