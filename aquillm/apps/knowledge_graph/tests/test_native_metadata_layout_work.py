"""Each live native boundary reads field method headers once before resolvers."""

import sys
from collections import Counter
from contextlib import ExitStack

import pytest
from django.db import connections

from apps.collections.tests.native_closed_seed_policy_fixtures import native_frame
from apps.knowledge_graph.models import GraphArtifact
from apps.knowledge_graph.retrieval import (
    native_seed_metadata_eligibility as eligibility,
)
from apps.knowledge_graph.retrieval import native_seed_metadata_layout as layouts

pytestmark = pytest.mark.django_db(transaction=True, databases="__all__")


def test_large_native_checks_scan_field_methods_once_per_live_boundary(monkeypatch):
    with native_frame(monkeypatch, large=True, metadata=True) as frame:
        active, owner, _, groups, budget = frame
        metadata = owner._native_metadata
        assert metadata is not None
        assert len(active[3].requested_document_ids) == 530
        assert len(active[3].authorization.selected_collection_ids) == 31
        metadata.bind(
            group=groups[0], budget=budget, authorization=active[3].authorization
        )
        fields = {id(row[0]) for model in eligibility._LAYOUTS for row in model[7]}
        scans, boundaries = Counter(), Counter()

        def observe(frame, event, arg):
            if event != "call":
                return
            if frame.f_code is layouts._methods.__code__:
                if frame.f_locals["names"] is layouts._FIELDS:
                    scans[id(frame.f_locals["obj"])] += 1
            if frame.f_code is eligibility.supported.__code__:
                boundaries["supported"] += 1

        before = (budget.queries, budget.source_rows, budget.wire_bytes)
        previous = sys.getprofile()
        with ExitStack() as stack:
            for alias in {active[3].using, "projection_source"}:
                stack.enter_context(
                    connections[alias].execute_wrapper(
                        lambda *args: pytest.fail("metadata check executed SQL")
                    )
                )
            try:
                sys.setprofile(observe)
                for _ in range(3):
                    metadata.check()
            finally:
                sys.setprofile(previous)
        assert boundaries == {"supported": 3}
        assert set(scans) == fields
        assert set(scans.values()) == {3}, dict(Counter(scans.values()))
        assert (budget.queries, budget.source_rows, budget.wire_bytes) == before


@pytest.mark.parametrize("resolver", ["get_lookup", "get_transform"])
def test_all_field_method_headers_reject_before_any_resolver(monkeypatch, resolver):
    expected = layouts.layout(GraphArtifact)
    fields = tuple(GraphArtifact._meta.local_concrete_fields)
    changed = fields[-1]
    monkeypatch.setattr(
        changed, resolver, lambda *args: pytest.fail("changed resolver executed")
    )
    original = layouts._proof_lookup
    seen = []

    def lookup(field):
        seen.append(field)
        return original(field)

    monkeypatch.setattr(layouts, "_proof_lookup", lookup)
    assert not layouts.unchanged(expected)
    assert seen == []


@pytest.mark.parametrize("change", ["column", "order", "manager", "lookup"])
def test_invocation_local_headers_preserve_remaining_layout_drift(monkeypatch, change):
    expected = layouts.layout(GraphArtifact)
    if change == "column":
        monkeypatch.setattr(GraphArtifact._meta.get_field("id"), "column", "changed")
    elif change == "order":
        monkeypatch.setattr(
            GraphArtifact._meta,
            "local_concrete_fields",
            tuple(reversed(GraphArtifact._meta.local_concrete_fields)),
        )
    elif change == "manager":
        monkeypatch.setitem(
            GraphArtifact.objects.__dict__, "filter", lambda *args: None
        )
    else:
        kind = GraphArtifact._meta.get_field("status").get_lookup("exact")
        monkeypatch.setattr(kind, "lookup_name", "changed")
    assert not layouts.unchanged(expected)


@pytest.mark.parametrize("change", ["order", "replacement"])
def test_field_shape_changed_after_method_guards_is_still_rejected(monkeypatch, change):
    expected = layouts.layout(GraphArtifact)
    fields = tuple(GraphArtifact._meta.local_concrete_fields)
    original = layouts._methods

    def methods(obj, names):
        result = original(obj, names)
        if obj is fields[-1] and names is layouts._FIELDS:
            current = (
                tuple(reversed(fields))
                if change == "order"
                else (*fields[:-1], fields[0])
            )
            monkeypatch.setattr(GraphArtifact._meta, "local_concrete_fields", current)
        return result

    monkeypatch.setattr(layouts, "_methods", methods)
    assert not layouts.unchanged(expected)


@pytest.mark.parametrize("change", ["order", "replacement"])
def test_same_manager_access_cannot_hide_late_field_shape_change(monkeypatch, change):
    expected = layouts.layout(GraphArtifact)
    fields = tuple(GraphArtifact._meta.local_concrete_fields)
    manager = GraphArtifact.objects
    accesses = []

    class MutatingManager:
        def __get__(self, instance, owner):
            accesses.append(True)
            current = (
                tuple(reversed(fields))
                if change == "order"
                else (*fields[:-1], fields[0])
            )
            monkeypatch.setattr(owner._meta, "local_concrete_fields", current)
            return manager

    monkeypatch.setattr(GraphArtifact, "objects", MutatingManager())
    assert not layouts.unchanged(expected)
    assert accesses == [True]
