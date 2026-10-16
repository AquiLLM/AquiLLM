"""Closed native data witnesses preserve drift rejection without rehashing."""

import sys
from collections import Counter
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
from time import perf_counter_ns

import pytest
from django.db import connections

from apps.collections.tests.native_closed_seed_policy_fixtures import native_frame
from apps.documents.services import representation_store
from apps.knowledge_graph.retrieval import ready_scope_v2
from apps.knowledge_graph.retrieval.topology import contracts

pytestmark = pytest.mark.django_db(transaction=True, databases="__all__")


def test_unchanged_metadata_checks_do_not_reenter_structural_validators(monkeypatch):
    functions = (
        ready_scope_v2.GraphAuthorityGroupV2.__post_init__,
        ready_scope_v2.SelectedGraphScopeV2.__post_init__,
        ready_scope_v2.graph_authority_group_checksum_v2,
        contracts.ReadyGenerationBundleV1.__post_init__,
        contracts.ready_generation_bundle_checksum,
        representation_store.representation_snapshot_digest,
    )
    codes = {f.__code__ for f in functions}
    counts = Counter()
    def observe(frame, event, arg):
        if event == "call" and frame.f_code in codes:
            counts[frame.f_code.co_name] += 1

    with native_frame(monkeypatch, metadata=True) as (_, owner, _, groups, budget):
        metadata = owner._native_metadata
        metadata.bind(group=groups[0], budget=budget,
                      authorization=owner.stage.scope.authorization)
        before = (budget.queries, budget.source_rows, budget.wire_bytes)
        previous = sys.getprofile()
        try:
            sys.setprofile(observe)
            for _ in range(3):
                metadata.check()
        finally:
            sys.setprofile(previous)
        assert not counts, dict(counts)
        assert (budget.queries, budget.source_rows, budget.wire_bytes) == before


class HostileString(str):
    def __eq__(self, other):
        pytest.fail("unapproved equality executed")


class Integer(int):
    pass


@contextmanager
def frozen_changes():
    changes = []
    def change(target, name, value):
        changes.append((target, name, getattr(target, name)))
        object.__setattr__(target, name, value)
    try:
        yield change
    finally:
        for target, name, value in reversed(changes):
            object.__setattr__(target, name, value)


@pytest.mark.parametrize("drift", [
    "generation", "projection_pair", "representation", "selection", "coverage",
    "bool_int", "int_bool", "str_subclass", "node", "container", "order",
    "membership", "equal_node", "equal_container", "uuid", "coherent",
])
def test_selected_data_drift_is_terminal_before_native_read(monkeypatch, drift):
    with native_frame(monkeypatch, metadata=True) as (_, owner, _, groups, budget):
        metadata = owner._native_metadata
        metadata.bind(group=groups[0], budget=budget,
                      authorization=owner.stage.scope.authorization)
        selected = metadata.selected
        group, ready = groups[0], groups[0].scope.ready
        projection = group.scope.projections[0]
        selection = selected.representation_snapshot.selections[0]
        target, field, value = {
            "generation": (ready.selected_generations[0], "resolver_version", "other"),
            "projection_pair": (projection, "documents",
                                ((projection.documents[0][0], 999),)),
            "representation": (selection.document, "source_hash", "f" * 64),
            "selection": (selection, "canonical_sha256", "f" * 64),
            "coverage": (selected.coverage, "status", "partial"),
            "bool_int": (ready.selected_generations[0], "membership_epoch",
                         Integer(ready.selected_generations[0].membership_epoch)),
            "int_bool": (selected.coverage, "exhaustive", 0),
            "str_subclass": (selected, "authorization_context_signature",
                             HostileString(selected.authorization_context_signature)),
            "node": (selection, "document", object()),
            "container": (selected, "groups", list(groups)),
            "order": (selected, "groups", tuple(reversed(groups))),
            "membership": (selected, "groups", ()),
            "equal_node": (selection, "document", replace(selection.document)),
            "equal_container": (selected, "groups", tuple(list(groups))),
            "uuid": (selection.document.id, "int", selection.document.id.int ^ 1),
            "coherent": (selection.document, "source_hash", "f" * 64),
        }[drift]
        before = (budget.queries, budget.source_rows, budget.wire_bytes)
        with ExitStack() as stack, frozen_changes() as change:
            for alias in ("default", "projection_source"):
                stack.enter_context(connections[alias].execute_wrapper(
                    lambda *a: pytest.fail("SQL executed after data drift")))
            change(target, field, value)
            if drift == "coherent":
                change(selection, "canonical_sha256", "f" * 64)
                checksum = ready_scope_v2.graph_authority_group_checksum_v2
                for g in groups:
                    change(g, "group_checksum", checksum(
                        scope=g.scope, ontology_checksum=g.ontology_checksum,
                        representation_snapshot=selected.representation_snapshot))
                ordered = tuple(sorted(groups, key=lambda g: g.group_checksum))
                change(selected, "groups", ordered)
                selected.__post_init__()
            with pytest.raises((TypeError, ValueError, AttributeError)):
                metadata.read("SELECT 1", (), using=metadata.using, maximum=1)
        assert metadata.failed
        with pytest.raises(ValueError):
            metadata.check()
        assert (budget.queries, budget.source_rows, budget.wire_bytes) == before


@pytest.mark.parametrize("entry", [
    "group", "selection", "group_hash", "ready", "ready_hash", "representation_hash",
    "capture", "direct_capture", "compare", "owner_alias", "direct_alias",
])
@pytest.mark.parametrize("postselection", [False, True])
def test_original_entrypoints_reject_replacement_before_execution(
    monkeypatch, entry, postselection
):
    from apps.knowledge_graph.retrieval import native_direct_seed_repository as direct
    from apps.knowledge_graph.retrieval import native_seed_metadata_owner as owners
    from apps.knowledge_graph.retrieval import native_seed_metadata_values as values

    with native_frame(monkeypatch, metadata=True) as (_, owner, _, groups, budget):
        metadata = owner._native_metadata
        metadata.bind(group=groups[0], budget=budget,
                      authorization=owner.stage.scope.authorization)
        target, name = {
            "group": (ready_scope_v2.GraphAuthorityGroupV2, "__post_init__"),
            "selection": (ready_scope_v2.SelectedGraphScopeV2, "__post_init__"),
            "group_hash": (ready_scope_v2, "graph_authority_group_checksum_v2"),
            "ready": (contracts.ReadyGenerationBundleV1, "__post_init__"),
            "ready_hash": (contracts, "ready_generation_bundle_checksum"),
            "representation_hash": (representation_store,
                                    "representation_snapshot_digest"),
            "capture": (values, "capture_selected"),
            "direct_capture": (values, "capture_direct"),
            "compare": (values, "compare"),
            "owner_alias": (owners, "compare"),
            "direct_alias": (direct, "capture_direct"),
        }[entry]
        before = (budget.queries, budget.source_rows, budget.wire_bytes)
        monkeypatch.setattr(target, name,
                            lambda *a, **kw: pytest.fail("replacement executed"))
        if postselection:
            with pytest.raises(ValueError):
                metadata.read("SELECT 1", (), using=metadata.using, maximum=1)
            assert metadata.failed
        else:
            assert owners.native_metadata(owner, metadata.selected) is None
        assert (budget.queries, budget.source_rows, budget.wire_bytes) == before


def test_every_declared_selected_field_and_original_edge_is_bound(monkeypatch):
    from apps.knowledge_graph.retrieval.native_seed_metadata_owner import (
        native_metadata,
    )
    from apps.knowledge_graph.retrieval.native_seed_metadata_values import (
        capture_selected,
        compare,
    )

    with native_frame(monkeypatch, metadata=True) as (_, owner, _, _, budget):
        value = deepcopy(owner._native_metadata.selected)
        witness = capture_selected(value)
        seen, coverage = set(), {}

        def visit(node):
            if id(node) in seen:
                return
            seen.add(id(node))
            if is_dataclass(node):
                coverage[type(node).__name__] = tuple(f.name for f in fields(node))
                for field in fields(node):
                    original = getattr(node, field.name)
                    object.__setattr__(node, field.name, object())
                    with pytest.raises((ValueError, TypeError)):
                        compare(witness, value)
                    object.__setattr__(node, field.name, original)
                    visit(original)
            elif type(node) in (tuple, frozenset):
                for child in node:
                    visit(child)

        visit(value)
        assert len(coverage) == 11
        compare(witness, value)
        document = value.representation_snapshot.selections[0].document
        altered = type("OtherDocument", (type(document),), {})
        with frozen_changes() as change:
            change(document, "__class__", altered)
            with pytest.raises(ValueError):
                compare(witness, value)
        object.__setattr__(value, "groups", list(value.groups))
        assert native_metadata(owner, value) is None
        assert budget.queries == budget.source_rows == 0


@pytest.mark.parametrize("drift", [
    "scope", "tuple", "uuid", "spans", "key", "value", "mapping", "fields",
])
def test_direct_scope_and_span_drift_is_terminal_before_read(monkeypatch, drift):
    from apps.knowledge_graph.retrieval.native_direct_seed_repository import (
        _prepare_direct_native,
    )
    from apps.knowledge_graph.tests.test_native_ontology_proof_postgres import (
        extractor_runtime,
    )

    with native_frame(monkeypatch, metadata=True,
                      prepare_runtime=extractor_runtime) as frame:
        active, owner, _, groups, budget = frame
        metadata = owner._native_metadata
        _prepare_direct_native(active[1], owner=owner, group=groups[0], query="Atlas",
                               deadline=budget.deadline, read_budget=budget)
        repository, scope, _, _ = metadata.direct_inputs[0]
        if drift == "fields":
            from apps.knowledge_graph.retrieval.native_seed_metadata_values import (
                compare,
            )
            for field in fields(scope):
                original = getattr(scope, field.name)
                object.__setattr__(scope, field.name, object())
                with pytest.raises((TypeError, ValueError)):
                    compare(metadata.direct_inputs[0][2], scope, repository._spans)
                object.__setattr__(scope, field.name, original)
            drift = "scope"
        key, value = next(iter(repository._spans.items()))
        before = (budget.queries, budget.source_rows, budget.wire_bytes)
        with monkeypatch.context() as patch, frozen_changes() as change:
            if drift == "scope":
                change(scope, "resolver_version", "changed")
            elif drift == "tuple":
                change(scope, "selected_collection_ids",
                       tuple(list(scope.selected_collection_ids)))
            elif drift == "uuid":
                identifier = scope.generation_ids_by_artifact[0][1]
                change(identifier, "int", identifier.int ^ 1)
            elif drift == "mapping":
                patch.setattr(repository, "_spans", dict(repository._spans))
            elif drift == "key":
                repository._spans.clear()
                patch.setitem(repository._spans, (False, *key[1:]), value)
            else:
                changed = ((value[0], "changed") if drift == "spans"
                           else (False, value[1]))
                patch.setitem(repository._spans, key, changed)
            with pytest.raises(ValueError):
                metadata.read("SELECT 1", (), using=metadata.using, maximum=1)
        assert metadata.failed
        with pytest.raises(ValueError):
            metadata.check()
        assert (budget.queries, budget.source_rows, budget.wire_bytes) == before


def test_ten_genuine_owner_checks_preserve_original_clock_and_ledger(monkeypatch):
    import json

    with native_frame(monkeypatch, metadata=True) as (_, owner, _, groups, budget):
        metadata = owner._native_metadata
        metadata.bind(group=groups[0], budget=budget,
                      authorization=owner.stage.scope.authorization)
        before = (budget.queries, budget.source_rows, budget.wire_bytes)
        bindings = (budget, budget.turn_budget, budget.clock, budget.deadline)
        durations = []
        with ExitStack() as stack:
            for alias in ("default", "projection_source"):
                stack.enter_context(connections[alias].execute_wrapper(
                    lambda *a: pytest.fail("unchanged checker executed SQL")))
            for _ in range(10):
                start = perf_counter_ns()
                metadata.check()
                durations.append((perf_counter_ns() - start) / 1_000_000)
        assert before == (budget.queries, budget.source_rows, budget.wire_bytes)
        assert bindings == (budget, budget.turn_budget, budget.clock, budget.deadline)
        print("TASK14_GENUINE_OWNER=" + json.dumps({"checks": 10,
              "durations_ms": durations, "ledger_before": before,
              "ledger_after": before, "data_control_statements": 0}))

