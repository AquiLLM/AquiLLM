"""Fresh authorization tuples are checked with linear per-invocation work."""

from hashlib import sha256
from types import SimpleNamespace
from uuid import UUID

import pytest

from apps.collections.services import retrieval_authorization as authorization
from apps.documents.services import chunk_search_scope as sources
from apps.documents.services import embedding_index_query_scope as dense
from apps.documents.services.representation_contracts import (
    DocumentRefV1,
    RepresentationSelectionV1,
    RepresentationSnapshotV1,
    SourceScopeUnavailable,
)
from apps.documents.services.source_loading import (
    SourcePreparationLimited,
    SourceRuntime,
    source_runtime_scope,
)
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


class Policy:
    policy_version = "fresh-membership-test-v1"
    policy_checksum = "1" * 64

    def __init__(self, rows):
        self.rows, self.calls = rows, 0

    def opaque_principal_reference(self, **_kwargs):
        return authorization.OpaquePrincipalReference("2" * 64)

    def current_authorized_document_scope(self, **_kwargs):
        self.calls += 1
        return self.rows


def frame(monkeypatch, count):
    # Distinct exact UUID objects with equal values prevent tuple identity from
    # hiding equality work. IDs are unique without hash collisions.
    ids = tuple(UUID(int=number + 1) for number in range(count))
    policy, principal = (
        Policy(tuple((1 + number % 2, uid) for number, uid in enumerate(ids))),
        object(),
    )
    capability = authorization.bind_retrieval_reauthorization_capability(
        principal=principal,
        policy=policy,
    )
    context = authorization.freeze_retrieval_authorization_context(
        principal=principal,
        policy=policy,
        database_alias="default",
        selected_collection_ids=(1, 2),
        selected_document_ids=ids,
        reauthorization_capability=capability,
    )
    snapshot = RepresentationSnapshotV1(
        tuple(
            RepresentationSelectionV1(
                DocumentRefV1(
                    "documents.rawtextdocument",
                    number + 1,
                    UUID(int=uid.int),
                    1 + number % 2,
                    sha256(str(number).encode()).hexdigest(),
                ),
                None,
                sha256(str(number).encode()).hexdigest(),
                None,
            )
            for number, uid in enumerate(ids)
        )
    )
    runtime = SourceRuntime(TurnBudget(TurnLimits()), context)
    scope = sources.FrozenRetrievalScope(
        snapshot,
        None,
        None,
        context,
        "default",
        runtime,
        requested_document_ids=frozenset(ids),
    )
    validations = []

    def validate(value, *, using):
        assert value is snapshot and using == "default"
        validations.append(value)
        return value

    monkeypatch.setattr(sources, "validate_representation_snapshot", validate)
    return SimpleNamespace(
        ids=ids,
        policy=policy,
        context=context,
        snapshot=snapshot,
        runtime=runtime,
        scope=scope,
        validations=validations,
    )


def invoke(case, kind):
    if kind == "source":
        return case.scope.validate()
    with source_runtime_scope(case.runtime):
        return dense._authorized_runtime(
            case.snapshot,
            case.runtime.budget,
            "default",
            runtime=case.runtime,
        )


@pytest.mark.parametrize("kind", ["source", "dense"])
def test_530_exact_uuid_checks_do_not_scan_each_fresh_tuple(monkeypatch, kind):
    case = frame(monkeypatch, 530)
    current = authorization.CurrentAuthorizedScopeV1((1, 2), case.ids)
    calls, comparisons = [], []

    def fresh(*, context):
        assert context is case.context
        calls.append(context)
        return current

    original = UUID.__eq__

    def equal(self, other):
        comparisons.append(True)
        return original(self, other)

    target = authorization if kind == "source" else dense
    monkeypatch.setattr(target, "revalidate_retrieval_authorization_context", fresh)
    monkeypatch.setattr(UUID, "__eq__", equal)
    result = invoke(case, kind)
    assert result == (() if kind == "source" else case.runtime)
    assert calls == [case.context]
    assert current.document_ids is case.ids and type(current.document_ids) is tuple
    assert len(comparisons) <= 530 * 3, len(comparisons)


@pytest.mark.parametrize("kind", ["source", "dense"])
@pytest.mark.parametrize("change", ["document", "collection", "permission"])
def test_each_invocation_uses_new_real_policy_result(monkeypatch, kind, change):
    case = frame(monkeypatch, 2)
    before = case.policy.calls
    result = invoke(case, kind)
    assert result == (() if kind == "source" else case.runtime)
    assert case.policy.calls == before + 1
    if change == "document":
        case.policy.rows = ((2, case.ids[1]),)
    elif change == "collection":
        # Both exact UUIDs remain authorized, but original collection2 is absent.
        case.policy.rows = tuple((1, uid) for uid in case.ids)
    else:
        case.policy.rows = ()
    with pytest.raises(SourceScopeUnavailable):
        invoke(case, kind)
    assert case.policy.calls == before + 2
    assert case.runtime.cache == {} and case.scope.bodies == {}
    assert case.runtime.budget.sources_used == 0


@pytest.mark.parametrize("kind", ["source", "dense"])
def test_fresh_scope_is_not_attached_or_mutated_for_later_calls(monkeypatch, kind):
    case = frame(monkeypatch, 2)
    first = authorization.CurrentAuthorizedScopeV1((1, 2), case.ids)
    second = authorization.CurrentAuthorizedScopeV1((2,), (case.ids[1],))
    results, calls = iter((first, second)), []

    def fresh(*, context):
        assert context is case.context
        value = next(results)
        calls.append(value)
        return value

    target = authorization if kind == "source" else dense
    monkeypatch.setattr(target, "revalidate_retrieval_authorization_context", fresh)
    invoke(case, kind)
    with pytest.raises(SourceScopeUnavailable):
        invoke(case, kind)
    assert calls == [first, second]
    assert first.document_ids is case.ids and second.document_ids == (case.ids[1],)
    assert case.scope.authorization is case.runtime.authorization is case.context


@pytest.mark.parametrize("kind", ["source", "dense"])
def test_budget_expiry_during_fresh_read_still_rejects(monkeypatch, kind):
    case = frame(monkeypatch, 2)
    current = authorization.CurrentAuthorizedScopeV1((1, 2), case.ids)
    calls = []

    def fresh(*, context):
        calls.append(context)
        case.runtime.budget.close("cancelled")
        return current

    target = authorization if kind == "source" else dense
    monkeypatch.setattr(target, "revalidate_retrieval_authorization_context", fresh)
    with pytest.raises(
        SourceScopeUnavailable if kind == "source" else SourcePreparationLimited
    ):
        invoke(case, kind)
    assert calls == [case.context]
    assert case.runtime.cache == {} and case.scope.bodies == {}
