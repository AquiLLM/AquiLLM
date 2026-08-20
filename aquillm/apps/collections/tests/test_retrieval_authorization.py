from __future__ import annotations

import json
import pickle
from dataclasses import FrozenInstanceError
from uuid import UUID

import pytest

from apps.collections.services.retrieval_authorization import (
    OpaquePrincipalReference,
    _make_test_reauthorization_capability,
    freeze_retrieval_authorization_context,
    revalidate_retrieval_authorization_context,
)

A = UUID("11111111-1111-4111-8111-111111111111")
B = UUID("22222222-2222-4222-8222-222222222222")
C = UUID("33333333-3333-4333-8333-333333333333")


class Policy:
    policy_version = "collection-view-v1"
    policy_checksum = "a" * 64

    def __init__(self) -> None:
        self.rows = ((1, A), (2, B), (1, C))
        self.calls = []

    def opaque_principal_reference(self, *, principal, database_alias):
        assert principal is PRINCIPAL
        assert database_alias == "default"
        return OpaquePrincipalReference("b" * 64)

    def current_authorized_document_scope(
        self, *, principal, database_alias, selected_collection_ids
    ):
        self.calls.append((principal, database_alias, selected_collection_ids))
        return self.rows


PRINCIPAL = object()


def _context(policy: Policy):
    capability = _make_test_reauthorization_capability(
        principal=PRINCIPAL, policy=policy
    )
    return freeze_retrieval_authorization_context(
        principal=PRINCIPAL,
        database_alias="default",
        policy=policy,
        selected_collection_ids=(1, 2),
        selected_document_ids=(A, B),
        reauthorization_capability=capability,
    )


def test_context_is_exact_immutable_redacted_and_nonserializable() -> None:
    context = _context(Policy())
    assert context.selected_collection_ids == frozenset((1, 2))
    assert context.selected_document_ids == frozenset((A, B))
    assert repr(context) == "<RetrievalAuthorizationContext redacted>"
    assert "11111111" not in repr(context) and "bbbb" not in repr(context)
    with pytest.raises(FrozenInstanceError):
        context.database_alias = "other"  # type: ignore[misc]
    with pytest.raises((TypeError, pickle.PicklingError)):
        pickle.dumps(context)
    with pytest.raises(TypeError):
        json.dumps(context)


def test_freeze_and_current_revalidation_intersect_only_selected_scope() -> None:
    policy = Policy()
    context = _context(policy)
    current = revalidate_retrieval_authorization_context(context=context)
    assert current.collection_ids == (1, 2)
    assert current.document_ids == (A, B)
    assert policy.calls[-1][2] == frozenset((1, 2))

    policy.rows = ((1, C), (3, B))  # later grant C ignored; A/B revoked in scope
    revoked = revalidate_retrieval_authorization_context(context=context)
    assert revoked.collection_ids == ()
    assert revoked.document_ids == ()


@pytest.mark.parametrize(
    "changes",
    (
        {"database_alias": " default"},
        {"selected_collection_ids": (True,)},
        {"selected_document_ids": (str(A),)},
        {"selected_document_ids": (A, A)},
    ),
)
def test_freeze_rejects_malformed_exact_scope(changes: dict[str, object]) -> None:
    policy = Policy()
    values = dict(
        principal=PRINCIPAL,
        database_alias="default",
        policy=policy,
        selected_collection_ids=(1,),
        selected_document_ids=(A,),
        reauthorization_capability=_make_test_reauthorization_capability(
            principal=PRINCIPAL, policy=policy
        ),
    )
    values.update(changes)
    with pytest.raises((TypeError, ValueError)):
        freeze_retrieval_authorization_context(**values)  # type: ignore[arg-type]


def test_capability_is_exact_principal_and_policy_bound() -> None:
    first, second = Policy(), Policy()
    capability = _make_test_reauthorization_capability(
        principal=PRINCIPAL, policy=first
    )
    with pytest.raises(ValueError, match="capability"):
        freeze_retrieval_authorization_context(
            principal=PRINCIPAL,
            database_alias="default",
            policy=second,
            selected_collection_ids=(1,),
            selected_document_ids=(A,),
            reauthorization_capability=capability,
        )
