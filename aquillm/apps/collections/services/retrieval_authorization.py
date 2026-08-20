"""Freeze and revalidate PostgreSQL-authoritative retrieval authorization."""

from __future__ import annotations

import json
from collections.abc import Iterable
from hashlib import sha256
from uuid import UUID

from .retrieval_authorization_contracts import (
    CurrentAuthorizedScopeV1,
    OpaquePrincipalReference,
    RetrievalAuthorizationContext,
    RetrievalPermissionPolicy,
    RetrievalReauthorizationCapability,
    _current_rows,
    _intersect,
    _policy_signature,
    _selected_collections,
    _selected_documents,
    _token,
)


def _context_signature(
    principal: OpaquePrincipalReference,
    database_alias: str,
    policy_version: str,
    policy_checksum: str,
    collections: frozenset[int],
    documents: frozenset[UUID],
) -> str:
    payload = {
        "collections": sorted(collections),
        "database_alias": database_alias,
        "documents": sorted(str(value) for value in documents),
        "policy_checksum": policy_checksum,
        "policy_version": policy_version,
        "principal_reference": principal.value,
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _make_test_reauthorization_capability(
    *, principal: object, policy: RetrievalPermissionPolicy
) -> RetrievalReauthorizationCapability:
    """Create the exact private capability used by isolated service tests."""

    if principal is None or not callable(
        getattr(policy, "current_authorized_document_scope", None)
    ):
        raise TypeError("test capability requires a principal and permission policy")
    _policy_signature(policy)
    return RetrievalReauthorizationCapability(principal, policy)


def freeze_retrieval_authorization_context(
    *,
    principal: object,
    database_alias: str,
    policy: RetrievalPermissionPolicy,
    selected_collection_ids: Iterable[int],
    selected_document_ids: Iterable[UUID],
    reauthorization_capability: RetrievalReauthorizationCapability,
) -> RetrievalAuthorizationContext:
    """Freeze only the currently authorized part of the user-selected scope."""

    database_alias = _token(database_alias, "database_alias")
    collections = _selected_collections(selected_collection_ids)
    documents = _selected_documents(selected_document_ids)
    if not collections or not documents:
        raise ValueError("selected retrieval scope must not be empty")
    capability = reauthorization_capability
    if (
        type(capability) is not RetrievalReauthorizationCapability
        or capability._principal is not principal
        or capability._policy is not policy
    ):
        raise ValueError("reauthorization capability is not principal/policy bound")
    version, checksum = _policy_signature(policy)
    reference = policy.opaque_principal_reference(
        principal=principal, database_alias=database_alias
    )
    if type(reference) is not OpaquePrincipalReference:
        raise TypeError("permission policy returned an invalid principal reference")
    current = _intersect(
        _current_rows(
            principal=principal,
            policy=policy,
            database_alias=database_alias,
            selected_collections=collections,
        ),
        collections,
        documents,
    )
    frozen_collections = frozenset(current.collection_ids)
    frozen_documents = frozenset(current.document_ids)
    if not frozen_collections or not frozen_documents:
        raise ValueError("selected retrieval scope has no current authorization")
    signature = _context_signature(
        reference,
        database_alias,
        version,
        checksum,
        frozen_collections,
        frozen_documents,
    )
    return RetrievalAuthorizationContext(
        reference,
        database_alias,
        version,
        checksum,
        frozen_collections,
        frozen_documents,
        signature,
        capability,
    )


def revalidate_retrieval_authorization_context(
    *, context: RetrievalAuthorizationContext
) -> CurrentAuthorizedScopeV1:
    """Return current permission intersected with the immutable selected scope."""

    if type(context) is not RetrievalAuthorizationContext:
        raise TypeError("context must be an exact RetrievalAuthorizationContext")
    expected_signature = _context_signature(
        context.principal_reference,
        context.database_alias,
        context.policy_version,
        context.policy_checksum,
        context.selected_collection_ids,
        context.selected_document_ids,
    )
    if context.authorization_context_signature != expected_signature:
        raise ValueError("authorization context signature changed")
    capability = context.reauthorization_capability
    version, checksum = _policy_signature(capability._policy)
    if (version, checksum) != (context.policy_version, context.policy_checksum):
        raise ValueError("permission policy signature changed")
    rows = _current_rows(
        principal=capability._principal,
        policy=capability._policy,
        database_alias=context.database_alias,
        selected_collections=context.selected_collection_ids,
    )
    return _intersect(
        rows, context.selected_collection_ids, context.selected_document_ids
    )


__all__ = [
    "CurrentAuthorizedScopeV1",
    "OpaquePrincipalReference",
    "RetrievalAuthorizationContext",
    "RetrievalPermissionPolicy",
    "RetrievalReauthorizationCapability",
    "freeze_retrieval_authorization_context",
    "revalidate_retrieval_authorization_context",
]
