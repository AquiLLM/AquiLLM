# ruff: noqa: E501, E701, E702, I001
# fmt: off
"""PostgreSQL-authoritative frozen retrieval authorization context."""
from __future__ import annotations
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Protocol
from uuid import UUID
_KEY = re.compile(r"[0-9a-f]{64}")
_MAX_COLLECTIONS = 128
_MAX_DOCUMENTS = 10_000
class RetrievalPermissionPolicy(Protocol):
    policy_version: str
    policy_checksum: str
    def opaque_principal_reference(
        self, *, principal: object, database_alias: str
    ) -> OpaquePrincipalReference: ...
    def current_authorized_document_scope(
        self,
        *,
        principal: object,
        database_alias: str,
        selected_collection_ids: frozenset[int],
    ) -> Iterable[tuple[int, UUID]]: ...
def _key(value: object, name: str) -> str:
    if type(value) is not str or _KEY.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 value")
    return value
def _token(value: object, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > 128
        or "\x00" in value
    ):
        raise ValueError(f"{name} must be a bounded exact token")
    return value
def _nonserializable(*_args, **_kwargs):
    raise TypeError("retrieval authorization values are not serializable")
@dataclass(frozen=True, slots=True, repr=False)
class OpaquePrincipalReference:
    value: str = field(repr=False)
    def __post_init__(self) -> None:
        _key(self.value, "opaque principal reference")
    def __repr__(self) -> str:
        return "<OpaquePrincipalReference redacted>"
@dataclass(frozen=True, slots=True, repr=False)
class RetrievalReauthorizationCapability:
    _principal: object = field(repr=False)
    _policy: RetrievalPermissionPolicy = field(repr=False)
    def __repr__(self) -> str:
        return "<RetrievalReauthorizationCapability redacted>"
    __reduce__ = _nonserializable
    __reduce_ex__ = _nonserializable
    __getstate__ = _nonserializable
@dataclass(frozen=True, slots=True)
class CurrentAuthorizedScopeV1:
    collection_ids: tuple[int, ...]
    document_ids: tuple[UUID, ...]
    def __post_init__(self) -> None:
        _selected_collections(self.collection_ids)
        _selected_documents(self.document_ids)
        if len(self.collection_ids) > len(self.document_ids):
            raise ValueError("current authorization scope is incoherent")
@dataclass(frozen=True, slots=True, repr=False)
class RetrievalAuthorizationContext:
    principal_reference: OpaquePrincipalReference = field(repr=False)
    database_alias: str
    policy_version: str
    policy_checksum: str = field(repr=False)
    selected_collection_ids: frozenset[int] = field(repr=False)
    selected_document_ids: frozenset[UUID] = field(repr=False)
    authorization_context_signature: str = field(repr=False)
    reauthorization_capability: RetrievalReauthorizationCapability = field(repr=False)
    def __post_init__(self) -> None:
        if type(self.principal_reference) is not OpaquePrincipalReference:
            raise TypeError("principal_reference must be exact")
        _token(self.database_alias, "database_alias")
        _token(self.policy_version, "policy_version")
        _key(self.policy_checksum, "policy_checksum")
        _selected_collections(self.selected_collection_ids)
        _selected_documents(self.selected_document_ids)
        _key(self.authorization_context_signature, "authorization context signature")
        if (
            type(self.reauthorization_capability)
            is not RetrievalReauthorizationCapability
        ):
            raise TypeError("reauthorization_capability must be exact")
    def __repr__(self) -> str:
        return "<RetrievalAuthorizationContext redacted>"
    __str__ = __repr__
    __reduce__ = _nonserializable
    __reduce_ex__ = _nonserializable
    __getstate__ = _nonserializable
def _selected_collections(value: object) -> frozenset[int]:
    if type(value) not in (tuple, frozenset):
        raise TypeError("selected collection IDs must be an exact tuple or frozenset")
    if len(value) > _MAX_COLLECTIONS:
        raise ValueError("selected collection IDs exceed their cap")
    if any(type(item) is not int or item < 1 for item in value):
        raise ValueError("selected collection IDs must be positive exact integers")
    if type(value) is tuple and len(set(value)) != len(value):
        raise ValueError("selected collection IDs must be unique")
    return frozenset(value)
def _selected_documents(value: object) -> frozenset[UUID]:
    if type(value) not in (tuple, frozenset):
        raise TypeError("selected document IDs must be an exact tuple or frozenset")
    if len(value) > _MAX_DOCUMENTS:
        raise ValueError("selected document IDs exceed their cap")
    if any(type(item) is not UUID for item in value):
        raise TypeError("selected document IDs must be exact UUIDs")
    if type(value) is tuple and len(set(value)) != len(value):
        raise ValueError("selected document IDs must be unique")
    return frozenset(value)
def _policy_signature(policy: RetrievalPermissionPolicy) -> tuple[str, str]:
    return (
        _token(getattr(policy, "policy_version", None), "policy_version"),
        _key(getattr(policy, "policy_checksum", None), "policy_checksum"),
    )
def _current_rows(
    *,
    principal: object,
    policy: RetrievalPermissionPolicy,
    database_alias: str,
    selected_collections: frozenset[int],
) -> tuple[tuple[int, UUID], ...]:
    raw = policy.current_authorized_document_scope(
        principal=principal,
        database_alias=database_alias,
        selected_collection_ids=selected_collections,
    )
    rows: list[tuple[int, UUID]] = []
    try:
        for row in raw:
            if (
                type(row) is not tuple
                or len(row) != 2
                or type(row[0]) is not int
                or row[0] < 1
                or type(row[1]) is not UUID
            ):
                raise ValueError("permission policy returned a malformed scope row")
            rows.append(row)
            if len(rows) > _MAX_DOCUMENTS:
                raise ValueError("permission policy scope exceeds its cap")
    except TypeError as error:
        raise ValueError("permission policy scope must be iterable") from error
    if len(set(rows)) != len(rows):
        raise ValueError("permission policy returned duplicate scope rows")
    documents: dict[UUID, int] = {}
    for collection_id, document_id in rows:
        previous = documents.setdefault(document_id, collection_id)
        if previous != collection_id:
            raise ValueError("permission policy returned conflicting document scope")
    return tuple(sorted(rows, key=lambda row: (row[0], row[1].int)))
def _intersect(
    rows: tuple[tuple[int, UUID], ...],
    collections: frozenset[int],
    documents: frozenset[UUID],
) -> CurrentAuthorizedScopeV1:
    retained = tuple(
        row for row in rows if row[0] in collections and row[1] in documents
    )
    return CurrentAuthorizedScopeV1(
        tuple(sorted({row[0] for row in retained})),
        tuple(sorted({row[1] for row in retained}, key=lambda value: value.int)),
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
