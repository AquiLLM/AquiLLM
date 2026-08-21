"""Django policy adapter for request-local selected-document retrieval."""

from __future__ import annotations

import hmac
from hashlib import sha256
from uuid import UUID

from django.conf import settings
from django.contrib.auth.models import User

from .retrieval_authorization import (
    OpaquePrincipalReference,
    RetrievalAuthorizationContext,
    bind_retrieval_reauthorization_capability,
    freeze_retrieval_authorization_context,
)

_POLICY_VERSION = "django-collection-view-v1"
_POLICY_CHECKSUM = sha256(b"django-collection-view-v1:selected-direct-v1").hexdigest()


def _selected_collection_ids(values) -> tuple[int, ...]:
    if type(values) not in (tuple, list, frozenset):
        raise TypeError("selected collections must be an exact bounded collection")
    normalized = []
    for value in values:
        if type(value) is str and value.isascii() and value.isdecimal():
            value = int(value)
        if type(value) is not int or value < 1:
            raise ValueError("selected collection IDs must be positive integers")
        normalized.append(value)
    result = tuple(sorted(set(normalized)))
    if not result or len(result) > 128:
        raise ValueError("selected collection IDs are empty or exceed their cap")
    return result


def _selected_document_ids(documents, collections: tuple[int, ...]) -> tuple[UUID, ...]:
    if type(documents) not in (tuple, list):
        raise TypeError("selected documents must be an exact tuple or list")
    collection_set = frozenset(collections)
    retained = []
    for document in documents:
        document_id = getattr(document, "id", None)
        collection_id = getattr(document, "collection_id", None)
        if type(document_id) is not UUID or type(collection_id) is not int:
            raise TypeError(
                "selected documents must expose exact persisted coordinates"
            )
        if collection_id in collection_set:
            retained.append(document_id)
    result = tuple(sorted(set(retained), key=lambda value: value.int))
    if not result or len(result) > 10_000:
        raise ValueError("selected document scope is empty or exceeds its cap")
    return result


class DjangoCollectionRetrievalPermissionPolicy:
    """Re-read direct collection VIEW/EDIT/MANAGE permission on every check."""

    policy_version = _POLICY_VERSION
    policy_checksum = _POLICY_CHECKSUM

    def __init__(self, principal_hmac_key: bytes) -> None:
        if type(principal_hmac_key) is not bytes or not principal_hmac_key:
            raise ValueError("principal HMAC key must be nonempty bytes")
        self._principal_hmac_key = principal_hmac_key

    @staticmethod
    def _principal_id(principal: object) -> int:
        if not isinstance(principal, User) or principal.is_authenticated is not True:
            raise TypeError("principal must be an authenticated Django User")
        if type(principal.pk) is not int or principal.pk < 1:
            raise ValueError("principal must have a persisted positive key")
        return principal.pk

    def opaque_principal_reference(self, *, principal, database_alias):
        principal_id = self._principal_id(principal)
        payload = f"{self.policy_version}\0{database_alias}\0{principal_id}".encode()
        return OpaquePrincipalReference(
            hmac.new(self._principal_hmac_key, payload, sha256).hexdigest()
        )

    def current_authorized_document_scope(
        self, *, principal, database_alias, selected_collection_ids
    ):
        from apps.collections.models import CollectionPermission
        from apps.documents.models.document import _get_document_types

        principal_id = self._principal_id(principal)
        authorized = tuple(
            CollectionPermission.objects.using(database_alias)
            .filter(
                user_id=principal_id,
                collection_id__in=selected_collection_ids,
                permission__in=("VIEW", "EDIT", "MANAGE"),
            )
            .order_by("collection_id")
            .values_list("collection_id", flat=True)
        )
        rows = []
        for model in _get_document_types():
            rows.extend(
                model._base_manager.using(database_alias)
                .filter(collection_id__in=authorized)
                .values_list("collection_id", "id")
            )
        return tuple(sorted(rows, key=lambda row: (row[0], row[1].int)))


def _hybrid_enabled() -> bool:
    return bool(
        getattr(settings, "KG_OVERLAY_ENABLED", False) is True
        and getattr(settings, "KG_MEMGRAPH_TRAVERSAL_ENABLED", False) is True
        and (
            getattr(settings, "KG_GRAPH_DIRECT_ENABLED", False) is True
            or getattr(settings, "KG_GRAPH_EXTENDED_ENABLED", False) is True
        )
    )


def build_production_retrieval_authorization_context(
    *,
    principal,
    selected_collection_ids,
    selected_documents,
    database_alias: str = "default",
) -> RetrievalAuthorizationContext | None:
    """Freeze exact current selected scope; return None on any disabled/invalid path."""

    if not _hybrid_enabled():
        return None
    try:
        collections = _selected_collection_ids(selected_collection_ids)
        documents = _selected_document_ids(selected_documents, collections)
        policy = DjangoCollectionRetrievalPermissionPolicy(
            settings.SECRET_KEY.encode("utf-8")
        )
        capability = bind_retrieval_reauthorization_capability(
            principal=principal, policy=policy
        )
        return freeze_retrieval_authorization_context(
            principal=principal,
            database_alias=database_alias,
            policy=policy,
            selected_collection_ids=collections,
            selected_document_ids=documents,
            reauthorization_capability=capability,
        )
    except Exception:
        return None


__all__ = [
    "DjangoCollectionRetrievalPermissionPolicy",
    "build_production_retrieval_authorization_context",
]
