"""Private frozen source/dense lifetime shared by candidate and cache boundaries."""

from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field, replace

from django.conf import settings

from lib.embeddings.document_contracts import digest, require_digest
from lib.embeddings.document_provider import verified_profile

from .hybrid_graph_authorization import is_exact_authorization_context
from .representation_contracts import DocumentRefV1, SourceScopeUnavailable
from .representation_store import (
    active_chunk_scope,
    freeze_active_representations,
    representation_snapshot_digest,
    validate_representation_snapshot,
)
from .source_loading import bounded_source_database, current_source_runtime

_SCOPE = ContextVar("chunk_search_scope", default=None)


def current_scope():
    return _SCOPE.get()


def freeze_dense_generations(*args, **kwargs):
    from .embedding_index_query import freeze_dense_generations as freeze

    return freeze(*args, **kwargs)


def resolve_query_space(using):
    from .embedding_index_store import load_embedding_space

    key = getattr(settings, "DOCUMENT_EMBEDDING_SPACE_KEY", None)
    if not key:
        return None
    require_digest(key)
    space = load_embedding_space(key, using=using)
    profile = verified_profile(space)
    from lib.embeddings.config import get_local_embed_config

    if profile["endpoint"] != get_local_embed_config()[0].rstrip("/"):
        raise ValueError("embedding_space_unavailable")
    return space


@dataclass
class FrozenRetrievalScope:
    representation: object
    dense: object
    space: object
    authorization: object
    using: str
    runtime: object
    requested_document_ids: frozenset = field(default_factory=frozenset)
    bodies: dict = field(default_factory=dict)
    dense_rows: dict = field(default_factory=dict)
    lexical_ids: set = field(default_factory=set)
    query_digest: str = ""
    query_compatibility: str | None = None
    graph_enabled: bool = False

    def database(self):
        return (
            bounded_source_database(self.runtime, self.using)
            if self.runtime
            else nullcontext()
        )

    def validate(self):
        from apps.collections.services.retrieval_authorization import (
            revalidate_retrieval_authorization_context,
        )

        from .embedding_index_query import validate_dense_snapshot

        if self.runtime and not self.runtime.budget.can_publish():
            raise SourceScopeUnavailable("source_scope_unavailable")
        if is_exact_authorization_context(self.authorization):
            current = revalidate_retrieval_authorization_context(
                context=self.authorization
            )
            if any(
                s.document.id not in current.document_ids
                or s.document.collection_id not in current.collection_ids
                for s in self.representation.selections
            ):
                raise SourceScopeUnavailable("source_scope_unavailable")
        validate_representation_snapshot(self.representation, using=self.using)
        keys = ()
        if self.dense is not None:
            keys = validate_dense_snapshot(
                self.representation, self.dense, using=self.using
            )
            try:
                current_space = resolve_query_space(self.using)
            except Exception:
                current_space = None
            if current_space != self.space:
                keys = ()
        if self.runtime and not self.runtime.budget.can_publish():
            raise SourceScopeUnavailable("source_scope_unavailable")
        return keys

    def cache_key(self):
        from .retrieval_scope_identity import retrieval_scope_key

        # The V1 graph route has no captured V2 authority digest. Never cache it.
        if self.graph_enabled or not is_exact_authorization_context(self.authorization):
            return None
        with self.database():
            valid = self.validate()
        if self.dense and set(valid) != set(self.dense.generation_keys):
            return None
        return retrieval_scope_key(
            authorization_digest=self.authorization.authorization_context_signature,
            representation_digest=representation_snapshot_digest(self.representation),
            dense_generation_digest=(
                self.dense.manifest_sha256 if self.dense else "dense-unavailable-v1"
            ),
            embedding_space_key=self.query_compatibility,
            graph_authority_digest="graph-disabled-v1",
            graph_policy_digest="graph-disabled-v1",
            query_digest=self.query_digest,
            ranking_policy=digest(
                {
                    "version": "hybrid-baseline-original-query-v1",
                    "requested_document_ids": sorted(
                        str(v) for v in self.requested_document_ids
                    ),
                }
            ),
            vocabulary_digest="vocabulary-disabled-v1",
        )

    def queryset(self, queryset):
        self.validate()
        return active_chunk_scope(
            queryset.filter(doc_id__in=self.requested_document_ids),
            snapshot=self.representation,
        )


def validate_requested_authorization(runtime, incoming, documents):
    from apps.collections.services.retrieval_authorization import (
        revalidate_retrieval_authorization_context,
    )

    original = runtime.authorization
    if not is_exact_authorization_context(incoming):
        raise SourceScopeUnavailable("source_scope_unavailable")
    try:
        if (
            any(
                getattr(incoming, name) != getattr(original, name)
                for name in (
                    "principal_reference",
                    "database_alias",
                    "policy_version",
                    "policy_checksum",
                )
            )
            or not incoming.selected_document_ids <= original.selected_document_ids
            or (
                not incoming.selected_collection_ids <= original.selected_collection_ids
            )
        ):
            raise ValueError()
        current = revalidate_retrieval_authorization_context(context=incoming)
        if (
            incoming.authorization_context_signature
            != original.authorization_context_signature
        ):
            if (
                frozenset(d.id for d in documents) != incoming.selected_document_ids
                or frozenset(d.collection_id for d in documents)
                != incoming.selected_collection_ids
            ):
                raise ValueError()
        if (
            frozenset(current.document_ids) != incoming.selected_document_ids
            or frozenset(current.collection_ids) != incoming.selected_collection_ids
        ):
            raise ValueError()
    except (TypeError, ValueError, AttributeError):
        raise SourceScopeUnavailable("source_scope_unavailable") from None


def freeze_search_scope(documents, *, using, authorization=None):
    runtime = current_source_runtime()
    authorization = authorization or (runtime.authorization if runtime else None)
    if runtime:
        with bounded_source_database(runtime, using):
            validate_requested_authorization(runtime, authorization, documents)
        authorization = runtime.authorization
    holder = runtime
    old = getattr(holder, "_frozen_retrieval_scope", None)
    ids = frozenset(d.id for d in documents)
    if len(ids) != len(documents):
        raise SourceScopeUnavailable("source_scope_unavailable")
    if old is not None:
        if not ids <= frozenset(s.document.id for s in old.representation.selections):
            raise SourceScopeUnavailable("source_scope_unavailable")
        with old.database():
            old.validate()
        return replace(old, requested_document_ids=ids)
    with bounded_source_database(runtime, using) if runtime else nullcontext():
        refs = tuple(
            DocumentRefV1(
                d._meta.label_lower, d.pkid, d.id, d.collection_id, d.full_text_hash
            )
            for d in documents
        )
        representation = freeze_active_representations(refs, using=using)
        space = None
        if runtime:
            try:
                space = resolve_query_space(using)
            except Exception:
                pass
        dense = (
            freeze_dense_generations(
                representation, space_key=space.space_key, using=using
            )
            if space
            else None
        )
        scope = FrozenRetrievalScope(
            representation,
            dense,
            space,
            authorization,
            using,
            runtime,
            requested_document_ids=ids,
        )
        scope.validate()
    if holder is not None:
        holder._frozen_retrieval_scope = scope
    return scope


def bind_search_scope(scope, query, *, graph_enabled):
    scope.query_digest = digest({"version": "unprepared-query-v1", "query": query})
    scope.graph_enabled = graph_enabled
    return _SCOPE.set(scope)


def reset_search_scope(token):
    if token is not None:
        _SCOPE.reset(token)
