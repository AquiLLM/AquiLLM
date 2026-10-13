"""Response-bound query cache; raw historical vectors are never hydrated."""

from lib.embeddings.document_binding import (
    restore_query_observation,
    serialize_query_observation,
)
from lib.embeddings.document_provider import observe_query_embedding, prepare_query

from . import rag_cache
from .chunk_search_scope import current_scope
from .representation_contracts import SourceScopeUnavailable


def query_embedding(query):
    scope = current_scope()
    if scope is None or scope.space is None or scope.runtime is None:
        return None
    _, prepared_digest = prepare_query(query, space=scope.space)
    scope.query_digest = prepared_digest
    with scope.database():
        scope.validate()
    key = rag_cache.stable_cache_key(
        "query-observation-v1", scope.space.space_key, prepared_digest
    )
    cached = rag_cache.cache_get(key)
    observation = None
    if cached is not None:
        try:
            restored = restore_query_observation(cached, space=scope.space)
            if restored.prepared_input_sha256 == prepared_digest:
                observation = restored
        except (ValueError, TypeError):
            pass
    if observation is None:
        observation = observe_query_embedding(
            query, space=scope.space, budget=scope.runtime.budget
        )
        if observation.prepared_input_sha256 != prepared_digest:
            raise SourceScopeUnavailable("source_scope_unavailable")
        payload = serialize_query_observation(observation, space=scope.space)
        with scope.database():
            scope.validate()
            if not scope.runtime.budget.can_publish():
                raise SourceScopeUnavailable("source_scope_unavailable")
            scope.runtime.budget.publish(
                lambda: rag_cache.cache_set(
                    key,
                    payload,
                    rag_cache.query_embed_ttl(),
                )
            )
    with scope.database():
        scope.validate()
    scope.query_compatibility = observation.compatibility_key
    return observation
