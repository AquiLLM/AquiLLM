"""Search request setup helpers, without shared request state."""

from time import perf_counter

from .hybrid_graph_authorization import (
    HybridGraphRetrievalDependencies,
    is_exact_authorization_context,
)


def start_graph(query, authorization, dependencies, enabled):
    if (
        not enabled
        or type(dependencies) is not HybridGraphRetrievalDependencies
        or not is_exact_authorization_context(authorization)
    ):
        return None
    settings = dependencies.settings
    if getattr(settings, "memgraph_traversal_enabled", None) is not True:
        return None
    timeout = getattr(settings, "graph_overall_timeout_ms", None)
    if type(timeout) is not int or not 1 <= timeout <= 5000:
        return None
    from apps.knowledge_graph.retrieval.scheduler import HybridGraphBranchScheduler

    try:
        return HybridGraphBranchScheduler(dependencies.runtime).start(
            query=query,
            authorization=authorization,
            settings=settings,
            deadline=perf_counter() + timeout / 1000,
        )
    except Exception:
        return None


def query_embedding(query):
    from apps.documents.services import rag_cache
    from aquillm.utils import get_embedding
    from lib.embeddings.config import get_local_embed_config

    _base, _key, model = get_local_embed_config()
    vector = rag_cache.get_cached_query_embedding(query, "search_query", model)
    if vector is None:
        vector = get_embedding(query)
        rag_cache.set_cached_query_embedding(query, "search_query", model, vector)
    return vector
