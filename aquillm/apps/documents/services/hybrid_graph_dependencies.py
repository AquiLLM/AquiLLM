"""Default-off production assembly for projected hybrid graph retrieval."""

from __future__ import annotations

from dataclasses import fields
from functools import lru_cache

from apps.collections.services.retrieval_authorization import (
    revalidate_retrieval_authorization_context,
)
from apps.documents.services.hybrid_graph_authorization import (
    HybridGraphRetrievalDependencies,
    is_exact_authorization_context,
)
from lib.knowledge_graph.retrieval_config import HybridRetrievalSettings


def django_hybrid_retrieval_settings() -> HybridRetrievalSettings:
    from django.conf import settings as django_settings

    return HybridRetrievalSettings(
        **{
            field.name: getattr(django_settings, f"KG_{field.name.upper()}")
            for field in fields(HybridRetrievalSettings)
        }
    )


@lru_cache(maxsize=4)
def _provider_components(settings: HybridRetrievalSettings):
    from apps.knowledge_graph.projection.memgraph_driver import Neo4jMemgraphDriver
    from apps.knowledge_graph.projection.runtime import projection_identifier_codec
    from apps.knowledge_graph.projection.topology_adapter import (
        Neo4jProjectedTopologyQueryAdapter,
    )
    from apps.knowledge_graph.retrieval.topology.memgraph import (
        MemgraphProjectedTopologyLoader,
    )

    driver = Neo4jMemgraphDriver(
        settings.memgraph_uri,
        settings.memgraph_query_username,
        settings.memgraph_query_password.get_secret_value(),
        database=settings.memgraph_database,
    )
    topology = MemgraphProjectedTopologyLoader(
        Neo4jProjectedTopologyQueryAdapter(driver)
    )
    return topology, projection_identifier_codec(settings)


def _production_runtime(*, authorization, settings):
    from apps.knowledge_graph.retrieval.production_runtime import (
        ProductionHybridBranchRuntime,
    )

    topology, codec = _provider_components(settings)
    return ProductionHybridBranchRuntime(
        authorization=authorization,
        settings=settings,
        topology_loader=topology,
        codec=codec,
    )


def build_hybrid_graph_dependencies(
    *, authorization, settings=None, runtime_factory=_production_runtime
) -> HybridGraphRetrievalDependencies | None:
    """Create one request-bound runtime only for an exact enabled capability."""

    if not is_exact_authorization_context(authorization):
        return None
    selected_settings = (
        django_hybrid_retrieval_settings() if settings is None else settings
    )
    traversal = getattr(selected_settings, "memgraph_traversal_enabled", None)
    direct = getattr(selected_settings, "graph_direct_enabled", None)
    extended = getattr(selected_settings, "graph_extended_enabled", None)
    if (
        traversal is not True
        or type(direct) is not bool
        or type(extended) is not bool
        or not (direct or extended)
    ):
        return None
    try:
        current = revalidate_retrieval_authorization_context(
            context=authorization
        )
        if (
            frozenset(current.collection_ids)
            != authorization.selected_collection_ids
            or frozenset(current.document_ids)
            != authorization.selected_document_ids
        ):
            return None
        runtime = runtime_factory(
            authorization=authorization,
            settings=selected_settings,
        )
        return HybridGraphRetrievalDependencies(
            runtime=runtime,
            settings=selected_settings,
            materialize=runtime.materialize,
        )
    except Exception:
        return None


def resolve(overlay_enabled, authorization, provided):
    """Resolve automatic shipping dependencies and whether hybrid owns the path."""

    from django.conf import settings as django_settings

    configured = bool(
        getattr(django_settings, "KG_MEMGRAPH_TRAVERSAL_ENABLED", False)
        and (
            getattr(django_settings, "KG_GRAPH_DIRECT_ENABLED", False)
            or getattr(django_settings, "KG_GRAPH_EXTENDED_ENABLED", False)
        )
    )
    dependencies = provided
    if (
        overlay_enabled
        and configured
        and dependencies is None
        and is_exact_authorization_context(authorization)
    ):
        dependencies = build_hybrid_graph_dependencies(authorization=authorization)
    requested = overlay_enabled and (configured or dependencies is not None)
    return dependencies, requested


__all__ = [
    "build_hybrid_graph_dependencies",
    "django_hybrid_retrieval_settings",
    "resolve",
]
