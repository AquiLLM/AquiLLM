from __future__ import annotations

import os
from collections.abc import Mapping

from lib.knowledge_graph.retrieval_config import (
    HybridRetrievalSettings,
    load_hybrid_retrieval_settings,
)

from .memgraph_driver import Neo4jMemgraphDriver
from .memgraph_repository import MemgraphProjectionRepository

_PROJECTION_SETTING_NAMES = frozenset(
    (
        "KG_GRAPH_OVERALL_TIMEOUT_MS",
        "KG_MEMGRAPH_DATABASE",
        "KG_MEMGRAPH_PROJECTION_ENABLED",
        "KG_MEMGRAPH_PROJECTION_PASSWORD",
        "KG_MEMGRAPH_PROJECTION_USERNAME",
        "KG_MEMGRAPH_URI",
        "KG_PROJECTION_BATCH_SIZE",
        "KG_PROJECTION_FORMAT_VERSION",
        "KG_PROJECTION_IDENTIFIER_HMAC_KEY",
        "KG_PROJECTION_IDENTIFIER_KEY_VERSION",
        "KG_PROJECTION_LEASE_SECONDS",
        "KG_PROJECTION_MAX_ATTEMPTS",
        "KG_PROJECTION_MAX_LAG_SECONDS",
        "KG_PROJECTION_POSTGRES_SOURCE_DSN",
        "KG_PROJECTION_POSTGRES_STATE_DSN",
        "KG_PROJECTION_QUEUE",
        "KG_PROJECTION_RETENTION",
        "KG_PROJECTION_SCHEMA_VERSION",
    )
)


def load_projection_runtime_settings(
    source: Mapping[str, str] | None = None,
) -> HybridRetrievalSettings:
    values = os.environ if source is None else source
    if not isinstance(values, Mapping):
        raise TypeError("projection configuration source must be a mapping")
    projection_values = {
        key: value for key, value in values.items() if key in _PROJECTION_SETTING_NAMES
    }
    return load_hybrid_retrieval_settings(projection_values)


def memgraph_projection_repository(
    settings: HybridRetrievalSettings,
) -> MemgraphProjectionRepository:
    if type(settings) is not HybridRetrievalSettings:
        raise TypeError("settings must be exact hybrid retrieval settings")
    if not settings.memgraph_projection_enabled:
        raise RuntimeError("memgraph_projection_disabled")
    driver = Neo4jMemgraphDriver(
        settings.memgraph_uri,
        settings.memgraph_projection_username,
        settings.memgraph_projection_password.get_secret_value(),
        database=settings.memgraph_database,
    )
    return MemgraphProjectionRepository(driver)


__all__ = [
    "load_projection_runtime_settings",
    "memgraph_projection_repository",
]
