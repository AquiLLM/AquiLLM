from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from django.db import transaction

from lib.knowledge_graph.retrieval_config import (
    HybridRetrievalSettings,
    load_hybrid_retrieval_settings,
)

from .django_projection_source import DjangoProjectionRowSource
from .memgraph_driver import Neo4jMemgraphDriver
from .memgraph_repository import MemgraphProjectionRepository
from .postgres_repository import PostgresProjectionRepository
from .state_repository import FunctionProjectionStateRepository

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


def _projection_hook_enabled(source: Mapping[str, str]) -> bool:
    value = source.get("KG_MEMGRAPH_PROJECTION_HOOK_ENABLED", "0")
    if value not in {"0", "1"}:
        raise ValueError(
            "KG_MEMGRAPH_PROJECTION_HOOK_ENABLED must be exactly 0 or 1"
        )
    return value == "1"


@dataclass(frozen=True, slots=True)
class ProjectionDatabaseAliases:
    source: str = "projection_source"
    state: str = "projection_state"

    def __post_init__(self) -> None:
        if (
            type(self.source) is not str
            or not self.source
            or type(self.state) is not str
            or not self.state
            or self.source == self.state
        ):
            raise ValueError("projection database aliases must be distinct tokens")


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


def postgres_projection_repository(
    settings: HybridRetrievalSettings,
    *,
    aliases: ProjectionDatabaseAliases | None = None,
    state_repository=None,
):
    if type(settings) is not HybridRetrievalSettings:
        raise TypeError("settings must be exact hybrid retrieval settings")
    if not settings.memgraph_projection_enabled:
        raise RuntimeError("memgraph_projection_disabled")
    selected = ProjectionDatabaseAliases() if aliases is None else aliases
    if type(selected) is not ProjectionDatabaseAliases:
        raise TypeError("aliases must be exact projection database aliases")
    if state_repository is None:
        raise RuntimeError("function state repository is required")
    if type(state_repository) is not FunctionProjectionStateRepository:
        raise TypeError("state_repository must be an exact function state repository")
    source = DjangoProjectionRowSource(
        selected.source,
        state_using=selected.source,
        identifier_key=settings.projection_identifier_hmac_key.get_secret_value().encode(
            "utf-8"
        ),
        identifier_key_version=settings.projection_identifier_key_version,
        schema_version=settings.projection_schema_version,
        projection_version=settings.projection_format_version,
    )
    return PostgresProjectionRepository(
        using=selected.source,
        source=source,
        chunk_store=state_repository,
    )


def projection_identifier_codec(
    settings: HybridRetrievalSettings,
    *,
    key_version: str | None = None,
):
    from .identifiers import HmacSha256ProjectionIdentifierCodec

    if type(settings) is not HybridRetrievalSettings:
        raise TypeError("settings must be exact hybrid retrieval settings")
    return HmacSha256ProjectionIdentifierCodec(
        settings.projection_identifier_hmac_key.get_secret_value().encode("utf-8"),
        key_version=(
            settings.projection_identifier_key_version
            if key_version is None
            else key_version
        ),
    )


def enqueue_activated_collection_projection(
    collection_id: int,
    artifact_id: int,
    *,
    source: Mapping[str, str] | None = None,
) -> bool:
    """Best-effort optional projection hook inside the web activation transaction."""

    try:
        values = os.environ if source is None else source
        settings = load_projection_runtime_settings(values)
        if not _projection_hook_enabled(values):
            return False
        from .lifecycle import enqueue_collection_projection_locked

        enqueue_collection_projection_locked(
            collection_id=collection_id,
            artifact_id=artifact_id,
            using="default",
            codec=projection_identifier_codec(settings),
        )

        def dispatch_projection_outbox() -> None:
            try:
                from .tasks import reconcile_knowledge_graph_projections

                reconcile_knowledge_graph_projections.delay(
                    collection_id=collection_id
                )
            except Exception:
                # The durable outbox remains pending for a later reconciliation.
                return

        transaction.on_commit(dispatch_projection_outbox, using="default")
    except Exception:
        return False
    return True


def enqueue_automatic_membership_projections(
    collection_ids: tuple[int, ...],
    *,
    using: str,
    source: Mapping[str, str] | None = None,
) -> bool:
    """Best-effort optional membership hook inside the canonical transaction."""

    try:
        settings = load_projection_runtime_settings(source)
        if not settings.memgraph_projection_enabled:
            return False
        from .lifecycle import enqueue_automatic_membership_changes_locked

        enqueue_automatic_membership_changes_locked(
            collection_ids=collection_ids,
            using=using,
            codec=projection_identifier_codec(settings),
        )
    except Exception:
        return False
    return True


__all__ = [
    "ProjectionDatabaseAliases",
    "enqueue_activated_collection_projection",
    "enqueue_automatic_membership_projections",
    "load_projection_runtime_settings",
    "memgraph_projection_repository",
    "postgres_projection_repository",
    "projection_identifier_codec",
]
