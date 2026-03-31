"""Compatibility wrapper for Mem0's Memgraph graph store."""

from __future__ import annotations

import logging
from typing import Any

from mem0.memory.memgraph_memory import MemoryGraph as UpstreamMemoryGraph
from mem0.utils.factory import EmbedderFactory, LlmFactory

try:
    from langchain_memgraph.graphs.memgraph import Memgraph
except ImportError as exc:  # pragma: no cover - exercised in container builds
    raise ImportError(
        "langchain_memgraph is not installed. Please install it using pip install langchain-memgraph"
    ) from exc


logger = logging.getLogger(__name__)


class CompatibleMemgraphMemoryGraph(UpstreamMemoryGraph):
    """Memgraph graph store with vector-index bootstrap fallbacks for newer servers."""

    def __init__(self, config: Any):
        self.config = config
        self.graph = Memgraph(
            self.config.graph_store.config.url,
            self.config.graph_store.config.username,
            self.config.graph_store.config.password,
        )
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            {"enable_embeddings": True},
        )

        self.llm_provider = "openai"
        if self.config.llm and self.config.llm.provider:
            self.llm_provider = self.config.llm.provider
        if self.config.graph_store and self.config.graph_store.llm and self.config.graph_store.llm.provider:
            self.llm_provider = self.config.graph_store.llm.provider

        llm_config = None
        if self.config.graph_store and self.config.graph_store.llm and hasattr(self.config.graph_store.llm, "config"):
            llm_config = self.config.graph_store.llm.config
        elif hasattr(self.config.llm, "config"):
            llm_config = self.config.llm.config
        self.llm = LlmFactory.create(self.llm_provider, llm_config)
        self.user_id = None
        self.threshold = self.config.graph_store.threshold if hasattr(self.config.graph_store, "threshold") else 0.7

        embedding_dims = self.config.embedder.config["embedding_dims"]
        index_info = self._fetch_existing_indexes()

        if not self._vector_index_exists(index_info, "memzero"):
            self._create_vector_index_with_compatibility("memzero", embedding_dims)

        if not self._label_property_index_exists(index_info, "Entity", "user_id"):
            self.graph.query("CREATE INDEX ON :Entity(user_id);")

        if not self._label_index_exists(index_info, "Entity"):
            self.graph.query("CREATE INDEX ON :Entity;")

    def _create_vector_index_with_compatibility(self, index_name: str, embedding_dims: int) -> None:
        queries = [
            (
                "current-docs-json-dimension",
                f'CREATE VECTOR INDEX {index_name} ON :Entity(embedding) WITH CONFIG {{"dimension": {embedding_dims}, "capacity": 1000, "metric": "cos"}};',
            ),
            (
                "current-docs-json-dimensions",
                f'CREATE VECTOR INDEX {index_name} ON :Entity(embedding) WITH CONFIG {{"dimensions": {embedding_dims}, "capacity": 1000, "metric": "cos"}};',
            ),
            (
                "cypher-map-dimension",
                f'CREATE VECTOR INDEX {index_name} ON :Entity(embedding) WITH CONFIG {{dimension: {embedding_dims}, capacity: 1000, metric: "cos"}};',
            ),
            (
                "cypher-map-dimensions",
                f'CREATE VECTOR INDEX {index_name} ON :Entity(embedding) WITH CONFIG {{dimensions: {embedding_dims}, capacity: 1000, metric: "cos"}};',
            ),
            (
                "quantized-dimensions",
                f'CREATE VECTOR INDEX {index_name} ON :Entity(embedding) WITH CONFIG {{"dimensions": {embedding_dims}, "scalar_kind": "f32"}};',
            ),
            (
                "quantized-cypher-map",
                f'CREATE VECTOR INDEX {index_name} ON :Entity(embedding) WITH CONFIG {{dimensions: {embedding_dims}, scalar_kind: "f32"}};',
            ),
        ]

        errors: list[str] = []
        for label, query in queries:
            try:
                self.graph.query(query)
                logger.info("Created Memgraph vector index using compatibility mode %s.", label)
                return
            except Exception as exc:  # pragma: no cover - depends on remote Memgraph version
                errors.append(f"{label}: {exc}")

        raise RuntimeError("Failed to create Memgraph vector index. " + " | ".join(errors))
