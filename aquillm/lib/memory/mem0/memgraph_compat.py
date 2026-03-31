"""Compatibility wrapper for Mem0's Memgraph graph store."""

from __future__ import annotations

import logging
import math
from os import getenv
import re
import time
from typing import Any

from mem0.memory.memgraph_memory import MemoryGraph as UpstreamMemoryGraph
from mem0.utils.factory import EmbedderFactory, LlmFactory

from .graph_payloads import normalize_extract_entities_arguments, normalize_graph_search_results_payload

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a project dependency in runtime environments
    np = None

try:
    from langchain_memgraph.graphs.memgraph import Memgraph
except ImportError as exc:  # pragma: no cover - exercised in container builds
    raise ImportError(
        "langchain_memgraph is not installed. Please install it using pip install langchain-memgraph"
    ) from exc


logger = logging.getLogger(__name__)
_PLACEHOLDER_ENTITIES = {
    "assistant": "assistant",
    "me": "user",
    "myself": "user",
    "the assistant": "assistant",
    "the user": "user",
    "user": "user",
}
_WEAK_RELATIONSHIPS = {"HAS", "IS", "NAME", "RELATED_TO"}
_RELATION_ALIASES = {
    "likes": "PREFERS",
    "prefers": "PREFERS",
    "profession": "PROFESSION",
    "uses": "USES",
    "work_on": "WORKS_ON",
    "working_on": "WORKS_ON",
    "works_on": "WORKS_ON",
}

_DEFAULT_GRAPH_SEARCH_CANDIDATE_LIMIT = 256
_QUERY_SEED_STOPWORDS = {
    "about",
    "and",
    "are",
    "do",
    "for",
    "from",
    "have",
    "how",
    "into",
    "me",
    "remember",
    "tell",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "use",
    "what",
    "who",
    "with",
    "you",
    "your",
}


def _graph_search_candidate_limit() -> int:
    """Return the maximum number of graph candidates to score per query."""
    raw = (getenv("MEM0_GRAPH_SEARCH_CANDIDATE_LIMIT", "") or "").strip()
    if not raw:
        return _DEFAULT_GRAPH_SEARCH_CANDIDATE_LIMIT
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_GRAPH_SEARCH_CANDIDATE_LIMIT


def _query_seed_terms(node_list: list[str] | None) -> tuple[list[str], list[str]]:
    """Extract exact-name and token seeds from the incoming query node list."""
    seed_names: list[str] = []
    seed_tokens: list[str] = []
    for raw in node_list or []:
        normalized = normalize_entity_name(raw)
        if not normalized:
            continue

        tokens = [
            token
            for token in re.findall(r"[a-z0-9_.+\-]+", normalized)
            if len(token) >= 3 and token not in _QUERY_SEED_STOPWORDS and token not in _PLACEHOLDER_ENTITIES.values()
        ]
        if len(tokens) == 1 and tokens[0] not in seed_names:
            seed_names.append(tokens[0])
        for token in tokens:
            if token not in seed_tokens:
                seed_tokens.append(token)
    return seed_names, seed_tokens


def normalize_entity_name(name: Any) -> str:
    """Normalize entity names for consistent graph filtering and merges."""
    normalized = re.sub(r"\s+", " ", str(name or "").strip()).strip("`'\"")
    normalized = re.sub(r"[^A-Za-z0-9_.+\- ]+", "", normalized)
    normalized = normalized.lower().strip()
    return _PLACEHOLDER_ENTITIES.get(normalized, normalized)


def normalize_relationship_name(relationship: Any) -> str:
    """Normalize relation names to Memgraph-safe uppercase identifiers."""
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", str(relationship or "").strip().lower()).strip("_")
    if not normalized:
        return ""
    return _RELATION_ALIASES.get(normalized, normalized.upper())


def prepare_graph_relation(item: dict[str, Any]) -> dict[str, str] | None:
    """Normalize a relation candidate and drop low-value graph edges."""
    source = normalize_entity_name(item.get("source"))
    relationship = normalize_relationship_name(item.get("relationship"))
    destination = normalize_entity_name(item.get("destination"))

    if not source or not relationship or not destination:
        return None
    if relationship in _WEAK_RELATIONSHIPS:
        return None
    if source == destination:
        return None
    if source in _PLACEHOLDER_ENTITIES.values() and destination in _PLACEHOLDER_ENTITIES.values():
        return None
    return {
        "source": source,
        "relationship": relationship,
        "destination": destination,
    }


class CompatibleMemgraphMemoryGraph(UpstreamMemoryGraph):
    """Memgraph graph store that avoids unsupported vector-index bootstrap DDL."""

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

        for query in (
            "CREATE INDEX ON :Entity(user_id);",
            "CREATE INDEX ON :Entity(name);",
            "CREATE INDEX ON :Entity;",
        ):
            try:
                self.graph.query(query)
            except Exception:
                pass

    def _base_params(self, filters: dict[str, Any]) -> dict[str, Any]:
        params = {"user_id": filters["user_id"]}
        if filters.get("agent_id"):
            params["agent_id"] = filters["agent_id"]
        return params

    def _node_props(self, filters: dict[str, Any], name_param: str | None = None) -> str:
        props: list[str] = []
        if name_param:
            props.append(f"name: ${name_param}")
        props.append("user_id: $user_id")
        if filters.get("agent_id"):
            props.append("agent_id: $agent_id")
        return ", ".join(props)

    def _label_expr(self, entity_type: str) -> str:
        return f":`{entity_type}`:Entity"

    def _normalize_extract_entities_arguments(self, arguments: Any) -> Any:
        return normalize_extract_entities_arguments(arguments)

    def _normalize_search_results_payload(self, search_results: Any) -> Any:
        return normalize_graph_search_results_payload(search_results)

    def _retrieve_nodes_from_data(self, search_results: Any, *args: Any, **kwargs: Any):
        normalized_results = self._normalize_search_results_payload(search_results)
        return super()._retrieve_nodes_from_data(normalized_results, *args, **kwargs)

    def _cosine_similarity(self, lhs: list[float] | None, rhs: list[float] | None) -> float:
        if not lhs or not rhs or len(lhs) != len(rhs):
            return -1.0
        dot = sum(float(a) * float(b) for a, b in zip(lhs, rhs))
        lhs_norm = math.sqrt(sum(float(a) * float(a) for a in lhs))
        rhs_norm = math.sqrt(sum(float(b) * float(b) for b in rhs))
        if lhs_norm == 0.0 or rhs_norm == 0.0:
            return -1.0
        return dot / (lhs_norm * rhs_norm)

    def _run_candidate_query(
        self, filters: dict[str, Any], query: str, params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        return list(self.graph.query(query, params=params))

    def _fetch_candidate_nodes_with_stats(
        self, filters: dict[str, Any], node_list: list[str] | None = None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        props = self._node_props(filters)
        generic_query = f"""
        MATCH (n:Entity {{{props}}})
        WHERE n.embedding IS NOT NULL
        WITH n
        ORDER BY id(n) DESC
        LIMIT $candidate_limit
        RETURN n.name AS name, id(n) AS node_id, n.embedding AS embedding
        """
        candidate_limit = _graph_search_candidate_limit()
        params = self._base_params(filters) | {"candidate_limit": candidate_limit}

        seed_names, seed_tokens = _query_seed_terms(node_list)
        fetch_started_at = time.perf_counter()
        if seed_names or seed_tokens:
            seeded_query = f"""
            MATCH (n:Entity {{{props}}})
            WHERE n.embedding IS NOT NULL
              AND (
                toLower(n.name) IN $seed_names
                OR any(token IN $seed_tokens WHERE toLower(n.name) CONTAINS token)
              )
            WITH n, CASE WHEN toLower(n.name) IN $seed_names THEN 0 ELSE 1 END AS seed_rank
            ORDER BY seed_rank ASC, id(n) DESC
            LIMIT $candidate_limit
            RETURN n.name AS name, id(n) AS node_id, n.embedding AS embedding
            """
            seeded_params = params | {
                "seed_names": seed_names,
                "seed_tokens": seed_tokens,
            }
            seeded = self._run_candidate_query(filters, seeded_query, seeded_params)
            if seeded:
                return seeded, {
                    "strategy": "seeded",
                    "candidate_limit": candidate_limit,
                    "seed_name_count": len(seed_names),
                    "seed_token_count": len(seed_tokens),
                    "fetched_candidate_count": len(seeded),
                    "fetch_ms": (time.perf_counter() - fetch_started_at) * 1000.0,
                }

        generic = self._run_candidate_query(filters, generic_query, params)
        return generic, {
            "strategy": "generic",
            "candidate_limit": candidate_limit,
            "seed_name_count": len(seed_names),
            "seed_token_count": len(seed_tokens),
            "fetched_candidate_count": len(generic),
            "fetch_ms": (time.perf_counter() - fetch_started_at) * 1000.0,
        }

    def _fetch_candidate_nodes(
        self, filters: dict[str, Any], node_list: list[str] | None = None
    ) -> list[dict[str, Any]]:
        candidates, _stats = self._fetch_candidate_nodes_with_stats(filters, node_list=node_list)
        return candidates

    def _nearest_nodes_python(
        self, candidates: list[dict[str, Any]], query_embedding: list[float], threshold: float, limit: int
    ) -> list[dict[str, Any]]:
        scored: list[dict[str, Any]] = []
        for candidate in candidates:
            similarity = self._cosine_similarity(candidate.get("embedding"), query_embedding)
            if similarity >= threshold:
                scored.append(
                    {
                        "name": candidate.get("name"),
                        "node_id": candidate.get("node_id"),
                        "similarity": similarity,
                    }
                )
        scored.sort(key=lambda item: item["similarity"], reverse=True)
        return scored[:limit]

    def _nearest_nodes_with_stats(
        self,
        query_embedding: list[float],
        filters: dict[str, Any],
        threshold: float,
        limit: int,
        candidates: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        candidates = list(candidates or self._fetch_candidate_nodes(filters))
        score_started_at = time.perf_counter()
        candidate_count = len(candidates)
        if not candidates:
            return [], {
                "mode": "empty",
                "candidate_count": 0,
                "valid_candidate_count": 0,
                "passed_threshold_count": 0,
                "score_ms": 0.0,
            }
        if np is None:
            results = self._nearest_nodes_python(candidates, query_embedding, threshold, limit)
            return results, {
                "mode": "python_no_numpy",
                "candidate_count": candidate_count,
                "valid_candidate_count": candidate_count,
                "passed_threshold_count": len(results),
                "score_ms": (time.perf_counter() - score_started_at) * 1000.0,
            }

        try:
            query_vector = np.asarray(query_embedding, dtype=np.float32)
            if query_vector.ndim != 1 or query_vector.size == 0:
                results = self._nearest_nodes_python(candidates, query_embedding, threshold, limit)
                return results, {
                    "mode": "python_invalid_query",
                    "candidate_count": candidate_count,
                    "valid_candidate_count": candidate_count,
                    "passed_threshold_count": len(results),
                    "score_ms": (time.perf_counter() - score_started_at) * 1000.0,
                }

            valid_candidates: list[dict[str, Any]] = []
            valid_embeddings: list[Any] = []
            for candidate in candidates:
                embedding = candidate.get("embedding")
                candidate_vector = np.asarray(embedding, dtype=np.float32)
                if candidate_vector.ndim != 1 or candidate_vector.shape != query_vector.shape:
                    continue
                valid_candidates.append(candidate)
                valid_embeddings.append(candidate_vector)

            if not valid_candidates:
                return [], {
                    "mode": "numpy",
                    "candidate_count": candidate_count,
                    "valid_candidate_count": 0,
                    "passed_threshold_count": 0,
                    "score_ms": (time.perf_counter() - score_started_at) * 1000.0,
                }

            matrix = np.vstack(valid_embeddings)
            query_norm = np.linalg.norm(query_vector)
            if float(query_norm) == 0.0:
                return [], {
                    "mode": "numpy",
                    "candidate_count": candidate_count,
                    "valid_candidate_count": len(valid_candidates),
                    "passed_threshold_count": 0,
                    "score_ms": (time.perf_counter() - score_started_at) * 1000.0,
                }
            candidate_norms = np.linalg.norm(matrix, axis=1)
            nonzero_mask = candidate_norms > 0
            if not np.any(nonzero_mask):
                return [], {
                    "mode": "numpy",
                    "candidate_count": candidate_count,
                    "valid_candidate_count": len(valid_candidates),
                    "passed_threshold_count": 0,
                    "score_ms": (time.perf_counter() - score_started_at) * 1000.0,
                }

            matrix = matrix[nonzero_mask]
            candidate_norms = candidate_norms[nonzero_mask]
            filtered_candidates = [
                candidate for candidate, keep in zip(valid_candidates, nonzero_mask.tolist()) if keep
            ]
            similarities = (matrix @ query_vector) / (candidate_norms * query_norm)
            scored = [
                {
                    "name": candidate.get("name"),
                    "node_id": candidate.get("node_id"),
                    "similarity": float(similarity),
                }
                for candidate, similarity in zip(filtered_candidates, similarities.tolist())
                if float(similarity) >= threshold
            ]
            scored.sort(key=lambda item: item["similarity"], reverse=True)
            limited = scored[:limit]
            return limited, {
                "mode": "numpy",
                "candidate_count": candidate_count,
                "valid_candidate_count": len(filtered_candidates),
                "passed_threshold_count": len(limited),
                "score_ms": (time.perf_counter() - score_started_at) * 1000.0,
            }
        except Exception:
            results = self._nearest_nodes_python(candidates, query_embedding, threshold, limit)
            return results, {
                "mode": "python_error",
                "candidate_count": candidate_count,
                "valid_candidate_count": candidate_count,
                "passed_threshold_count": len(results),
                "score_ms": (time.perf_counter() - score_started_at) * 1000.0,
            }

    def _nearest_nodes(
        self,
        query_embedding: list[float],
        filters: dict[str, Any],
        threshold: float,
        limit: int,
        candidates: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        results, _stats = self._nearest_nodes_with_stats(
            query_embedding,
            filters,
            threshold,
            limit,
            candidates=candidates,
        )
        return results

    def _search_graph_db(self, node_list, filters, limit=100):
        """Search related graph edges using Python-side vector similarity."""
        total_started_at = time.perf_counter()
        node_similarity: dict[int, float] = {}
        candidate_nodes, fetch_stats = self._fetch_candidate_nodes_with_stats(filters, node_list=list(node_list or []))
        score_modes: list[str] = []
        max_valid_candidates = 0
        passed_threshold_total = 0
        score_ms_total = 0.0
        for node in node_list:
            node_embedding = self.embedding_model.embed(node)
            nearest, score_stats = self._nearest_nodes_with_stats(
                node_embedding,
                filters,
                self.threshold,
                limit,
                candidates=candidate_nodes,
            )
            score_modes.append(score_stats.get("mode", "unknown"))
            max_valid_candidates = max(max_valid_candidates, int(score_stats.get("valid_candidate_count", 0)))
            passed_threshold_total += int(score_stats.get("passed_threshold_count", 0))
            score_ms_total += float(score_stats.get("score_ms", 0.0))
            for candidate in nearest:
                node_id = candidate["node_id"]
                similarity = candidate["similarity"]
                node_similarity[node_id] = max(node_similarity.get(node_id, -1.0), similarity)

        if not node_similarity:
            logger.info(
                "Memgraph graph search stats: strategy=%s candidate_limit=%d seed_names=%d "
                "seed_tokens=%d fetched_candidates=%d score_mode=%s valid_candidates=%d "
                "passed_threshold=%d query_nodes=%d fetch_ms=%.2f score_ms=%.2f edge_query_ms=%.2f total_ms=%.2f",
                fetch_stats.get("strategy", "generic"),
                int(fetch_stats.get("candidate_limit", 0)),
                int(fetch_stats.get("seed_name_count", 0)),
                int(fetch_stats.get("seed_token_count", 0)),
                int(fetch_stats.get("fetched_candidate_count", 0)),
                ",".join(sorted(set(score_modes))) if score_modes else "empty",
                max_valid_candidates,
                passed_threshold_total,
                len(list(node_list or [])),
                float(fetch_stats.get("fetch_ms", 0.0)),
                score_ms_total,
                0.0,
                (time.perf_counter() - total_started_at) * 1000.0,
            )
            return []

        edge_started_at = time.perf_counter()
        params = self._base_params(filters) | {"node_ids": list(node_similarity)}
        props = self._node_props(filters)
        outgoing = self.graph.query(
            f"""
            UNWIND $node_ids AS node_id
            MATCH (n:Entity)-[r]->(m:Entity {{{props}}})
            WHERE id(n) = node_id
            RETURN n.name AS source, id(n) AS source_id, type(r) AS relationship,
                   id(r) AS relation_id, m.name AS destination, id(m) AS destination_id
            """,
            params=params,
        )
        incoming = self.graph.query(
            f"""
            UNWIND $node_ids AS node_id
            MATCH (m:Entity {{{props}}})-[r]->(n:Entity)
            WHERE id(n) = node_id
            RETURN m.name AS source, id(m) AS source_id, type(r) AS relationship,
                   id(r) AS relation_id, n.name AS destination, id(n) AS destination_id
            """,
            params=params,
        )
        result_relations = []
        for row in outgoing:
            similarity = node_similarity.get(row.get("source_id"), -1.0)
            row["similarity"] = similarity
            result_relations.append(row)
        for row in incoming:
            similarity = node_similarity.get(row.get("destination_id"), -1.0)
            row["similarity"] = similarity
            result_relations.append(row)
        result_relations.sort(key=lambda item: item.get("similarity", -1.0), reverse=True)
        edge_query_ms = (time.perf_counter() - edge_started_at) * 1000.0
        logger.info(
            "Memgraph graph search stats: strategy=%s candidate_limit=%d seed_names=%d "
            "seed_tokens=%d fetched_candidates=%d score_mode=%s valid_candidates=%d "
            "passed_threshold=%d query_nodes=%d fetch_ms=%.2f score_ms=%.2f edge_query_ms=%.2f total_ms=%.2f",
            fetch_stats.get("strategy", "generic"),
            int(fetch_stats.get("candidate_limit", 0)),
            int(fetch_stats.get("seed_name_count", 0)),
            int(fetch_stats.get("seed_token_count", 0)),
            int(fetch_stats.get("fetched_candidate_count", 0)),
            ",".join(sorted(set(score_modes))) if score_modes else "empty",
            max_valid_candidates,
            passed_threshold_total,
            len(list(node_list or [])),
            float(fetch_stats.get("fetch_ms", 0.0)),
            score_ms_total,
            edge_query_ms,
            (time.perf_counter() - total_started_at) * 1000.0,
        )
        return result_relations[:limit]

    def _search_source_node(self, source_embedding, filters, threshold=0.9):
        nearest = self._nearest_nodes(source_embedding, filters, threshold, 1)
        if not nearest:
            return []
        return [{"id(source_candidate)": nearest[0]["node_id"]}]

    def _search_destination_node(self, destination_embedding, filters, threshold=0.9):
        nearest = self._nearest_nodes(destination_embedding, filters, threshold, 1)
        if not nearest:
            return []
        return [{"id(destination_candidate)": nearest[0]["node_id"]}]

    def _add_entities(self, to_be_added, filters, entity_type_map):
        """Add or merge entities without relying on Memgraph vector index DDL."""
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id")
        results = []

        for item in to_be_added:
            prepared = prepare_graph_relation(item)
            if prepared is None:
                logger.info("Dropping low-value graph relation candidate: %r", item)
                continue

            source = prepared["source"]
            destination = prepared["destination"]
            relationship = prepared["relationship"]

            source_type = entity_type_map.get(item["source"], entity_type_map.get(source, "__User__"))
            destination_type = entity_type_map.get(
                item["destination"],
                entity_type_map.get(destination, "__User__"),
            )
            source_embedding = self.embedding_model.embed(source)
            destination_embedding = self.embedding_model.embed(destination)

            source_match = self._nearest_nodes(source_embedding, filters, self.threshold, 1)
            destination_match = self._nearest_nodes(destination_embedding, filters, self.threshold, 1)

            params = {"user_id": user_id}
            if agent_id:
                params["agent_id"] = agent_id

            if source_match and destination_match:
                params |= {
                    "source_id": source_match[0]["node_id"],
                    "destination_id": destination_match[0]["node_id"],
                }
                cypher = f"""
                MATCH (source:Entity) WHERE id(source) = $source_id
                MATCH (destination:Entity) WHERE id(destination) = $destination_id
                MERGE (source)-[r:{relationship}]->(destination)
                ON CREATE SET r.created_at = timestamp(), r.updated_at = timestamp()
                RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                """
            elif source_match:
                params |= {
                    "source_id": source_match[0]["node_id"],
                    "destination_name": destination,
                    "destination_embedding": destination_embedding,
                }
                cypher = f"""
                MATCH (source:Entity) WHERE id(source) = $source_id
                MERGE (destination{self._label_expr(destination_type)} {{{self._node_props(filters, "destination_name")}}})
                ON CREATE SET destination.created = timestamp()
                SET destination.embedding = $destination_embedding
                MERGE (source)-[r:{relationship}]->(destination)
                ON CREATE SET r.created = timestamp()
                RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                """
            elif destination_match:
                params |= {
                    "destination_id": destination_match[0]["node_id"],
                    "source_name": source,
                    "source_embedding": source_embedding,
                }
                cypher = f"""
                MATCH (destination:Entity) WHERE id(destination) = $destination_id
                MERGE (source{self._label_expr(source_type)} {{{self._node_props(filters, "source_name")}}})
                ON CREATE SET source.created = timestamp()
                SET source.embedding = $source_embedding
                MERGE (source)-[r:{relationship}]->(destination)
                ON CREATE SET r.created = timestamp()
                RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                """
            else:
                params |= {
                    "source_name": source,
                    "destination_name": destination,
                    "source_embedding": source_embedding,
                    "destination_embedding": destination_embedding,
                }
                cypher = f"""
                MERGE (source{self._label_expr(source_type)} {{{self._node_props(filters, "source_name")}}})
                ON CREATE SET source.created = timestamp()
                SET source.embedding = $source_embedding
                MERGE (destination{self._label_expr(destination_type)} {{{self._node_props(filters, "destination_name")}}})
                ON CREATE SET destination.created = timestamp()
                SET destination.embedding = $destination_embedding
                MERGE (source)-[r:{relationship}]->(destination)
                ON CREATE SET r.created = timestamp()
                RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                """

            results.append(self.graph.query(cypher, params=params))

        return results
