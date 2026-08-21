"""Closed adapter from provider-neutral topology queries to Memgraph Cypher."""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Mapping
from math import isfinite
from time import monotonic

from apps.knowledge_graph.retrieval.topology import contracts as c
from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError

from .memgraph_driver import MemgraphDriverError
from .memgraph_records import read_bundle
from .topology_request import decode_topology_request
from .topology_results import family_response
from .topology_snapshot import build_projected_topology_snapshot

_MANIFEST_CYPHER = (
    "MATCH (g:CollectionGeneration {generation_key:$generation_key}) "
    "RETURN g.collection_key AS collection_key, "
    "g.generation_key AS generation_key, g.projection_key AS projection_key, "
    "g.artifact_key AS active_artifact_key, g.graph_checksum AS graph_checksum, "
    "g.membership_checksum AS membership_checksum, g.state AS state"
)
_MANIFEST_FIELDS = frozenset(
    {
        "collection_key",
        "generation_key",
        "projection_key",
        "active_artifact_key",
        "graph_checksum",
        "membership_checksum",
        "state",
    }
)


def _failure(code: str):
    if code == "memgraph_authentication_failed":
        return c.TopologyFailureReason.BACKEND_AUTHENTICATION
    if code in {"memgraph_result_limit", "memgraph_result_invalid"}:
        return c.TopologyFailureReason.BACKEND_SCHEMA_MISMATCH
    return c.TopologyFailureReason.BACKEND_UNAVAILABLE


def _selected_matches_bundle(selected, bundle) -> bool:
    marker = bundle.generation
    collection = tuple(
        row
        for row in bundle.artifact_provenance
        if row.scope_type == "collection" and row.scope_key == marker.collection_key
    )
    return (
        len(collection) == 1
        and marker.generation_key == selected.generation_key
        and marker.collection_key == selected.collection_key
        and marker.projection_key == selected.projection_key
        and marker.artifact_key == selected.active_artifact_key
        and marker.schema_version == selected.schema_version
        and marker.projection_version == selected.projection_version
        and marker.identifier_key_version == selected.identifier_key_version
        and marker.membership_epoch == selected.membership_epoch
        and marker.membership_checksum == selected.membership_checksum
        and collection[0].resolver_version == selected.resolver_version
        and collection[0].resolution_config_checksum
        == selected.resolution_config_checksum
        and collection[0].ontology_checksum == selected.ontology_checksum
        and collection[0].embedding_model_signature
        == selected.embedding_model_signature
    )


class _DeadlineProjectionDriver:
    def __init__(self, driver, *, deadline: float, clock):
        self.driver, self.deadline, self.clock = driver, deadline, clock

    def execute_read(self, cypher, parameters, *, timeout_seconds, max_records):
        del timeout_seconds
        remaining = self.deadline - self.clock()
        if remaining <= 0.0:
            raise TimeoutError("projected topology deadline expired")
        return self.driver.execute_read(
            cypher,
            parameters,
            timeout_seconds=remaining,
            max_records=max_records,
        )


class Neo4jProjectedTopologyQueryAdapter:
    """Execute only the four frozen topology query families."""

    def __init__(self, driver, *, clock=monotonic):
        if not callable(getattr(driver, "execute_read", None)):
            raise TypeError("driver must implement the projection read boundary")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._driver, self._clock = driver, clock
        self._cache: OrderedDict[tuple[str, str, str], object] = OrderedDict()

    def _remaining(self, deadline: float) -> float:
        if type(deadline) is not float or not isfinite(deadline) or deadline <= 0.0:
            raise ValueError("deadline must be a finite positive monotonic float")
        remaining = deadline - self._clock()
        if remaining <= 0.0:
            raise TimeoutError("projected topology deadline expired")
        return remaining

    def _decode(self, parameters):
        try:
            return decode_topology_request(parameters)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise TopologyLoadError(
                c.TopologyFailureReason.AUTHORIZATION_CONTEXT_INVALID
            ) from error

    def _execute(self, cypher, parameters, *, deadline, max_records):
        timeout = self._remaining(deadline)
        try:
            return self._driver.execute_read(
                cypher,
                parameters,
                timeout_seconds=timeout,
                max_records=max_records,
            )
        except MemgraphDriverError as error:
            if error.code == "memgraph_timeout":
                raise TimeoutError("projected topology read timed out") from error
            raise TopologyLoadError(_failure(error.code)) from error

    def _manifests(self, ready, *, deadline: float, max_records: int):
        if max_records != len(ready.selected_generations):
            raise TopologyLoadError(c.TopologyFailureReason.BACKEND_SCHEMA_MISMATCH)
        result = []
        for selected in ready.selected_generations:
            rows = self._execute(
                _MANIFEST_CYPHER,
                {"generation_key": selected.generation_key},
                deadline=deadline,
                max_records=1,
            )
            if (
                type(rows) is not tuple
                or len(rows) != 1
                or not isinstance(rows[0], Mapping)
                or set(rows[0]) != _MANIFEST_FIELDS
                or rows[0]["state"] != "ready"
            ):
                raise TopologyLoadError(c.TopologyFailureReason.READINESS_MISMATCH)
            result.append({key: rows[0][key] for key in _MANIFEST_FIELDS - {"state"}})
        return tuple(result)

    def _snapshot(self, ready, seeds, caps, parameters, *, deadline: float):
        cache_key = (
            ready.bundle_checksum,
            parameters["seed_checksum"],
            parameters["caps_json"],
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._remaining(deadline)
            self._cache.move_to_end(cache_key)
            return cached
        deadline_driver = _DeadlineProjectionDriver(
            self._driver, deadline=deadline, clock=self._clock
        )
        bundles = []
        for selected in ready.selected_generations:
            try:
                authorized_documents = tuple(
                    row.document_key
                    for row in ready.authorized_documents
                    if row.generation_key == selected.generation_key
                )
                bounded_parameters = {
                    "seed_keys_csv": ",".join(row.identity_key for row in seeds),
                    "max_depth": caps.max_depth,
                    "authorized_document_keys_csv": ",".join(authorized_documents),
                    "collection_key": selected.collection_key,
                }
                bundle = read_bundle(
                    deadline_driver,
                    generation_key=selected.generation_key,
                    maxima=(
                        caps.max_nodes,
                        caps.max_nodes,
                        max(1, len(authorized_documents)),
                        caps.max_edges,
                        caps.max_edges,
                        caps.max_edges,
                        caps.max_edges,
                        caps.max_nodes * 2,
                        len(authorized_documents) + 1,
                    ),
                    timeout=self._remaining(deadline),
                    reject_full_pages=True,
                    topology_parameters=bounded_parameters,
                )
            except MemgraphDriverError as error:
                if error.code == "memgraph_timeout":
                    raise TimeoutError("projected topology read timed out") from error
                raise TopologyLoadError(_failure(error.code)) from error
            except TimeoutError:
                raise
            except (KeyError, TypeError, ValueError) as error:
                raise TopologyLoadError(
                    c.TopologyFailureReason.BACKEND_SCHEMA_MISMATCH
                ) from error
            if not _selected_matches_bundle(selected, bundle):
                raise TopologyLoadError(
                    c.TopologyFailureReason.BACKEND_PROVENANCE_MISMATCH
                )
            bundles.append(bundle)
        invalid = (
            c.TopologyFailureReason.DIRECT_TOPOLOGY_INVALID
            if caps.branch_kind is c.HybridBranchKind.DIRECT
            else c.TopologyFailureReason.EXTENDED_TOPOLOGY_INVALID
        )
        try:
            snapshot = build_projected_topology_snapshot(
                ready=ready, seeds=seeds, caps=caps, bundles=tuple(bundles)
            )
        except (KeyError, TypeError, ValueError) as error:
            raise TopologyLoadError(invalid) from error
        self._cache[cache_key] = snapshot
        if len(self._cache) > 8:
            self._cache.popitem(last=False)
        return snapshot

    def execute_read(
        self,
        *,
        query: c.TopologyQueryName,
        parameters: Mapping[str, c.TopologyScalar],
        deadline: float,
        max_records: int,
    ) -> tuple[Mapping[str, c.TopologyScalar], ...]:
        if type(query) is not c.TopologyQueryName:
            raise TypeError("query must be an exact TopologyQueryName")
        if type(max_records) is not int or not 1 <= max_records <= 5_000:
            raise ValueError("max_records must be an integer in 1..5000")
        ready, seeds, caps = self._decode(parameters)
        if query is c.TopologyQueryName.GENERATION_MANIFESTS:
            return self._manifests(ready, deadline=deadline, max_records=max_records)
        expected = {
            c.TopologyQueryName.AUTOMATIC_MEMBERSHIPS: caps.max_nodes,
            c.TopologyQueryName.RELATION_TOPOLOGY: caps.max_edges,
            c.TopologyQueryName.EVIDENCE_MENTIONS: 3_000 + caps.max_nodes * 2,
        }[query]
        if max_records != expected:
            raise TopologyLoadError(c.TopologyFailureReason.BACKEND_SCHEMA_MISMATCH)
        snapshot = self._snapshot(ready, seeds, caps, parameters, deadline=deadline)
        return family_response(query, snapshot)


__all__ = ["Neo4jProjectedTopologyQueryAdapter"]
