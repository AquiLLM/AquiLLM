# ruff: noqa: E501
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from math import isfinite

from apps.knowledge_graph.retrieval.topology.contracts import TopologyCapsV1

from .identifiers import OpaqueProjectionKey, ProjectionIdentifierDomain
from .memgraph_driver import Neo4jMemgraphDriver
from .memgraph_records import manifest_from_row, read_bundle
from .memgraph_validation import (
    ProjectionValidationV1,
    mark_ready,
    validate,
)
from .records import (
    CollectionGraphProjectionBundleV1,
    ProjectionGenerationManifestV1,
)
from .serialization import projection_checksum

_MAX_PAGE = 5_000
_EMPTY_PRIVATE_MAPPING = sha256(b"[]").hexdigest()


def _size(value: object, name: str) -> int:
    if type(value) is not int or not 1 <= value <= _MAX_PAGE:
        raise ValueError(f"{name} must be an integer in 1..5000")
    return value


def _timeout(value: object) -> float:
    if type(value) is not float or not isfinite(value) or value <= 0.0:
        raise ValueError("timeout_seconds must be a finite positive float")
    return value


def _key(value: object) -> str:
    if type(value) is not OpaqueProjectionKey:
        raise TypeError("generation key must be an exact opaque key")
    return value.value


class MemgraphProjectionRepository:
    def __init__(self, driver: Neo4jMemgraphDriver) -> None:
        if not hasattr(driver, "execute_read") or not hasattr(driver, "execute_write"):
            raise TypeError("driver must implement bounded Memgraph reads and writes")
        self._driver = driver

    @staticmethod
    def opaque_generation_key(value: str) -> OpaqueProjectionKey:
        return OpaqueProjectionKey(ProjectionIdentifierDomain.COLLECTION, value)

    def ensure_schema(self, *, timeout_seconds: float) -> None:
        timeout = _timeout(timeout_seconds)
        for statement in (
            "CREATE INDEX ON :CollectionGeneration(generation_key)",
            "CREATE INDEX ON :ProjectedRecord(generation_key)",
        ):
            self._driver.execute_write(statement, {}, timeout_seconds=timeout)

    def write_staging_generation(
        self,
        *,
        bundle: CollectionGraphProjectionBundleV1,
        batch_size: int,
        timeout_seconds: float,
    ) -> None:
        if type(bundle) is not CollectionGraphProjectionBundleV1:
            raise TypeError("bundle must be an exact projection bundle")
        size = _size(batch_size, "batch_size")
        timeout = _timeout(timeout_seconds)
        checksum = projection_checksum(bundle)
        marker = bundle.generation
        parameters = {
            "generation_key": marker.generation_key,
            "collection_key": marker.collection_key,
            "artifact_key": marker.artifact_key,
            "schema_version": marker.schema_version,
            "projection_version": marker.projection_version,
            "identifier_key_version": marker.identifier_key_version,
            "membership_epoch": marker.membership_epoch,
            "membership_checksum": marker.membership_checksum,
            "graph_checksum": checksum,
            "snapshot_checksum": checksum,
            "private_mapping_checksum": _EMPTY_PRIVATE_MAPPING,
            "state": "staging",
            **asdict(bundle.counts),
        }
        self._driver.execute_write(
            "MERGE (g:CollectionGeneration {generation_key:$generation_key}) "
            + self._set_clause("g", parameters),
            parameters,
            timeout_seconds=timeout,
        )
        families = (
            ("ProjectedEntity", bundle.entities, "entity_key"),
            ("AutomaticMembership", bundle.automatic_memberships, "entity_key"),
            ("ProjectedDocument", bundle.documents, "document_key"),
            ("ProjectedChunk", bundle.chunks, "chunk_key"),
            ("ProjectedRelation", bundle.relations, "relation_key"),
            ("ProjectedEvidence", bundle.evidence, "evidence_key"),
            ("ArtifactProvenance", bundle.artifact_provenance, "scope_key"),
        )
        for label, records, identity in families:
            for start in range(0, len(records), size):
                for record in records[start : start + size]:
                    values = asdict(record)
                    values["generation_key"] = marker.generation_key
                    values["opaque_key"] = values[identity]
                    self._driver.execute_write(
                        f"MERGE (n:{label}:ProjectedRecord "
                        "{generation_key:$generation_key, opaque_key:$opaque_key}) "
                        + self._set_clause("n", values),
                        values,
                        timeout_seconds=timeout,
                    )

    @staticmethod
    def _set_clause(alias: str, values: dict[str, object]) -> str:
        assignments = ", ".join(
            f"{alias}.{name} = ${name}"
            for name in sorted(values)
            if name not in {"generation_key", "opaque_key"}
        )
        return "SET " + assignments if assignments else ""

    def read_generation_manifest(
        self, *, generation_key: OpaqueProjectionKey, timeout_seconds: float
    ) -> ProjectionGenerationManifestV1:
        value = _key(generation_key)
        rows = self._driver.execute_read(
            "MATCH (g:CollectionGeneration {generation_key:$generation_key}) RETURN g AS manifest",
            {"generation_key": value},
            timeout_seconds=_timeout(timeout_seconds),
            max_records=1,
        )
        if len(rows) != 1:
            raise ValueError("Memgraph generation manifest is missing")
        return manifest_from_row(rows[0])

    def read_generation_records(
        self,
        *,
        generation_key: OpaqueProjectionKey,
        caps: TopologyCapsV1,
        timeout_seconds: float,
    ) -> CollectionGraphProjectionBundleV1:
        if type(caps) is not TopologyCapsV1:
            raise TypeError("caps must be exact")
        maxima = (
            caps.max_nodes,
            caps.max_nodes,
            caps.max_nodes,
            caps.max_edges,
            caps.max_edges,
            caps.max_edges,
            caps.max_nodes,
        )
        return read_bundle(
            self._driver,
            generation_key=_key(generation_key),
            maxima=maxima,
            timeout=_timeout(timeout_seconds),
        )

    def validate_generation(
        self, *, expected: ProjectionGenerationManifestV1, timeout_seconds: float
    ) -> ProjectionValidationV1:
        return validate(
            self,
            expected=expected,
            timeout=_timeout(timeout_seconds),
            empty_private_checksum=_EMPTY_PRIVATE_MAPPING,
        )

    def mark_generation_ready(
        self,
        *,
        generation_key: OpaqueProjectionKey,
        validation_checksum: str,
        timeout_seconds: float,
    ) -> None:
        mark_ready(
            self._driver,
            generation_key=_key(generation_key),
            validation_checksum=validation_checksum,
            timeout=_timeout(timeout_seconds),
            empty_private_checksum=_EMPTY_PRIVATE_MAPPING,
        )

    def list_generations(
        self,
        *,
        collection_key: OpaqueProjectionKey | None = None,
        after_generation_key: OpaqueProjectionKey | None = None,
        limit: int,
        timeout_seconds: float,
    ) -> tuple[ProjectionGenerationManifestV1, ...]:
        size = _size(limit, "limit")
        parameters: dict[str, object] = {"limit": size}
        match = "MATCH (g:CollectionGeneration)"
        if collection_key is not None:
            match = "MATCH (g:CollectionGeneration {collection_key:$collection_key})"
            parameters["collection_key"] = _key(collection_key)
        where = ""
        if after_generation_key is not None:
            where = " WHERE g.generation_key > $after_generation_key"
            parameters["after_generation_key"] = _key(after_generation_key)
        rows = self._driver.execute_read(
            match
            + where
            + " RETURN g AS manifest ORDER BY g.generation_key LIMIT $limit",
            parameters,
            timeout_seconds=_timeout(timeout_seconds),
            max_records=size,
        )
        return tuple(manifest_from_row(row) for row in rows)

    def delete_generation(
        self, *, generation_key: OpaqueProjectionKey, timeout_seconds: float
    ) -> None:
        self._driver.execute_write(
            "MATCH (n {generation_key:$generation_key}) DETACH DELETE n",
            {"generation_key": _key(generation_key)},
            timeout_seconds=_timeout(timeout_seconds),
        )
