# ruff: noqa: E501, I001
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite

from apps.knowledge_graph.retrieval.topology.contracts import TopologyCapsV1

from .identifiers import OpaqueProjectionKey, ProjectionIdentifierDomain
from .memgraph_driver import Neo4jMemgraphDriver

# fmt: off
from .records import AutomaticCanonicalMembershipV1, CollectionGraphProjectionBundleV1, ProjectedArtifactProvenanceV1, ProjectedChunkMembershipV1, ProjectedDocumentMembershipV1, ProjectedEntityV1, ProjectedPhysicalRelationV1, ProjectedRelationEvidenceV1, ProjectionCountsV1, ProjectionGenerationManifestV1, ProjectionGenerationMarkerV1, ProjectionLifecycleState
# fmt: on
from .serialization import projection_checksum

_MAX_PAGE = 5_000
_EMPTY_PRIVATE_MAPPING = sha256(b"[]").hexdigest()
# fmt: off
_COUNT_LABELS = tuple(tuple(item.split(":")) for item in "entity_count:ProjectedEntity automatic_membership_count:AutomaticMembership document_count:ProjectedDocument chunk_count:ProjectedChunk relation_count:ProjectedRelation evidence_count:ProjectedEvidence artifact_provenance_count:ArtifactProvenance".split())
_CLOSURE_QUERIES = (
    "MATCH (e:ProjectedEntity {generation_key:$generation_key}) OPTIONAL MATCH (m:AutomaticMembership {generation_key:$generation_key, opaque_key:e.opaque_key}) WITH e,m WHERE m IS NULL RETURN count(e) AS invalid|MATCH (m:AutomaticMembership {generation_key:$generation_key}) OPTIONAL MATCH (e:ProjectedEntity {generation_key:$generation_key, opaque_key:m.entity_key}) WITH m,e WHERE e IS NULL RETURN count(m) AS invalid|MATCH (c:ProjectedChunk {generation_key:$generation_key}) OPTIONAL MATCH (d:ProjectedDocument {generation_key:$generation_key, opaque_key:c.document_key}) WITH c,d WHERE d IS NULL RETURN count(c) AS invalid|MATCH (r:ProjectedRelation {generation_key:$generation_key}) OPTIONAL MATCH (s:ProjectedEntity {generation_key:$generation_key, opaque_key:r.source_entity_key}) OPTIONAL MATCH (t:ProjectedEntity {generation_key:$generation_key, opaque_key:r.target_entity_key}) WITH r,s,t WHERE s IS NULL OR t IS NULL OR r.source_entity_key=r.target_entity_key RETURN count(r) AS invalid|MATCH (e:ProjectedEvidence {generation_key:$generation_key}) OPTIONAL MATCH (r:ProjectedRelation {generation_key:$generation_key, opaque_key:e.relation_key}) OPTIONAL MATCH (c:ProjectedChunk {generation_key:$generation_key, opaque_key:e.chunk_key}) WITH e,r,c WHERE r IS NULL OR c IS NULL OR c.document_key<>e.document_key OR c.chunk_number<>e.chunk_number RETURN count(e) AS invalid"
).split("|")
# fmt: on


@dataclass(frozen=True, slots=True)
class ProjectionValidationV1:
    generation_key: str
    validation_checksum: str
    counts: ProjectionCountsV1
    valid: bool


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


def _manifest(row: object) -> ProjectionGenerationManifestV1:
    if type(row) is not dict:
        raise ValueError("Memgraph manifest row is invalid")
    state = row.get("state")
    # fmt: off
    lifecycle = ProjectionLifecycleState.BUILDING if state == "staging" else ProjectionLifecycleState(state)
    count_names = ("entity_count", "automatic_membership_count", "document_count", "chunk_count", "relation_count", "evidence_count", "artifact_provenance_count")
    counts = ProjectionCountsV1(*(row[name] for name in count_names))
    return ProjectionGenerationManifestV1(row["generation_key"], row["schema_version"], row["projection_version"], row["identifier_key_version"], row["graph_checksum"], row["snapshot_checksum"], row["private_mapping_checksum"], counts, lifecycle)
    # fmt: on


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
        statements = (
            "CREATE INDEX ON :CollectionGeneration(generation_key)",
            "CREATE INDEX ON :ProjectedRecord(generation_key)",
        )
        for statement in statements:
            self._driver.execute_write(statement, {}, timeout_seconds=timeout)

    # fmt: off
    def write_staging_generation(self, *, bundle: CollectionGraphProjectionBundleV1, batch_size: int, timeout_seconds: float) -> None:
        # fmt: on
        if type(bundle) is not CollectionGraphProjectionBundleV1:
            raise TypeError("bundle must be an exact projection bundle")
        size = _size(batch_size, "batch_size")
        timeout = _timeout(timeout_seconds)
        graph_checksum = projection_checksum(bundle)
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
            "graph_checksum": graph_checksum,
            "snapshot_checksum": graph_checksum,
            "private_mapping_checksum": _EMPTY_PRIVATE_MAPPING,
            "state": "staging",
        }
        parameters.update(asdict(bundle.counts))
        marker_query = "MERGE (g:CollectionGeneration {generation_key:$generation_key}) SET g += $marker"
        self._driver.execute_write(
            marker_query.replace("SET g += $marker", self._set_clause("g", parameters)),
            parameters,
            timeout_seconds=timeout,
        )
        # fmt: off
        families = (("ProjectedEntity", bundle.entities, "entity_key"), ("AutomaticMembership", bundle.automatic_memberships, "entity_key"), ("ProjectedDocument", bundle.documents, "document_key"), ("ProjectedChunk", bundle.chunks, "chunk_key"), ("ProjectedRelation", bundle.relations, "relation_key"), ("ProjectedEvidence", bundle.evidence, "evidence_key"), ("ArtifactProvenance", bundle.artifact_provenance, "scope_key"))
        # fmt: on
        for label, records, identity in families:
            for start in range(0, len(records), size):
                for record in records[start : start + size]:
                    values = asdict(record)
                    values["generation_key"] = marker.generation_key
                    opaque_key = values[identity]
                    values["opaque_key"] = opaque_key
                    query = (
                        f"MERGE (n:{label}:ProjectedRecord "
                        "{generation_key:$generation_key, opaque_key:$opaque_key}) "
                        + self._set_clause("n", values)
                    )
                    self._driver.execute_write(query, values, timeout_seconds=timeout)

    @staticmethod
    def _set_clause(alias: str, values: dict[str, object]) -> str:
        # fmt: off
        assignments = ", ".join(f"{alias}.{name} = ${name}" for name in sorted(values) if name not in {"generation_key", "opaque_key"})
        # fmt: on
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
        row = rows[0]
        manifest = row.get("manifest", row)
        return _manifest(dict(manifest))

    def read_generation_records(
        self,
        *,
        generation_key: OpaqueProjectionKey,
        caps: TopologyCapsV1,
        timeout_seconds: float,
    ) -> CollectionGraphProjectionBundleV1:
        value = _key(generation_key)
        if type(caps) is not TopologyCapsV1:
            raise TypeError("caps must be exact")
        timeout = _timeout(timeout_seconds)
        marker_rows = self._driver.execute_read(
            "MATCH (g:CollectionGeneration {generation_key:$generation_key}) RETURN g AS record",
            {"generation_key": value},
            timeout_seconds=timeout,
            max_records=1,
        )
        if len(marker_rows) != 1:
            raise ValueError("generation marker is missing")
        marker = ProjectionGenerationMarkerV1(**dict(marker_rows[0]["record"]))
        # fmt: off
        families = (("ProjectedEntity", ProjectedEntityV1, caps.max_nodes), ("AutomaticMembership", AutomaticCanonicalMembershipV1, caps.max_nodes), ("ProjectedDocument", ProjectedDocumentMembershipV1, caps.max_nodes), ("ProjectedChunk", ProjectedChunkMembershipV1, caps.max_edges), ("ProjectedRelation", ProjectedPhysicalRelationV1, caps.max_edges), ("ProjectedEvidence", ProjectedRelationEvidenceV1, caps.max_edges), ("ArtifactProvenance", ProjectedArtifactProvenanceV1, caps.max_nodes))
        # fmt: on
        loaded = []
        for label, kind, maximum in families:
            rows = self._driver.execute_read(
                f"MATCH (n:{label} {{generation_key:$generation_key}}) RETURN n AS record ORDER BY n.opaque_key",
                {"generation_key": value},
                timeout_seconds=timeout,
                max_records=maximum,
            )
            loaded.append(tuple(kind(**dict(row["record"])) for row in rows))
        # fmt: off
        entities, memberships, documents, chunks, relations, evidence, provenance = loaded
        counts = ProjectionCountsV1(len(entities), len(memberships), len(documents), len(chunks), len(relations), len(evidence), len(provenance))
        return CollectionGraphProjectionBundleV1(marker, entities, memberships, documents, chunks, relations, evidence, provenance, counts)
        # fmt: on

    # fmt: off
    def validate_generation(self, *, expected: ProjectionGenerationManifestV1, timeout_seconds: float) -> ProjectionValidationV1:
        # fmt: on
        if type(expected) is not ProjectionGenerationManifestV1:
            raise TypeError("expected must be an exact manifest")
        observed = self.read_generation_manifest(
            generation_key=self.opaque_generation_key(expected.generation_key),
            timeout_seconds=timeout_seconds,
        )
        timeout = _timeout(timeout_seconds)
        values = {"generation_key": expected.generation_key}
        counts = []
        for _field, label in _COUNT_LABELS:
            rows = self._driver.execute_read(
                f"MATCH (n:{label} {{generation_key:$generation_key}}) RETURN count(n) AS count",
                values,
                timeout_seconds=timeout,
                max_records=1,
            )
            if len(rows) != 1 or type(rows[0].get("count")) is not int:
                raise ValueError("Memgraph generation audit is invalid")
            counts.append(rows[0]["count"])
        closed = True
        for query in _CLOSURE_QUERIES:
            rows = self._driver.execute_read(
                query, values, timeout_seconds=timeout, max_records=1
            )
            if len(rows) != 1 or type(rows[0].get("invalid")) is not int:
                raise ValueError("Memgraph generation audit is invalid")
            closed = closed and rows[0]["invalid"] == 0
        valid = (
            observed == expected
            and tuple(counts) == tuple(asdict(expected.counts).values())
            and closed
        )
        if valid:
            self._driver.execute_write(
                "MATCH (g:CollectionGeneration {generation_key:$generation_key}) WHERE g.state IN ['staging','building'] SET g.validation_checksum=$validation_checksum",
                {**values, "validation_checksum": expected.graph_checksum},
                timeout_seconds=timeout,
            )
        return ProjectionValidationV1(
            generation_key=expected.generation_key,
            validation_checksum=observed.graph_checksum,
            counts=observed.counts,
            valid=valid,
        )

    # fmt: off
    def mark_generation_ready(self, *, generation_key: OpaqueProjectionKey, validation_checksum: str, timeout_seconds: float) -> None:
        # fmt: on
        value = _key(generation_key)
        timeout = _timeout(timeout_seconds)
        rows = self._driver.execute_read(
            "MATCH (g:CollectionGeneration {generation_key:$generation_key}) RETURN g.validation_checksum AS validation_checksum, g.graph_checksum AS graph_checksum, g.state AS state",
            {"generation_key": value},
            timeout_seconds=timeout,
            max_records=1,
        )
        if (
            len(rows) != 1
            or rows[0]
            != {
                "validation_checksum": validation_checksum,
                "graph_checksum": validation_checksum,
                "state": rows[0].get("state"),
            }
            or rows[0]["state"] not in {"staging", "building", "ready"}
        ):
            raise ValueError("generation validation checksum/state mismatch")
        self._driver.execute_write(
            "MATCH (g:CollectionGeneration {generation_key:$generation_key}) WHERE g.graph_checksum=$validation_checksum SET g.state=$state",
            {
                "generation_key": value,
                "validation_checksum": validation_checksum,
                "state": "ready",
            },
            timeout_seconds=timeout,
        )

    def list_generations(
        self,
        *,
        collection_key: OpaqueProjectionKey,
        limit: int,
        timeout_seconds: float,
    ) -> tuple[ProjectionGenerationManifestV1, ...]:
        rows = self._driver.execute_read(
            "MATCH (g:CollectionGeneration {collection_key:$collection_key}) RETURN g AS manifest ORDER BY g.generation_key LIMIT $limit",
            {"collection_key": _key(collection_key), "limit": _size(limit, "limit")},
            timeout_seconds=_timeout(timeout_seconds),
            max_records=limit,
        )
        return tuple(_manifest(dict(row.get("manifest", row))) for row in rows)

    def delete_generation(
        self, *, generation_key: OpaqueProjectionKey, timeout_seconds: float
    ) -> None:
        self._driver.execute_write(
            "MATCH (n {generation_key:$generation_key}) DETACH DELETE n",
            {"generation_key": _key(generation_key)},
            timeout_seconds=_timeout(timeout_seconds),
        )
