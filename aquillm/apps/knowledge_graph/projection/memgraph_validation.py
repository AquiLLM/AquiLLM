from __future__ import annotations

from dataclasses import asdict, dataclass

from .memgraph_edge_validation import (
    validate_topology_edges,
    validate_topology_marker,
)
from .memgraph_edges import topology_edge_attestation
from .memgraph_records import read_bundle
from .records import (
    ProjectionCountsV1,
    ProjectionGenerationManifestV1,
    ProjectionLifecycleState,
)
from .serialization import projection_checksum

_MAX_VALIDATED_FAMILY = 4_999


@dataclass(frozen=True, slots=True)
class ProjectionValidationV1:
    generation_key: str
    validation_checksum: str
    counts: ProjectionCountsV1
    valid: bool


def _invalid(expected, observed, checksum=None, counts=None):
    return ProjectionValidationV1(
        expected.generation_key,
        observed.graph_checksum if checksum is None else checksum,
        observed.counts if counts is None else counts,
        False,
    )


def validate(repository, *, expected, timeout, empty_private_checksum):
    if type(expected) is not ProjectionGenerationManifestV1:
        raise TypeError("expected must be an exact manifest")
    observed = repository.read_generation_manifest(
        generation_key=repository.opaque_generation_key(expected.generation_key),
        timeout_seconds=timeout,
    )
    count_values = tuple(asdict(expected.counts).values())
    if any(value > _MAX_VALIDATED_FAMILY for value in count_values):
        raise ValueError("expected projection count exceeds validation cap")
    try:
        bundle = read_bundle(
            repository._driver,
            generation_key=expected.generation_key,
            maxima=tuple(maximum + 1 for maximum in count_values),
            timeout=timeout,
        )
        checksum = projection_checksum(bundle)
    except (KeyError, TypeError, ValueError):
        return _invalid(expected, observed)
    marker = bundle.generation
    core_matches = (
        observed.generation_key,
        observed.schema_version,
        observed.projection_version,
        observed.identifier_key_version,
        observed.graph_checksum,
        observed.snapshot_checksum,
        observed.counts,
        observed.state,
    ) == (
        expected.generation_key,
        expected.schema_version,
        expected.projection_version,
        expected.identifier_key_version,
        expected.graph_checksum,
        expected.snapshot_checksum,
        expected.counts,
        expected.state,
    )
    marker_matches = (
        marker.generation_key,
        marker.schema_version,
        marker.projection_version,
        marker.identifier_key_version,
    ) == (
        expected.generation_key,
        expected.schema_version,
        expected.projection_version,
        expected.identifier_key_version,
    )
    private_valid = (
        observed.private_mapping_checksum == expected.private_mapping_checksum
        and (
            (
                expected.counts.chunk_count == 0
                and expected.private_mapping_checksum == empty_private_checksum
            )
            or (
                expected.counts.chunk_count > 0
                and expected.private_mapping_checksum != empty_private_checksum
            )
        )
    )
    valid = (
        core_matches
        and marker_matches
        and private_valid
        and bundle.counts == expected.counts
        and checksum == expected.graph_checksum == expected.snapshot_checksum
    )
    if valid:
        try:
            valid = validate_topology_edges(
                repository._driver,
                bundle,
                timeout_seconds=timeout,
            )
        except (KeyError, TypeError, ValueError):
            valid = False
    if not valid:
        return _invalid(expected, observed, checksum, bundle.counts)
    if expected.state is ProjectionLifecycleState.READY:
        return ProjectionValidationV1(
            expected.generation_key,
            checksum,
            bundle.counts,
            True,
        )
    values = {
        "generation_key": expected.generation_key,
        "validation_checksum": checksum,
        "private_mapping_checksum": expected.private_mapping_checksum,
        "validated_private_mapping_checksum": expected.private_mapping_checksum,
        "validated_topology_checksum": topology_edge_attestation(bundle).checksum,
    }
    repository._driver.execute_write(
        "MATCH (g:CollectionGeneration {generation_key:$generation_key}) "
        "WHERE g.state IN ['staging','building'] "
        "AND g.private_mapping_checksum=$private_mapping_checksum "
        "SET g.validation_checksum=$validation_checksum, "
        "g.validated_private_mapping_checksum=$validated_private_mapping_checksum, "
        "g.validated_topology_checksum=$validated_topology_checksum",
        values,
        timeout_seconds=timeout,
    )
    refreshed = repository.read_generation_manifest(
        generation_key=repository.opaque_generation_key(expected.generation_key),
        timeout_seconds=timeout,
    )
    return ProjectionValidationV1(
        expected.generation_key,
        checksum,
        bundle.counts,
        refreshed == expected,
    )


def mark_ready(
    driver,
    *,
    generation_key,
    validation_checksum,
    timeout,
    empty_private_checksum,
):
    row = _ready_state(driver, generation_key, timeout)
    private_checksum = row.get("private_mapping_checksum")
    private_checksum_valid = (
        type(private_checksum) is str
        and len(private_checksum) == 64
        and all(character in "0123456789abcdef" for character in private_checksum)
        and (
            (row.get("chunk_count") == 0 and private_checksum == empty_private_checksum)
            or (
                row.get("chunk_count", 0) > 0
                and private_checksum != empty_private_checksum
            )
        )
    )
    if (
        row.get("validation_checksum") != validation_checksum
        or row.get("graph_checksum") != validation_checksum
        or row.get("state") not in {"staging", "building", "ready"}
        or type(row.get("chunk_count")) is not int
        or row.get("validated_private_mapping_checksum") != private_checksum
        or row.get("validated_topology_checksum") != row.get("topology_checksum")
        or not private_checksum_valid
    ):
        raise ValueError("generation validation checksum/state mismatch")
    if not validate_topology_marker(
        driver, generation_key, timeout_seconds=timeout
    ):
        raise ValueError("generation topology attestation mismatch")
    parameters = {
        "generation_key": generation_key,
        "validation_checksum": validation_checksum,
        "private_mapping_checksum": private_checksum,
        "chunk_count": row["chunk_count"],
        "topology_checksum": row["topology_checksum"],
        "state": "ready",
    }
    driver.execute_write(
        "MATCH (g:CollectionGeneration {generation_key:$generation_key}) "
        "WHERE g.validation_checksum=$validation_checksum "
        "AND g.graph_checksum=$validation_checksum "
        "AND g.private_mapping_checksum=$private_mapping_checksum "
        "AND g.validated_private_mapping_checksum=$private_mapping_checksum "
        "AND g.topology_checksum=$topology_checksum "
        "AND g.validated_topology_checksum=$topology_checksum "
        "AND g.chunk_count=$chunk_count "
        "AND g.state IN ['staging','building','ready'] SET g.state=$state",
        parameters,
        timeout_seconds=timeout,
    )
    refreshed = _ready_state(driver, generation_key, timeout)
    if refreshed != {**row, "state": "ready"}:
        raise ValueError("generation ready publication fence was lost")


def _ready_state(driver, generation_key, timeout):
    rows = driver.execute_read(
        "MATCH (g:CollectionGeneration {generation_key:$generation_key}) "
        "RETURN g.validation_checksum AS validation_checksum, "
        "g.graph_checksum AS graph_checksum, "
        "g.private_mapping_checksum AS private_mapping_checksum, "
        "g.validated_private_mapping_checksum AS validated_private_mapping_checksum, "
        "g.topology_checksum AS topology_checksum, "
        "g.validated_topology_checksum AS validated_topology_checksum, "
        "g.chunk_count AS chunk_count, g.state AS state",
        {"generation_key": generation_key},
        timeout_seconds=timeout,
        max_records=1,
    )
    return dict(rows[0]) if len(rows) == 1 else {}
