"""Opt-in staging and independent attestation of the derived access structure."""

from .frontier_index_build_v2 import build_frontier_index
from .frontier_index_contracts_v2 import (
    MARKER_FIELDS,
    PACKED_VERSION,
    decode_record,
    marker_present,
)
from .memgraph_edges import write_parameterized_batches
from .memgraph_records import FAMILIES, _dto, _properties
from .memgraph_streaming import _BUNDLE_FIELDS, _COUNT_FIELDS, stream_family_records
from .records import CollectionGraphProjectionBundleV1, ProjectionGenerationMarkerV1

# Deploy this writer only after compatible readers; existing markers keep their recipe.
_NEW_INDEX_VERSION = PACKED_VERSION


def _stored_version(marker):
    present = {name for name in MARKER_FIELDS if name in marker}
    if not present:
        return None
    if present != set(MARKER_FIELDS):
        raise ValueError("partial frontier marker cannot be reconstructed")
    if marker["validated_frontier_checksum"] not in (
        "0" * 64,
        marker["frontier_checksum"],
    ):
        raise ValueError("invalid staging frontier attestation")
    marker_present(
        {**marker, "validated_frontier_checksum": marker["frontier_checksum"]}
    )
    return marker["frontier_version"]


def read_marker(driver, generation, timeout, *, alias="record"):
    rows = driver.execute_read(
        "MATCH (g:CollectionGeneration {generation_key:$generation_key}) "
        "RETURN g AS " + alias,
        {"generation_key": generation},
        timeout_seconds=timeout,
        max_records=1,
    )
    if len(rows) != 1:
        raise ValueError("Memgraph generation manifest is missing")
    return _properties(rows[0], alias)


def write_frontier_index(driver, bundle, *, batch_size, timeout):
    # This is staging-only. There is no in-place ready-generation backfill.
    marker = read_marker(driver, bundle.generation.generation_key, timeout)
    if marker.get("state") not in {"staging", "building"}:
        raise ValueError("frontier population requires a staging generation")
    existing = _stored_version(marker)
    index = build_frontier_index(
        bundle, directory_version=existing or _NEW_INDEX_VERSION
    )
    if existing and any(
        marker[name] != value
        for name, value in index.marker.items()
        if name != "validated_frontier_checksum"
    ):
        raise ValueError("frontier marker is not derived from this projection")
    assignments = ", ".join(
        "n." + field + "=row." + field
        for field in index.records[0]
        if field not in {"index_key", "generation_key"}
    )
    write_parameterized_batches(
        driver,
        "MERGE (n:FrontierIndexV2 {index_key:row.index_key}) "
        "SET n.generation_key=$generation_key, " + assignments,
        iter(index.records),
        generation_key=bundle.generation.generation_key,
        batch_size=batch_size,
        timeout_seconds=timeout,
    )
    properties = {**index.marker, "validated_frontier_checksum": "0" * 64}
    driver.execute_write(
        "MATCH (g:CollectionGeneration {generation_key:$generation_key}) "
        "WHERE g.state IN ['staging','building'] SET "
        + ", ".join("g." + name + "=$" + name for name in properties),
        {"generation_key": bundle.generation.generation_key, **properties},
        timeout_seconds=timeout,
    )


def validate_frontier_if_present(driver, expected, *, timeout, marker):
    """Compare stored index to a reconstruction from independently read source."""
    generation = expected.generation_key
    version = _stored_version(marker)
    if version is None:
        return
    families = {}
    field_by_label = {
        label: name for name, label in _BUNDLE_FIELDS if label is not None
    }
    for label, _kind in FAMILIES:
        families[field_by_label[label]] = tuple(
            stream_family_records(
                driver,
                generation_key=generation,
                label=label,
                expected_count=getattr(expected.counts, _COUNT_FIELDS[label]),
                timeout_seconds=timeout,
            )
        )
    bundle = CollectionGraphProjectionBundleV1(
        generation=_dto(ProjectionGenerationMarkerV1, marker),
        counts=expected.counts,
        **families,
    )
    index = build_frontier_index(bundle, directory_version=version)
    if any(
        marker[name] != value
        for name, value in index.marker.items()
        if name != "validated_frontier_checksum"
    ):
        raise ValueError("frontier marker is not derived from this projection")
    count, after = 0, ""
    expected_rows = iter(index.records)
    while True:
        limit = min(1000, len(index.records) - count + 1)
        rows = driver.execute_read(
            "MATCH (n:FrontierIndexV2 {generation_key:$generation_key}) "
            "WHERE n.index_key > $after_key RETURN properties(n) AS record "
            "ORDER BY n.index_key LIMIT $limit",
            {"generation_key": generation, "after_key": after, "limit": limit},
            timeout_seconds=timeout,
            max_records=limit,
        )
        for row in rows:
            value = row["record"]
            wanted = next(expected_rows, None)
            if value != wanted:
                raise ValueError(
                    "frontier stored records differ from source reconstruction"
                )
            decode_record(value, expected_key=value["index_key"], generation=generation)
            count += 1
            after = value["index_key"]
        if len(rows) < limit:
            break
    if count != len(index.records):
        raise ValueError("frontier stored record count differs")
    if marker["state"] == "ready":
        marker_present(marker)
        return
    driver.execute_write(
        "MATCH (g:CollectionGeneration {generation_key:$generation_key}) "
        "WHERE g.state IN ['staging','building'] AND g.graph_checksum=$graph_checksum "
        "AND g.frontier_checksum=$frontier_checksum "
        "SET g.validated_frontier_checksum=$frontier_checksum",
        {
            "generation_key": generation,
            "graph_checksum": expected.graph_checksum,
            "frontier_checksum": index.marker["frontier_checksum"],
        },
        timeout_seconds=timeout,
    )
    marker_present(read_marker(driver, generation, timeout))


def frontier_ready_parameters(row):
    """Optional CAS fields are used only for indexed generations."""
    clean = {name: row[name] for name in MARKER_FIELDS if row.get(name) is not None}
    if not clean:
        return {}, ""
    marker_present(clean)
    return clean, "".join("AND g." + name + "=$" + name + " " for name in MARKER_FIELDS)
