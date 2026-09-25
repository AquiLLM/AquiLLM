"""Cross-collection assertion checks for the immutable fixture manifest."""

from __future__ import annotations

from collections.abc import Mapping

from .fixture_manifest_types import (
    FixtureCanonicalIdentityAssertion,
    FixtureChunkBinding,
    FixtureCollectionBinding,
    FixtureInaccessibleNeighborAssertion,
    FixtureValidationError,
)
from .fixture_manifest_validation_helpers import (
    exact_map,
    exact_sequence,
    exact_text,
)


def validate_fixture_assertions(
    *,
    canonical_rows: object,
    inaccessible_rows: object,
    expected_canonical: set[tuple[str, str]],
    expected_hidden: set[tuple[str, str]],
    collections: Mapping[str, FixtureCollectionBinding],
    chunks: Mapping[str, FixtureChunkBinding],
) -> tuple[
    tuple[FixtureCanonicalIdentityAssertion, ...],
    tuple[FixtureInaccessibleNeighborAssertion, ...],
]:
    canonical: list[FixtureCanonicalIdentityAssertion] = []
    for raw in exact_sequence(canonical_rows, "canonical assertions"):
        row = exact_map(
            raw,
            {"source_chunk_symbol", "target_chunk_symbol", "expected_outcome"},
            "canonical assertion",
        )
        canonical.append(
            FixtureCanonicalIdentityAssertion(
                exact_text(row["source_chunk_symbol"], "canonical source"),
                exact_text(row["target_chunk_symbol"], "canonical target"),
                exact_text(row["expected_outcome"], "canonical outcome"),
            )
        )
    observed_canonical = {
        (row.source_chunk_symbol, row.target_chunk_symbol) for row in canonical
    }
    if (
        observed_canonical != expected_canonical
        or len(observed_canonical) != len(canonical)
        or any(row.expected_outcome != "automatic" for row in canonical)
    ):
        raise FixtureValidationError(
            "canonical assertions differ from current fixtures"
        )
    for row in canonical:
        if (
            chunks[row.source_chunk_symbol].collection_id
            == chunks[row.target_chunk_symbol].collection_id
        ):
            raise FixtureValidationError(
                "canonical identity endpoints require distinct collections"
            )

    inaccessible: list[FixtureInaccessibleNeighborAssertion] = []
    for raw in exact_sequence(inaccessible_rows, "inaccessible assertions"):
        row = exact_map(
            raw,
            {"source_chunk_symbol", "target_chunk_symbol"},
            "inaccessible assertion",
        )
        inaccessible.append(
            FixtureInaccessibleNeighborAssertion(
                exact_text(row["source_chunk_symbol"], "inaccessible source"),
                exact_text(row["target_chunk_symbol"], "inaccessible target"),
            )
        )
    observed_hidden = {
        (row.source_chunk_symbol, row.target_chunk_symbol) for row in inaccessible
    }
    if observed_hidden != expected_hidden or len(observed_hidden) != len(inaccessible):
        raise FixtureValidationError(
            "inaccessible assertions differ from current fixtures"
        )
    for row in inaccessible:
        source, target = (
            chunks[row.source_chunk_symbol],
            chunks[row.target_chunk_symbol],
        )
        if (
            not collections[source.collection_symbol].authorized
            or collections[target.collection_symbol].authorized
            or source.collection_id == target.collection_id
        ):
            raise FixtureValidationError(
                "inaccessible assertion must cross distinct declared collections"
            )
    return tuple(canonical), tuple(inaccessible)
