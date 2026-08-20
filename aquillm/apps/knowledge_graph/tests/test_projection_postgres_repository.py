from __future__ import annotations

import uuid

import pytest

from apps.knowledge_graph.projection.postgres_repository import (
    PostgresProjectionRepository,
)
from apps.knowledge_graph.projection.records import (
    AutomaticCanonicalMembershipV1,
    ProjectedArtifactProvenanceV1,
    ProjectedEntityV1,
    ProjectionCountsV1,
    ProjectionGenerationMarkerV1,
)


def _key(character: str) -> str:
    return character * 64


class _BundleSource:
    def __init__(self, *, endpoint_closed: bool = True) -> None:
        self.calls: list[tuple[uuid.UUID, int]] = []
        self.endpoint_closed = endpoint_closed

    def load_projection_rows(self, *, projection_id: uuid.UUID, batch_size: int):
        self.calls.append((projection_id, batch_size))
        marker = ProjectionGenerationMarkerV1(
            _key("1"),
            _key("2"),
            _key("3"),
            "schema-v1",
            "projection-v1",
            "key-v1",
            4,
            _key("4"),
        )
        entities = (
            ProjectedEntityV1(
                _key("5"), _key("1"), _key("3"), _key("2"), "person", _key("6"), 0.75
            ),
        )
        if not self.endpoint_closed:
            from apps.knowledge_graph.projection.records import (
                ProjectedPhysicalRelationV1,
            )

            relations = (
                ProjectedPhysicalRelationV1(
                    _key("7"), _key("3"), _key("5"), "knows", _key("8")
                ),
            )
        else:
            relations = ()
        provenance = (
            ProjectedArtifactProvenanceV1(
                _key("3"),
                "collection",
                _key("2"),
                _key("2"),
                None,
                False,
                _key("9"),
                1,
                1,
                _key("a"),
                "ontology-v1",
                _key("b"),
                "extractor-v1",
                "resolver-v1",
                _key("c"),
                "filter-v1",
                _key("d"),
                "embed-v1",
                "assembly-v1",
                _key("e"),
            ),
        )
        return {
            "generation": marker,
            "entities": entities,
            "automatic_memberships": (
                AutomaticCanonicalMembershipV1(
                    _key("5"), None, _key("4"), "resolver-v1", _key("c")
                ),
            ),
            "documents": (),
            "chunks": (),
            "relations": relations,
            "evidence": (),
            "artifact_provenance": provenance,
            "counts": ProjectionCountsV1(1, 1, 0, 0, len(relations), 0, 1),
        }


def test_bundle_loader_uses_bounded_ordered_source_and_returns_no_private_text() -> (
    None
):
    source = _BundleSource()
    repository = PostgresProjectionRepository(source=source)
    projection_id = uuid.uuid4()

    bundle = repository.load_projection_bundle(
        projection_id=projection_id, batch_size=37
    )

    assert source.calls == [(projection_id, 37)]
    assert bundle.counts.entity_count == 1
    assert "label" not in repr(bundle)
    assert "text" not in repr(bundle)


def test_bundle_loader_rejects_partial_relation_endpoints() -> None:
    repository = PostgresProjectionRepository(
        source=_BundleSource(endpoint_closed=False)
    )

    with pytest.raises(ValueError, match="endpoint"):
        repository.load_projection_bundle(projection_id=uuid.uuid4(), batch_size=10)


def test_bundle_loader_rejects_nonfinite_values_before_provider_write() -> None:
    source = _BundleSource()
    original = source.load_projection_rows

    def invalid(**kwargs):
        rows = original(**kwargs)
        entity = rows["entities"][0]
        object.__setattr__(entity, "retrieval_utility", float("nan"))
        return rows

    source.load_projection_rows = invalid  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="finite"):
        PostgresProjectionRepository(source=source).load_projection_bundle(
            projection_id=uuid.uuid4(), batch_size=10
        )
