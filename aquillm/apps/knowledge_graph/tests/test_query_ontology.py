from __future__ import annotations

from pathlib import Path

import pytest

from apps.knowledge_graph.retrieval import query_ontology
from apps.knowledge_graph.retrieval.direct_seed_contracts import DirectFailureReason
from apps.knowledge_graph.services.ontology import load_ontology

ONTOLOGY_PATH = Path(__file__).resolve().parents[1] / "ontologies" / "research-v1.yaml"


def _artifact_rows(checksums: tuple[str, ...]) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "id": index + 1,
            "ontology_checksum": checksum,
            "ontology_version": "1.0.0",
        }
        for index, checksum in enumerate(checksums)
    )


def test_loads_exact_active_yaml_for_selected_artifact_checksum(monkeypatch) -> None:
    definition = load_ontology(ONTOLOGY_PATH)
    monkeypatch.setattr(
        query_ontology,
        "_load_selected_artifact_rows",
        lambda **_kwargs: _artifact_rows((definition.checksum, definition.checksum)),
    )
    monkeypatch.setattr(
        query_ontology,
        "_load_active_ontology_row",
        lambda **_kwargs: {
            "checksum": definition.checksum,
            "version": definition.version,
            "metadata": {"yaml": definition.raw_yaml},
        },
    )

    outcome = query_ontology.load_query_ontology(
        selected_artifact_ids=(1, 2), using="default"
    )

    assert outcome.failure_reason is None
    assert outcome.ontology is not None
    assert outcome.ontology.checksum == definition.checksum
    assert outcome.selected_artifact_count == 2


def test_mixed_or_nonactive_ontology_is_a_direct_branch_local_failure(
    monkeypatch,
) -> None:
    definition = load_ontology(ONTOLOGY_PATH)
    monkeypatch.setattr(
        query_ontology,
        "_load_selected_artifact_rows",
        lambda **_kwargs: _artifact_rows((definition.checksum, "f" * 64)),
    )
    monkeypatch.setattr(
        query_ontology,
        "_load_active_ontology_row",
        lambda **_kwargs: None,
    )

    outcome = query_ontology.load_query_ontology(
        selected_artifact_ids=(1, 2), using="default"
    )

    assert outcome.ontology is None
    assert outcome.failure_reason is DirectFailureReason.MIXED_ONTOLOGY
    assert outcome.selected_artifact_count == 2


@pytest.mark.parametrize(
    "selected_state", ("active", "superseded", "wrong_checksum", "invalid_yaml")
)
@pytest.mark.parametrize("collection_scoped", (False, True))
def test_unrelated_collection_ontology_does_not_change_selected_identity(
    monkeypatch, selected_state, collection_scoped
):
    from apps.knowledge_graph.models import OntologyVersion

    definition = load_ontology(ONTOLOGY_PATH)
    metadata = {
        "yaml": definition.raw_yaml if selected_state != "invalid_yaml" else "invalid"
    }
    if collection_scoped:
        metadata["collection_id"] = 1
    records = (
        {
            "kind": "graph",
            "status": "superseded" if selected_state == "superseded" else "active",
            "version": definition.version,
            "checksum": "e" * 64
            if selected_state == "wrong_checksum"
            else definition.checksum,
            "metadata": metadata,
        },
        {
            "kind": "graph",
            "status": "active",
            "version": "0.0.1+collection.99",
            "checksum": "f" * 64,
            "metadata": {"collection_id": 99},
        },
    )

    class Rows:
        def __init__(self, rows):
            self.rows = rows

        def using(self, alias):
            assert alias == "default"
            return self

        def filter(self, **kwargs):
            return Rows(
                tuple(
                    row
                    for row in self.rows
                    if all(row[key] == value for key, value in kwargs.items())
                )
            )

        def order_by(self, *fields):
            return self

        def values(self, *fields):
            return tuple({key: row[key] for key in fields} for row in self.rows)

    monkeypatch.setattr(OntologyVersion, "objects", Rows(records))
    monkeypatch.setattr(
        query_ontology,
        "_load_selected_artifact_rows",
        lambda **_: _artifact_rows((definition.checksum,)),
    )
    outcome = query_ontology.load_query_ontology(
        selected_artifact_ids=(1,), using="default"
    )
    if selected_state == "active":
        assert outcome.failure_reason is None
        assert outcome.ontology.checksum == definition.checksum
    else:
        assert outcome.failure_reason is DirectFailureReason.MIXED_ONTOLOGY


@pytest.mark.parametrize("artifact_ids", ((), (2, 1), (1, 1), (True,)))
def test_selected_artifact_ids_are_bounded_canonical(artifact_ids) -> None:
    with pytest.raises((TypeError, ValueError)):
        query_ontology.load_query_ontology(
            selected_artifact_ids=artifact_ids, using="default"
        )
