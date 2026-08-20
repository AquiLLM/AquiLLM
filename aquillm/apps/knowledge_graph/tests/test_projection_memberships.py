from __future__ import annotations

from dataclasses import replace

import pytest

from apps.knowledge_graph.projection import memberships
from apps.knowledge_graph.projection.records import AutomaticCanonicalMembershipV1

A = "a" * 64
B = "b" * 64
C = "c" * 64


def test_membership_loader_returns_frozen_rows_including_explicit_null(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        memberships,
        "_load_membership_source_rows",
        lambda **_kwargs: (
            (11, "resolver-v1", A, None, None),
            (12, "resolver-v1", A, 91, B),
        ),
    )

    rows = memberships.load_automatic_membership_assignments(
        collection_ids=(7,), using="graph_reader", batch_size=2
    )

    assert all(type(row) is AutomaticCanonicalMembershipV1 for row in rows)
    assert len(rows) == 2
    assert {row.automatic_membership_key for row in rows} == {
        None,
        memberships._opaque_membership_key(91),
    }
    assert rows == tuple(sorted(rows, key=lambda row: row.entity_key))


def test_membership_checksum_changes_for_assignment_and_resolver_changes(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        memberships,
        "_load_membership_source_rows",
        lambda **_kwargs: ((11, "resolver-v1", A, None, None),),
    )
    rows = memberships.load_automatic_membership_assignments(
        collection_ids=(7,), using="default", batch_size=1
    )

    original = memberships.membership_decision_checksum(rows)
    linked = (replace(rows[0], automatic_membership_key=C),)
    resolver_changed = (replace(rows[0], resolver_version="resolver-v2"),)

    assert original != memberships.membership_decision_checksum(linked)
    assert original != memberships.membership_decision_checksum(resolver_changed)


@pytest.mark.parametrize("batch_size", [0, 5001, True])
def test_membership_loader_rejects_unbounded_reads(batch_size: object) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        memberships.load_automatic_membership_assignments(
            collection_ids=(7,),
            using="default",
            batch_size=batch_size,  # type: ignore[arg-type]
        )


def test_membership_loader_rejects_unsorted_or_duplicate_collection_ids() -> None:
    with pytest.raises(ValueError, match="collection_ids"):
        memberships.load_automatic_membership_assignments(
            collection_ids=(7, 7), using="default", batch_size=2
        )
