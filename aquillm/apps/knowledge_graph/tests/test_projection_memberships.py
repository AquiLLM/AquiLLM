from __future__ import annotations

from dataclasses import replace

import pytest

from apps.knowledge_graph.projection import memberships
from apps.knowledge_graph.projection.identifiers import (
    HmacSha256ProjectionIdentifierCodec,
)
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

    codec = HmacSha256ProjectionIdentifierCodec(b"secret-a", key_version="key-v1")
    rows = memberships.load_automatic_membership_assignments(
        collection_ids=(7,),
        using="graph_reader",
        batch_size=2,
        codec=codec,
        generation="generation-a",
    )

    assert all(type(row) is AutomaticCanonicalMembershipV1 for row in rows)
    assert len(rows) == 2
    assert {row.automatic_membership_key for row in rows} == {
        None,
        memberships._opaque_membership_key(codec, 91),
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
        collection_ids=(7,),
        using="default",
        batch_size=1,
        codec=HmacSha256ProjectionIdentifierCodec(b"secret-a", key_version="key-v1"),
        generation="generation-a",
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


def test_membership_keys_are_hmac_domain_separated_and_not_plain_pk_hashes(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        memberships,
        "_load_membership_source_rows",
        lambda **_kwargs: ((11, "resolver-v1", A, 91, B),),
    )

    def load(secret: bytes, version: str, generation: str):
        return memberships.load_automatic_membership_assignments(
            collection_ids=(7,),
            using="default",
            batch_size=1,
            codec=HmacSha256ProjectionIdentifierCodec(secret, key_version=version),
            generation=generation,
        )[0]

    baseline = load(b"secret-a", "key-v1", "generation-a")
    secret_changed = load(b"secret-b", "key-v1", "generation-a")
    version_changed = load(b"secret-a", "key-v2", "generation-a")
    generation_changed = load(b"secret-a", "key-v1", "generation-b")

    assert baseline.entity_key != secret_changed.entity_key
    assert baseline.automatic_membership_key != secret_changed.automatic_membership_key
    assert baseline.entity_key != version_changed.entity_key
    assert baseline.automatic_membership_key != version_changed.automatic_membership_key
    assert baseline.entity_key != generation_changed.entity_key
    assert (
        baseline.automatic_membership_key == generation_changed.automatic_membership_key
    )
    assert baseline.entity_key != memberships.sha256(b"11").hexdigest()
    assert (
        baseline.entity_key
        != memberships.sha256(b"membership-collection-entity-v1\x0011").hexdigest()
    )
    assert (
        baseline.automatic_membership_key
        != memberships.sha256(b"automatic-canonical-identity-v1\x0091").hexdigest()
    )
