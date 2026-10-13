"""Frozen retrieval identities must invalidate every independently changing input."""

import hashlib
import json

import pytest

from apps.documents.services.retrieval_scope_identity import retrieval_scope_key

BASE = dict(
    authorization_digest="auth-a",
    representation_digest="repr-a",
    dense_generation_digest="dense-a",
    embedding_space_key="space-a",
    graph_authority_digest="graph-a",
    graph_policy_digest="policy-a",
    query_digest="query-a",
    ranking_policy="hybrid-v2",
    vocabulary_digest="vocab-a",
)


@pytest.mark.parametrize("field", tuple(BASE))
def test_every_scope_component_invalidates_cache(field):
    assert retrieval_scope_key(**BASE) != retrieval_scope_key(
        **{**BASE, field: "changed"}
    )


def test_explicit_unknown_space_and_canonical_nine_field_payload():
    unknown = {**BASE, "embedding_space_key": None}
    expected = hashlib.sha256(
        json.dumps(
            unknown,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert retrieval_scope_key(**unknown) == expected
    assert expected != retrieval_scope_key(**BASE)


@pytest.mark.parametrize("field", tuple(BASE))
@pytest.mark.parametrize("value", ["", " ", False, 12])
def test_required_values_are_exact_nonempty_strings(field, value):
    with pytest.raises(ValueError):
        retrieval_scope_key(**{**BASE, field: value})
