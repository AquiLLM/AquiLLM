"""Captured authority is never replaced by a caller-supplied mapping."""

from dataclasses import replace
from uuid import uuid4

import pytest

from apps.knowledge_graph.retrieval.topology.frontier_selection_v2 import (
    FrontierSelectionV2,
    _representation_values,
)
from apps.knowledge_graph.tests.frontier_fixture_support import FrontierFixture


def test_original_selection_and_real_codec_bind_explicit_legacy_null():
    fixture = FrontierFixture()
    binding = FrontierSelectionV2(fixture.selection, fixture.codec)
    assert binding.require_group(fixture.group) == fixture.group
    for document in fixture.group.scope.ready.authorized_documents:
        assert (
            binding.representation(document.generation_key, document.document_key)
            is None
        )
        with pytest.raises(ValueError):
            binding.representation("f" * 64, document.document_key)
        with pytest.raises(ValueError):
            binding.representation(document.generation_key, "f" * 64)


def test_uuid_mapping_is_only_pure_plumbing_until_i5_persisted_ownership():
    fixture = FrontierFixture()
    legacy = fixture.selection.representation_snapshot.selections[0]
    value = uuid4()
    nonnull = replace(legacy, generation_key=value)
    assert _representation_values((nonnull,))[legacy.document.id] == value
    assert _representation_values((legacy,))[legacy.document.id] is None
    with pytest.raises(ValueError):
        _representation_values((legacy, nonnull))
    # Current actual I4 producer must still reject nonnull authority.
    with pytest.raises(ValueError):
        replace(
            fixture.selection,
            representation_snapshot=replace(
                fixture.selection.representation_snapshot, selections=(nonnull,)
            ),
        )


def test_wrong_codec_group_or_snapshot_cannot_rebind_frontier():
    from apps.knowledge_graph.projection.identifiers import (
        HmacSha256ProjectionIdentifierCodec,
    )

    fixture = FrontierFixture()
    with pytest.raises(ValueError):
        FrontierSelectionV2(
            fixture.selection,
            HmacSha256ProjectionIdentifierCodec(b"wrong-key", key_version="key-v1"),
        )
    binding = FrontierSelectionV2(fixture.selection, fixture.codec)
    with pytest.raises(ValueError):
        binding.require_group(replace(fixture.group, group_checksum="f" * 64))
    with pytest.raises(ValueError):
        replace(
            fixture.selection,
            representation_snapshot=replace(
                fixture.selection.representation_snapshot,
                selections=fixture.selection.representation_snapshot.selections[:1],
            ),
        )
