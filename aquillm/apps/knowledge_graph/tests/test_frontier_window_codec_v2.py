"""Strict independent-frame container and finite checkpoint replay."""

from importlib import import_module
from importlib.util import find_spec

import pytest

from apps.knowledge_graph.projection.frontier_copy_preparation_v2 import CopyPreparation
from apps.knowledge_graph.retrieval.topology.frontier_binary_codec_v2 import (
    encode_binary_response,
)
from apps.knowledge_graph.retrieval.topology.gateway_contracts import _canonical
from apps.knowledge_graph.tests.frontier_copy_batch_support import CopyFixture

PREFIX = "apps.knowledge_graph.retrieval.topology."


def codec():
    assert find_spec(PREFIX + "frontier_window_codec_v2"), (
        "finite container codec missing"
    )
    return import_module(PREFIX + "frontier_window_codec_v2")


def frame():
    return encode_binary_response(
        _canonical({"rows": [], "scanned_rows": 0, "queries": 1})
    )


def test_independent_frames_bind_exact_request_digest_and_reject_trailing_data():
    module = codec()
    payload = module.encode_container(
        "a" * 64, [("index_batch_multi", frame()), ("raw_batch_multi", frame())]
    )
    result = module.decode_container(payload, "a" * 64)
    assert [kind for kind, _ in result] == ["index_batch_multi", "raw_batch_multi"]
    for changed, digest in [
        (payload + b"x", "a" * 64),
        (payload[:-1], "a" * 64),
        (payload, "b" * 64),
        (payload[:40] + b"\x00", "a" * 64),
    ]:
        with pytest.raises(ValueError):
            module.decode_container(changed, digest)


def test_later_corrupt_frame_and_raw_count_never_return_a_partial_prefix():
    module = codec()
    with pytest.raises(ValueError):
        module.encode_container("a" * 64, [("raw_batch_multi", frame())] * 5)
    payload = module.encode_container("a" * 64, [("index_batch_multi", frame())] * 2)
    with pytest.raises(ValueError):
        module.decode_container(payload[:-3] + b"bad", "a" * 64)


def test_copy_checkpoint_roundtrip_derives_same_first_physical_effect():
    assert find_spec(PREFIX + "frontier_window_checkpoint_v2"), (
        "finite checkpoint codec missing"
    )
    module = import_module(PREFIX + "frontier_window_checkpoint_v2")
    fixture = CopyFixture()
    state = CopyPreparation(fixture.session, fixture.pairs, relation_heads=True)
    first = state.next_read()
    operation, checkpoint = module.snapshot(state)
    restored = module.restore(fixture.session.reads, operation, checkpoint)
    assert restored.next_read() == first
