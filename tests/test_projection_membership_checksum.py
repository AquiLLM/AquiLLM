"""Membership hashing preserves the public canonical bytes without recursive work."""

import sys
from dataclasses import replace
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aquillm"))

from apps.knowledge_graph.projection import serialization
from apps.knowledge_graph.projection.membership_checksum import (
    membership_decision_checksum as checksum,
)
from apps.knowledge_graph.projection.records import AutomaticCanonicalMembershipV1


def row(number=1, *, linked=False, resolver="resolver-v1"):
    return AutomaticCanonicalMembershipV1(
        f"{number:064x}",
        "a" * 64 if linked else None,
        "b" * 64,
        resolver,
        "c" * 64,
    )


def legacy_checksum(assignments):
    if isinstance(assignments, (str, bytes)):
        raise TypeError("assignments must be projection membership rows")
    rows = tuple(assignments)
    if any(type(item) is not AutomaticCanonicalMembershipV1 for item in rows):
        raise TypeError("assignments must contain exact membership rows")
    rows = tuple(sorted(rows, key=lambda item: item.entity_key))
    return (
        serialization.projection_checksum(rows) if rows else sha256(b"[]").hexdigest()
    )


def test_large_membership_hash_avoids_recursive_record_serialization(monkeypatch):
    rows = tuple(row(i) for i in range(1, 1026))
    expected = legacy_checksum(rows)
    original = serialization._encode
    recursive_calls = 0

    def observe(value, provider_types):
        nonlocal recursive_calls
        recursive_calls += 1
        return original(value, provider_types)

    monkeypatch.setattr(serialization, "_encode", observe)
    assert checksum(rows) == expected
    assert recursive_calls == 0


@pytest.mark.parametrize("factory", [tuple, list, iter, set])
def test_unsorted_null_and_linked_rows_keep_canonical_hash(factory):
    rows = (row(7, linked=True), row(2), row(3, resolver='résolveur-星-"\\'))
    assert checksum(factory(rows)) == legacy_checksum(rows)


@pytest.mark.parametrize("value", [(), [], iter(()), {}, bytearray()])
def test_empty_accepted_iterables_keep_empty_array_hash(value):
    assert (
        checksum(value)
        == "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
    )


@pytest.mark.parametrize("value", ["", b"", "rows", None, [object()], [row(), 2]])
def test_invalid_input_types_keep_rejection(value):
    with pytest.raises(TypeError):
        legacy_checksum(value)
    with pytest.raises(TypeError):
        checksum(value)


def test_record_subclass_is_not_an_exact_membership_row():
    class OtherRow(AutomaticCanonicalMembershipV1):
        pass

    with pytest.raises(TypeError):
        checksum((OtherRow("a" * 64, None, "b" * 64, "v1", "c" * 64),))


def test_duplicate_entity_with_different_decision_is_rejected():
    rows = (row(), replace(row(), decision_checksum="d" * 64))
    with pytest.raises(ValueError):
        legacy_checksum(rows)
    with pytest.raises(ValueError):
        checksum(rows)


@pytest.mark.parametrize("count", [0, 1, 256, 257, 513])
def test_streamed_bytes_match_generic_array_at_chunk_boundaries(monkeypatch, count):
    from apps.knowledge_graph.projection import membership_checksum as optimized

    rows = tuple(row(i, linked=bool(i % 2)) for i in range(count, 0, -1))
    expected = (
        serialization.canonical_projection_bytes(tuple(reversed(rows)))
        if rows
        else b"[]"
    )
    observed = bytearray()

    class ObservedDigest:
        def __init__(self, value):
            self.digest = sha256(value)
            observed.extend(value)

        def update(self, value):
            observed.extend(value)
            self.digest.update(value)

        def hexdigest(self):
            return self.digest.hexdigest()

    monkeypatch.setattr(optimized, "sha256", ObservedDigest)
    assert checksum(rows) == sha256(expected).hexdigest()
    assert bytes(observed) == expected


@pytest.mark.parametrize(
    "value", [3, True, 1.25, ("nested", 2), datetime(2026, 1, 1, tzinfo=UTC)]
)
def test_defensively_altered_record_keeps_generic_scalar_encoding(value):
    item = row()
    object.__setattr__(item, "resolver_version", value)
    assert checksum((item,)) == legacy_checksum((item,))


def test_defensively_altered_enum_keeps_generic_encoding():
    class Version(StrEnum):
        CURRENT = "v1"

    item = row()
    object.__setattr__(item, "resolver_version", Version.CURRENT)
    assert checksum((item,)) == legacy_checksum((item,))


@pytest.mark.parametrize("value", [object(), float("nan"), {"nested": 1}, "\ud800"])
def test_defensively_altered_unsupported_value_keeps_rejection(value):
    item = row()
    object.__setattr__(item, "resolver_version", value)
    with pytest.raises((TypeError, ValueError, UnicodeEncodeError)):
        legacy_checksum((item,))
    with pytest.raises((TypeError, ValueError, UnicodeEncodeError)):
        checksum((item,))


def test_captured_deadline_rejects_before_reading_input():
    def expired():
        raise TimeoutError("captured deadline")

    def unread():
        pytest.fail("expired caller was consumed")
        yield row()

    with pytest.raises(TimeoutError, match="captured deadline"):
        checksum(unread(), check=expired)


def test_captured_deadline_is_checked_between_input_batches():
    consumed = 0

    def stream():
        nonlocal consumed
        for i in range(1, 2001):
            consumed += 1
            yield row(i)

    def check():
        if consumed >= 300:
            raise TimeoutError("captured deadline")

    with pytest.raises(TimeoutError):
        checksum(stream(), check=check)
    assert 300 <= consumed < 1000


def test_expiry_during_serialization_prevents_remaining_chunks(monkeypatch):
    from apps.knowledge_graph.projection import membership_checksum as optimized

    rows = tuple(row(i) for i in range(1, 1026))
    expired = False
    encoded = 0
    original = optimized.json.dumps

    def encode(*args, **kwargs):
        nonlocal expired, encoded
        encoded += len(args[0])
        result = original(*args, **kwargs)
        expired = True
        return result

    def check():
        if expired:
            raise TimeoutError("original caller expired")

    monkeypatch.setattr(optimized.json, "dumps", encode)
    with pytest.raises(TimeoutError):
        checksum(rows, check=check)
    assert 0 < encoded < len(rows)


def test_empty_result_still_checks_original_caller():
    checked = False

    def check():
        nonlocal checked
        checked = True

    assert checksum((), check=check) == legacy_checksum(())
    assert checked
