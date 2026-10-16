"""Exact native key specialization preserves public encoding and fallback."""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aquillm"))
from apps.knowledge_graph.projection import identifiers as ids
from apps.knowledge_graph.projection import membership_identifiers as helper

Domain = ids.ProjectionIdentifierDomain


def codec():
    return ids.HmacSha256ProjectionIdentifierCodec(
        b"private-test-key", key_version="key-v1"
    )


def expected(value, generation, source, canonical=False):
    return value.encode(
        Domain.AUTOMATIC_CANONICAL_IDENTITY if canonical else Domain.ENTITY,
        **({} if canonical else {"generation": generation}),
        source=source,
    ).value


def test_native_prefix_framing_is_constant_work_and_repeated_canonical_ids_are_memoized():
    value, generation = codec(), "registry-v1"
    rows = tuple(range(1, 129))
    golden = [
        (expected(value, generation, pk), expected(value, generation, pk % 7, True))
        for pk in rows
    ]
    counts = {"frame": 0, "encode": 0}
    codes = {
        value.frame.__func__.__code__: "frame",
        value.encode.__func__.__code__: "encode",
    }

    def count(frame, event, arg):
        if event == "call" and frame.f_code in codes:
            counts[codes[frame.f_code]] += 1

    previous = sys.getprofile()
    sys.setprofile(count)
    try:
        encoder = helper.MembershipKeyEncoder(value, generation)
        actual = [(encoder.entity(pk), encoder.canonical(pk % 7)) for pk in rows]
    finally:
        sys.setprofile(previous)
    assert actual == golden
    assert counts == {"frame": 2, "encode": 0}
    assert len(encoder._canonical) == 7


@pytest.mark.parametrize(
    "generation", ["registry-v1", "é研究", UUID(int=8), str(UUID(int=9))]
)
@pytest.mark.parametrize("source", [-(2**63), -1, 0, 1, 2**63 - 1])
def test_integer_byte_parity(generation, source):
    for key, version in [(b"one", "k1"), (b"two", "κ2")]:
        value = ids.HmacSha256ProjectionIdentifierCodec(key, key_version=version)
        encoder = helper.MembershipKeyEncoder(value, generation)
        assert encoder.entity(source) == expected(value, generation, source)
        assert encoder.canonical(source) == expected(value, generation, source, True)


@pytest.mark.parametrize("source", ["entity", "研究", UUID(int=12), str(UUID(int=12))])
def test_non_integer_sources_preserve_generic_accepted_types(source):
    value, generation = codec(), "registry-v1"
    encoder = helper.MembershipKeyEncoder(value, generation)
    assert encoder.entity(source) == expected(value, generation, source)
    assert encoder.canonical(source) == expected(value, generation, source, True)


@pytest.mark.parametrize(
    "source",
    [
        True,
        None,
        2**63,
        -(2**63) - 1,
        "",
        " ",
        "\ud800",
        "{00000000-0000-0000-0000-000000000001}",
        1.5,
    ],
)
def test_invalid_sources_preserve_exception_type(source):
    value, generation = codec(), "registry-v1"
    encoder = helper.MembershipKeyEncoder(value, generation)
    for canonical in (False, True):
        with pytest.raises((TypeError, ValueError)) as original:
            expected(value, generation, source, canonical)
        with pytest.raises(type(original.value)):
            (encoder.canonical if canonical else encoder.entity)(source)


@pytest.mark.parametrize(
    "generation",
    [None, "", " ", "-", True, "\ud800", "{00000000-0000-0000-0000-000000000001}"],
)
def test_invalid_generation_is_lazy_and_does_not_break_canonical_domain(generation):
    value = codec()
    encoder = helper.MembershipKeyEncoder(
        value, generation
    )  # Empty loaders need no validation.
    assert encoder.canonical(1) == expected(value, generation, 1, True)
    with pytest.raises((TypeError, ValueError)) as original:
        expected(value, generation, 1)
    with pytest.raises(type(original.value)):
        encoder.entity(1)


def test_custom_codec_and_subclass_are_always_delegated():
    calls = []

    class Custom:
        def encode(self, *args, **kwargs):
            calls.append((args, kwargs))
            return SimpleNamespace(value=f"custom-{len(calls)}")

    class Subclass(ids.HmacSha256ProjectionIdentifierCodec):
        def encode(self, *args, **kwargs):
            return Custom().encode(*args, **kwargs)

    for value in (Custom(), Subclass(b"test", key_version="k1")):
        encoder = helper.MembershipKeyEncoder(value, "g")
        start = len(calls)
        assert encoder.canonical(7) == f"custom-{start + 1}"
        assert encoder.canonical(7) == f"custom-{start + 2}"
        assert encoder.entity(8) == f"custom-{start + 3}"


@pytest.mark.parametrize(
    "field,new",
    [
        ("_key", b"changed"),
        ("key_version", "changed"),
        ("codec_version", "changed"),
        ("_key", bytearray(b"changed")),
    ],
)
def test_callback_field_mutation_discards_prefix_and_cached_canonical(field, new):
    value = codec()
    encoder = helper.MembershipKeyEncoder(value, "g")
    original = encoder.canonical(7)
    object.__setattr__(
        value, field, new
    )  # Represents the original check callback between rows.
    assert encoder.canonical(7) == expected(value, "g", 7, True)
    assert encoder.canonical(7) != original
    assert encoder.entity(8) == expected(value, "g", 8)


def test_uuid_generation_mutation_cannot_reuse_old_entity_prefix():
    value, generation = codec(), UUID(int=4)
    encoder = helper.MembershipKeyEncoder(value, generation)
    old = encoder.entity(8)
    object.__setattr__(generation, "int", 5)
    assert encoder.entity(8) == expected(value, generation, 8)
    assert encoder.entity(8) != old


def test_method_replacement_after_memoization_is_generic(monkeypatch):
    value = codec()
    encoder = helper.MembershipKeyEncoder(value, "g")
    encoder.canonical(7)
    monkeypatch.setattr(
        ids.HmacSha256ProjectionIdentifierCodec,
        "encode",
        lambda self, *args, **kwargs: SimpleNamespace(value="replaced"),
    )
    assert encoder.canonical(7) == "replaced"
    assert encoder.entity(8) == "replaced"


def test_method_replacement_before_lazy_helper_import_is_not_canonical(monkeypatch):
    with monkeypatch.context() as changed:
        changed.setattr(
            ids.HmacSha256ProjectionIdentifierCodec,
            "encode",
            lambda self, *args, **kwargs: SimpleNamespace(value="pre-import"),
        )
        importlib.reload(helper)
        assert helper.MembershipKeyEncoder(codec(), "g").entity(8) == "pre-import"
    importlib.reload(helper)


def test_framing_delegate_replacement_after_memoization_is_generic(monkeypatch):
    value = codec()
    encoder = helper.MembershipKeyEncoder(value, "g")
    before = encoder.canonical(7)
    original = ids._frame_field
    monkeypatch.setattr(ids, "_frame_field", lambda value: original(value) + b"x")
    assert encoder.canonical(7) == expected(value, "g", 7, True)
    assert encoder.canonical(7) != before


def test_new_invocation_has_no_cross_call_memo_or_key_authority():
    one, two = (
        codec(),
        ids.HmacSha256ProjectionIdentifierCodec(b"new", key_version="key-v1"),
    )
    first = helper.MembershipKeyEncoder(one, "g")
    second = helper.MembershipKeyEncoder(two, "g")
    assert first.canonical(7) != second.canonical(7)
    assert second.canonical(7) == expected(two, "g", 7, True)


def test_equal_valued_non_native_codec_field_cannot_reuse_prefix():
    class ChangedText(str):
        def encode(self, *args, **kwargs):
            return super().encode(*args, **kwargs) + b"x"

    value = codec()
    encoder = helper.MembershipKeyEncoder(value, "g")
    old = encoder.canonical(7)
    object.__setattr__(value, "key_version", ChangedText(value.key_version))
    assert encoder.canonical(7) == expected(value, "g", 7, True)
    assert encoder.canonical(7) != old


def test_equal_bytearray_key_discards_native_preparation():
    value = codec()
    encoder = helper.MembershipKeyEncoder(value, "g")
    encoder.canonical(7)
    object.__setattr__(value, "_key", bytearray(value._key))
    assert encoder.canonical(7) == expected(value, "g", 7, True)
    assert encoder._generic and encoder._prefixes is None and encoder._canonical == {}


@pytest.mark.parametrize("delegate", ["canonicalization", "output_class", "hash"])
def test_replaced_encoding_delegate_discards_native_memo(monkeypatch, delegate):
    value = codec()
    encoder = helper.MembershipKeyEncoder(value, "g")
    old = encoder.canonical(7)
    if delegate == "canonicalization":
        monkeypatch.setattr(ids, "_canonical_generation", lambda *args: "changed")
    elif delegate == "output_class":
        monkeypatch.setattr(
            ids,
            "OpaqueProjectionKey",
            lambda **kwargs: SimpleNamespace(value="output-replaced"),
        )
    else:
        monkeypatch.setattr(
            ids, "hmac_new", lambda *args: SimpleNamespace(hexdigest=lambda: "f" * 64)
        )
    assert encoder.canonical(7) == expected(value, "g", 7, True)
    assert encoder.canonical(7) != old
    assert encoder._generic and encoder._canonical == {}


def test_integer_subclass_and_bool_never_collide_with_cached_native_id():
    class IntegerSubclass(int):
        pass

    value = codec()
    encoder = helper.MembershipKeyEncoder(value, "g")
    encoder.canonical(1)
    for source in (True, IntegerSubclass(1)):
        with pytest.raises(TypeError):
            encoder.canonical(source)
        with pytest.raises(TypeError):
            encoder.entity(source)


@pytest.mark.parametrize("domain", [Domain.ENTITY, Domain.AUTOMATIC_CANONICAL_IDENTITY])
def test_domain_value_mutation_cannot_reuse_prepared_bytes(domain):
    value = codec()
    encoder = helper.MembershipKeyEncoder(value, "g")
    canonical = domain is Domain.AUTOMATIC_CANONICAL_IDENTITY
    call = encoder.canonical if canonical else encoder.entity
    before = call(7)
    original = domain.value
    try:
        object.__setattr__(domain, "_value_", original + "_changed")
        assert call(7) == expected(value, "g", 7, canonical)
        assert call(7) != before
        assert (
            encoder._generic and encoder._prefixes is None and encoder._canonical == {}
        )
    finally:
        object.__setattr__(domain, "_value_", original)
