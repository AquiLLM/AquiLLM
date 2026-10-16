"""Canonical UUID callbacks cannot be reclassified by a lazy helper import."""

import importlib
import sys
from pathlib import Path
from uuid import UUID

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aquillm"))
from apps.knowledge_graph.projection import identifiers as ids
from apps.knowledge_graph.projection import membership_identifiers as helper


@pytest.mark.parametrize("delegate", ["str", "init", "getattribute"])
@pytest.mark.parametrize("stage", ["before_import", "after_prefix"])
def test_replaced_uuid_callback_uses_current_generic_encoding(
    monkeypatch, delegate, stage
):
    value = ids.HmacSha256ProjectionIdentifierCodec(b"old", key_version="key-v1")
    generation = str(UUID(int=8)) if delegate == "init" else UUID(int=8)
    encoder = helper.MembershipKeyEncoder(value, generation)
    if stage == "after_prefix":
        encoder.entity(7)
        encoder.canonical(7)
    original_getattribute = UUID.__getattribute__

    def changed_init(self, *args, **kwargs):
        raise RuntimeError("canonical UUID parsing replaced")

    def changed_getattribute(self, name):
        if name == "int":
            object.__setattr__(value, "_key", b"changed-by-callback")
        return original_getattribute(self, name)

    with monkeypatch.context() as changed:
        if delegate == "str":
            changed.setattr(UUID, "__str__", lambda self: "replacement-generation")
        elif delegate == "init":
            changed.setattr(UUID, "__init__", changed_init)
        else:
            changed.setattr(UUID, "__getattribute__", changed_getattribute)
        if stage == "before_import":
            importlib.reload(helper)
            encoder = helper.MembershipKeyEncoder(value, generation)
        if delegate == "init":
            with pytest.raises(RuntimeError):
                encoder.entity(7)
            with pytest.raises(RuntimeError):
                value.encode(
                    ids.ProjectionIdentifierDomain.ENTITY,
                    generation=generation,
                    source=7,
                )
        else:
            actual = encoder.entity(7)  # Must run before generic, including callbacks.
            assert (
                actual
                == value.encode(
                    ids.ProjectionIdentifierDomain.ENTITY,
                    generation=generation,
                    source=7,
                ).value
            )
        assert (
            encoder._generic and encoder._prefixes is None and encoder._canonical == {}
        )
        assert (
            encoder.canonical(7)
            == value.encode(
                ids.ProjectionIdentifierDomain.AUTOMATIC_CANONICAL_IDENTITY, source=7
            ).value
        )
    importlib.reload(helper)


@pytest.mark.parametrize("stage", ["before_import", "after_prefix"])
@pytest.mark.parametrize(
    "owner,field,replacement",
    [
        (ids._ProjectionIdentifierInput, "codec_version", "replacement-version"),
        (ids._ProjectionIdentifierInput, "key_version", "replacement-key-version"),
        (
            ids._ProjectionIdentifierInput,
            "domain",
            ids.ProjectionIdentifierDomain.DOCUMENT,
        ),
        (ids._ProjectionIdentifierInput, "generation", "replacement-generation"),
        (ids._ProjectionIdentifierInput, "source_kind", "replacement-kind"),
        (ids._ProjectionIdentifierInput, "canonical_source", "999"),
        (ids.OpaqueProjectionKey, "domain", "invalid-domain"),
        (ids.OpaqueProjectionKey, "value", "f" * 64),
    ],
)
def test_replaced_record_slot_uses_current_generic_encoding(
    monkeypatch, stage, owner, field, replacement
):
    codec = ids.HmacSha256ProjectionIdentifierCodec(b"key", key_version="key-v1")
    encoder = helper.MembershipKeyEncoder(codec, "generation")
    if stage == "after_prefix":
        encoder.entity(7)
        encoder.canonical(7)
    original = getattr(owner, field)
    descriptor = property(
        lambda self: replacement, lambda self, value: original.__set__(self, value)
    )
    with monkeypatch.context() as changed:
        changed.setattr(owner, field, descriptor)
        if stage == "before_import":
            importlib.reload(helper)
            encoder = helper.MembershipKeyEncoder(codec, "generation")
        for method, domain, kwargs in (
            (
                encoder.entity,
                ids.ProjectionIdentifierDomain.ENTITY,
                {"generation": "generation"},
            ),
            (
                encoder.canonical,
                ids.ProjectionIdentifierDomain.AUTOMATIC_CANONICAL_IDENTITY,
                {},
            ),
        ):
            if owner is ids.OpaqueProjectionKey and field == "domain":
                with pytest.raises(TypeError):
                    method(7)
                with pytest.raises(TypeError):
                    codec.encode(domain, source=7, **kwargs)
            else:
                assert method(7) == codec.encode(domain, source=7, **kwargs).value
            assert encoder._generic and encoder._prefixes is None
            assert encoder._canonical == {}
    importlib.reload(helper)


@pytest.mark.parametrize("stage", ["before_import", "after_prefix"])
@pytest.mark.parametrize(
    "owner,delegate",
    [
        (ids.HmacSha256ProjectionIdentifierCodec, "__getattribute__"),
        (ids._ProjectionIdentifierInput, "__getattribute__"),
        (ids.OpaqueProjectionKey, "__getattribute__"),
    ],
)
def test_replaced_native_lookup_uses_current_generic(
    monkeypatch, stage, owner, delegate
):
    from types import SimpleNamespace

    codec = ids.HmacSha256ProjectionIdentifierCodec(b"key", key_version="key-v1")
    encoder = helper.MembershipKeyEncoder(codec, "generation")
    if stage == "after_prefix":
        encoder.entity(7)
        encoder.canonical(7)
    original = owner.__getattribute__

    def changed_lookup(self, name):
        if owner is ids.HmacSha256ProjectionIdentifierCodec and name == "encode":
            return lambda *args, **kwargs: SimpleNamespace(value="f" * 64)
        if owner is ids._ProjectionIdentifierInput and name == "canonical_source":
            return "999"
        if owner is ids.OpaqueProjectionKey and name == "value":
            return "f" * 64
        return original(self, name)

    with monkeypatch.context() as changed:
        changed.setattr(owner, delegate, changed_lookup)
        if stage == "before_import":
            importlib.reload(helper)
            encoder = helper.MembershipKeyEncoder(codec, "generation")
        for method, domain, kwargs in (
            (
                encoder.entity,
                ids.ProjectionIdentifierDomain.ENTITY,
                {"generation": "generation"},
            ),
            (
                encoder.canonical,
                ids.ProjectionIdentifierDomain.AUTOMATIC_CANONICAL_IDENTITY,
                {},
            ),
        ):
            assert method(7) == codec.encode(domain, source=7, **kwargs).value
            assert encoder._generic and encoder._prefixes is None
            assert encoder._canonical == {}
    importlib.reload(helper)


@pytest.mark.parametrize("stage", ["before_import", "after_prefix"])
@pytest.mark.parametrize("owner", ["_ProjectionIdentifierInput", "OpaqueProjectionKey"])
def test_replaced_record_allocator_uses_current_generic(owner, stage):
    # CPython heap-type __new__ slot restoration can outlive delattr: isolate it.
    import subprocess

    source = """
import importlib
import sys
sys.path.insert(0, "aquillm")
from apps.knowledge_graph.projection import identifiers as ids
from apps.knowledge_graph.projection import membership_identifiers as helper
codec = ids.HmacSha256ProjectionIdentifierCodec(b"key", key_version="key-v1")
encoder = helper.MembershipKeyEncoder(codec, "generation")
if STAGE == "after_prefix":
    encoder.entity(7)
    encoder.canonical(7)
def changed_new(*args, **kwargs):
    raise RuntimeError("record allocation replaced")
setattr(getattr(ids, OWNER), "__new__", staticmethod(changed_new))
if STAGE == "before_import":
    importlib.reload(helper)
    encoder = helper.MembershipKeyEncoder(codec, "generation")
for operation in (
    lambda: codec.encode(ids.ProjectionIdentifierDomain.ENTITY,
                         generation="generation", source=7),
    lambda: encoder.entity(7),
    lambda: encoder.canonical(7),
):
    try:
        operation()
    except RuntimeError as error:
        assert str(error) == "record allocation replaced"
    else:
        raise AssertionError("current record allocation callback was bypassed")
assert encoder._generic and encoder._prefixes is None and encoder._canonical == {}
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "OWNER=" + repr(owner) + ";STAGE=" + repr(stage) + "\n" + source,
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
