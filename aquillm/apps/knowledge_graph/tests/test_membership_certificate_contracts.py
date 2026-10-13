"""Signed proof binds exact native semantics, key and projection occurrence."""

from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from apps.knowledge_graph.projection import identifiers as ids
from apps.knowledge_graph.projection import (
    membership_certificate_contracts as contracts,
)


def evidence():
    codec = ids.HmacSha256ProjectionIdentifierCodec(
        b"certificate-test", key_version="key-v1"
    )
    payload = dict(
        format="membership-certificate-v1",
        projection_id=str(UUID(int=1)),
        generation_key=str(UUID(int=2)),
        collection_id=3,
        artifact_id=4,
        schema_version="schema-v1",
        projection_version="projection-v1",
        identifier_key_version="key-v1",
        membership_epoch=0,
        membership_checksum="a" * 64,
        source_format="kg-membership-source-sql-v1",
        source_sha256="b" * 64,
        entity_count=2,
        derivation_identity=contracts.native_membership_identity(codec),
    )
    projection = SimpleNamespace(
        **{
            k: v
            for k, v in payload.items()
            if k
            in {
                "generation_key",
                "collection_id",
                "artifact_id",
                "schema_version",
                "projection_version",
                "identifier_key_version",
                "membership_epoch",
                "membership_checksum",
                "entity_count",
            }
        }
    )
    projection.id = UUID(payload["projection_id"])
    projection.generation_key = UUID(payload["generation_key"])
    projection.state = "ready"
    fingerprint = SimpleNamespace(
        collection_id=3,
        input_sha256="b" * 64,
        entity_count=2,
        source_rows=2,
        duplicate_links=False,
    )
    return codec, payload, projection, fingerprint


def verify(codec, payload, projection, fingerprint, signature=None):
    return contracts.verify_membership_certificate(
        payload=payload,
        signature=signature or contracts.sign_membership_certificate(payload, codec),
        codec=codec,
        projection=projection,
        fingerprint=fingerprint,
    )


def test_valid_certificate_and_wrong_actual_key():
    codec, payload, projection, fingerprint = evidence()
    signature = contracts.sign_membership_certificate(payload, codec)
    assert verify(codec, payload, projection, fingerprint, signature)
    wrong = ids.HmacSha256ProjectionIdentifierCodec(b"rotated", key_version="key-v1")
    assert not verify(wrong, payload, projection, fingerprint, signature)


def test_valid_shape_payload_tampering_breaks_signature():
    codec, payload, projection, fingerprint = evidence()
    signature = contracts.sign_membership_certificate(payload, codec)
    payload["source_sha256"] = "c" * 64
    fingerprint.input_sha256 = "c" * 64
    assert not verify(codec, payload, projection, fingerprint, signature)


@pytest.mark.parametrize(
    "field,value",
    [
        ("collection_id", True),
        ("artifact_id", 0),
        ("membership_epoch", -1),
        ("entity_count", True),
        ("source_sha256", "B" * 64),
        ("projection_id", "bad"),
        ("schema_version", " bad"),
        ("extra", "secret"),
    ],
)
def test_malformed_payload_is_rejected(field, value):
    codec, payload, projection, fingerprint = evidence()
    payload[field] = value
    with pytest.raises((ValueError, TypeError)):
        contracts.sign_membership_certificate(payload, codec)
    assert not verify(codec, payload, projection, fingerprint, "0" * 64)


@pytest.mark.parametrize(
    "field,value",
    [
        ("generation_key", UUID(int=7)),
        ("membership_checksum", "c" * 64),
        ("membership_epoch", 1),
        ("artifact_id", 7),
        ("state", "superseded"),
    ],
)
def test_wrong_occurrence_or_authority_is_rejected(field, value):
    codec, payload, projection, fingerprint = evidence()
    setattr(projection, field, value)
    assert not verify(codec, payload, projection, fingerprint)


def test_duplicate_and_changed_source_are_rejected():
    codec, payload, projection, fingerprint = evidence()
    fingerprint.duplicate_links = True
    assert not verify(codec, payload, projection, fingerprint)
    fingerprint.duplicate_links = False
    fingerprint.input_sha256 = "c" * 64
    assert not verify(codec, payload, projection, fingerprint)


def test_native_replacement_and_custom_codec_are_unsupported(monkeypatch):
    codec, payload, projection, fingerprint = evidence()
    assert contracts.native_membership_identity(SimpleNamespace()) is None

    class Custom(ids.HmacSha256ProjectionIdentifierCodec):
        pass

    assert contracts.native_membership_identity(Custom(b"k", key_version="v")) is None
    monkeypatch.setattr(ids, "_frame_field", lambda value: b"changed")
    assert contracts.native_membership_identity(codec) is None
    assert not verify(codec, payload, projection, fingerprint, "0" * 64)


def test_mid_acceptance_key_change_is_rejected():
    codec, payload, projection, fingerprint = evidence()
    signature = contracts.sign_membership_certificate(payload, codec)

    class ChangingProjection:
        def __getattr__(self, name):
            object.__setattr__(codec, "_key", b"changed-mid-operation")
            return getattr(projection, name)

    assert not verify(codec, payload, ChangingProjection(), fingerprint, signature)


def test_mid_acceptance_derivation_change_is_rejected(monkeypatch):
    codec, payload, projection, fingerprint = evidence()
    signature = contracts.sign_membership_certificate(payload, codec)

    class ChangingProjection:
        def __getattr__(self, name):
            monkeypatch.setattr(contracts.checksums, "_payload", lambda row: {})
            return getattr(projection, name)

    assert not verify(codec, payload, ChangingProjection(), fingerprint, signature)


@pytest.mark.parametrize(
    "module,name",
    [
        ("memberships", "AutomaticCanonicalMembershipV1"),
        ("memberships", "sha256"),
        ("checksums", "_encode"),
        ("checksums", "AutomaticCanonicalMembershipV1"),
        ("records", "_key"),
    ],
)
def test_actual_derivation_global_replacement_fails_closed(monkeypatch, module, name):
    codec, payload, projection, fingerprint = evidence()
    monkeypatch.setattr(getattr(contracts, module), name, object())
    assert contracts.native_membership_identity(codec) is None
    assert not verify(codec, payload, projection, fingerprint, "0" * 64)


def test_equality_friendly_function_replacement_is_rejected(monkeypatch):
    codec, payload, projection, fingerprint = evidence()

    class Equal:
        def __eq__(self, other):
            return True

    monkeypatch.setattr(contracts.memberships, "sha256", Equal())
    assert contracts.native_membership_identity(codec) is None


@pytest.mark.parametrize(
    "module,name",
    [
        ("memberships", "load_automatic_membership_assignments"),
        ("membership_checksum", "_payload"),
        ("membership_identifiers", "MembershipKeyEncoder._integer"),
        ("serialization", "_encode"),
        ("records", "_key"),
        ("membership_source_query", "captured_entity_links"),
    ],
)
def test_mutation_before_first_contract_import_is_not_canonized(module, name):
    import os
    import subprocess
    import sys

    script = """
import django
django.setup()
from importlib import import_module
target = import_module("apps.knowledge_graph.projection." + MODULE)
parts = NAME.split(".")
for part in parts[:-1]:
    target = getattr(target, part)
setattr(target, parts[-1], lambda *args, **kw: ())
from apps.knowledge_graph.projection import membership_certificate_contracts as c
from apps.knowledge_graph.projection import identifiers as ids
codec = ids.HmacSha256ProjectionIdentifierCodec(b"test", key_version="v")
assert c.native_membership_identity(codec) is None
"""
    script = "MODULE=" + repr(module) + "\nNAME=" + repr(name) + "\n" + script
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        env=dict(os.environ),
        timeout=45,
    )
    assert result.returncode == 0, result.stderr
