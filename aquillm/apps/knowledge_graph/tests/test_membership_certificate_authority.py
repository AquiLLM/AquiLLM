"""Fresh serving proof accepts only native signed evidence."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from apps.knowledge_graph.projection import (
    membership_certificate_contracts as contracts,
)
from apps.knowledge_graph.projection import membership_source_fingerprint as source
from apps.knowledge_graph.retrieval import membership_certificate_authority as authority
from apps.knowledge_graph.tests.test_membership_certificate_contracts import evidence


@pytest.fixture
def proof(monkeypatch):
    codec, payload, projection, fingerprint = evidence()
    certificate = SimpleNamespace(
        projection_id=projection.id,
        payload=payload,
        signature=contracts.sign_membership_certificate(payload, codec),
    )
    calls = []

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, sql, params):
            calls.append((sql, params))

        def fetchmany(self, size):
            return [(3, 2, 2, False, fingerprint.input_sha256, "read committed")]

    database = SimpleNamespace(vendor="postgresql", cursor=Cursor)
    monkeypatch.setattr(authority, "connections", {"default": database})
    monkeypatch.setattr(source, "connections", {"default": database})
    monkeypatch.setattr(authority, "_certificate_rows", lambda **kw: (certificate,))

    def load(check=lambda: None):
        return authority.certified_membership_checksums(
            projections=(projection,), using="default", codec=codec, check=check
        )

    return SimpleNamespace(**locals())


def test_hit_reads_fresh_sql_each_boundary(proof):
    assert proof.load() == {proof.projection.id: "a" * 64}
    assert proof.load() == {proof.projection.id: "a" * 64}
    assert len(proof.calls) == 2
    assert all("transaction_isolation" in sql for sql, _ in proof.calls)


@pytest.mark.parametrize("field,value", [("signature", "0" * 64), ("payload", {})])
def test_invalid_certificate_is_miss(proof, field, value):
    setattr(proof.certificate, field, value)
    assert proof.load() == {}
    assert not proof.calls


def test_source_change_is_miss(proof):
    proof.fingerprint.input_sha256 = "c" * 64
    assert proof.load() == {}


def test_certificate_is_reauthenticated_after_fresh_source_read(proof):
    original_cursor = proof.database.cursor

    class Cursor(original_cursor):
        def execute(self, sql, params):
            super().execute(sql, params)
            proof.certificate.signature = "0" * 64

    proof.database.cursor = Cursor
    assert proof.load() == {}
    assert len(proof.calls) == 1


@pytest.mark.parametrize("state", ["pending", "building"])
def test_nonready_is_not_certified(proof, state):
    proof.projection.state = state
    assert proof.load() == {}
    assert not proof.calls


def test_no_certificate_is_miss(proof, monkeypatch):
    monkeypatch.setattr(authority, "_certificate_rows", lambda **kw: ())
    assert proof.load() == {}
    assert not proof.calls


def test_nonpostgres_is_miss(proof):
    proof.database.vendor = "sqlite"
    assert proof.load() == {}
    assert not proof.calls


def test_custom_codec_is_miss(proof):
    assert (
        authority.certified_membership_checksums(
            projections=(proof.projection,),
            using="default",
            codec=object(),
            check=lambda: None,
        )
        == {}
    )
    assert not proof.calls


def test_deadline_propagates(proof):
    def expired():
        raise TimeoutError("expired")

    with pytest.raises(TimeoutError):
        proof.load(expired)


def _replace_rows(proof, rows):
    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def execute(self, *a):
            pass

        def fetchmany(self, size):
            return rows

    proof.database.cursor = Cursor


def test_partial_fingerprint_result_rejects(proof):
    _replace_rows(proof, [])
    with pytest.raises(ValueError, match="coverage"):
        proof.load()


def test_duplicate_links_reject_when_certificate_starts_source_proof(proof):
    _replace_rows(proof, [(3, 2, 3, True, "b" * 64, "read committed")])
    with pytest.raises(ValueError, match="duplicate"):
        proof.load()


def test_source_scan_covers_only_authenticated_projection_candidates(proof):
    from copy import copy
    from uuid import UUID

    missing = copy(proof.projection)
    missing.id, missing.collection_id = UUID(int=99), 99
    result = authority.certified_membership_checksums(
        projections=(proof.projection, missing),
        using="default",
        codec=proof.codec,
        check=lambda: None,
    )
    assert result == {proof.projection.id: "a" * 64}
    assert proof.calls[0][1] == [[3], [3]]


def test_wrong_isolation_rejects(proof):
    _replace_rows(proof, [(3, 2, 2, False, "b" * 64, "repeatable read")])
    with pytest.raises(ValueError, match="isolation"):
        proof.load()


def test_native_replacement_disables_shortcut(proof, monkeypatch):
    monkeypatch.setattr(
        contracts.memberships, "membership_decision_checksum", lambda *a, **kw: "a" * 64
    )
    assert proof.load() == {}
    assert not proof.calls


def test_source_reader_replacement_disables_shortcut(proof, monkeypatch):
    monkeypatch.setattr(source, "load_membership_source_fingerprints", lambda **kw: {})
    assert proof.load() == {}
    assert not proof.calls


def test_wrong_actual_key_is_miss(proof):
    wrong = contracts.ids.HmacSha256ProjectionIdentifierCodec(
        b"wrong", key_version="key-v1"
    )
    assert (
        authority.certified_membership_checksums(
            projections=(proof.projection,),
            using="default",
            codec=wrong,
            check=lambda: None,
        )
        == {}
    )


def test_native_change_during_source_read_rejects(proof, monkeypatch):
    original_cursor = proof.database.cursor

    class Cursor(original_cursor):
        def execute(self, sql, params):
            super().execute(sql, params)
            monkeypatch.setattr(
                contracts.memberships,
                "membership_decision_checksum",
                lambda *a, **kw: "a" * 64,
            )

    proof.database.cursor = Cursor
    with pytest.raises(ValueError, match="native identity changed"):
        proof.load()


def test_source_failure_is_never_a_certificate_miss(proof):
    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, *args):
            raise RuntimeError("source context lost")

    proof.database.cursor = Cursor
    with pytest.raises(RuntimeError, match="source context lost"):
        proof.load()


@pytest.mark.parametrize(
    "name",
    ["load_membership_source_fingerprints", "_SQL", "MembershipSourceFingerprintV1"],
)
def test_source_substitution_before_first_helper_import_is_not_blessed(name):
    import os
    import subprocess
    import sys

    script = """
import django
django.setup()
from apps.knowledge_graph.projection import membership_source_fingerprint as source
from apps.knowledge_graph.projection.identifiers import (
    HmacSha256ProjectionIdentifierCodec,
)
setattr(source, NAME, 'substituted SQL' if NAME == '_SQL' else lambda *a, **kw: {})
from apps.knowledge_graph.retrieval import membership_certificate_authority as authority
class ForbiddenConnection:
    def __getitem__(self, alias):
        raise AssertionError('unsupported proof attempted database access')
authority.connections = ForbiddenConnection()
codec = HmacSha256ProjectionIdentifierCodec(b'secret', key_version='key-v1')
assert authority.certified_membership_checksums(
    projections=(object(),), using='default', codec=codec, check=lambda: None) == {}
"""
    result = subprocess.run(
        [sys.executable, "-c", "NAME=" + repr(name) + "\n" + script],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        env=dict(os.environ),
        timeout=45,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name", ["_SQL", "MembershipSourceFingerprintV1"])
def test_source_structure_replacement_disables_shortcut(proof, monkeypatch, name):
    monkeypatch.setattr(source, name, object())
    assert proof.load() == {}
    assert not proof.calls


@pytest.mark.parametrize("name", ["_SQL", "MembershipSourceFingerprintV1"])
def test_source_structure_change_during_read_rejects(proof, monkeypatch, name):
    original_cursor = proof.database.cursor

    class Cursor(original_cursor):
        def execute(self, sql, params):
            super().execute(sql, params)
            monkeypatch.setattr(source, name, object())

    proof.database.cursor = Cursor
    with pytest.raises(ValueError, match="native identity changed"):
        proof.load()
