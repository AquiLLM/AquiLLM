"""Canonical SQL bytes stay compatible without timing out on retained sources."""
import json
from hashlib import sha256
from importlib import import_module

import pytest
from django.db import connection, transaction
from django.db.migrations.executor import MigrationExecutor

from .representation_support import retained as retained
from .representation_support import upload as upload
from .test_representation_schema import capture

pytestmark = pytest.mark.django_db(transaction=True)
BEFORE = [("apps_documents", "0015_chunk_delete_lifecycle_state")]
HISTORICAL_STRING_SQL = import_module(
    "apps.documents.migrations._representation_identity_sql"
).IDENTITY_SQL.split("CREATE FUNCTION representation_json_v1", 1)[0]


@pytest.fixture
def previous_serializer():
    # Execute the applied historical implementation under a temporary name.
    sql = HISTORICAL_STRING_SQL.replace(
        "representation_string_v1", "pg_temp.representation_string_previous_v1"
    )
    with connection.cursor() as cursor:
        cursor.execute(sql)
    try:
        yield
    finally:
        with connection.cursor() as cursor:
            cursor.execute(
                "DROP FUNCTION pg_temp.representation_string_previous_v1(text)"
            )


@pytest.mark.parametrize("value", [
    "",
    "plain ASCII / with spaces and punctuation: []{}",
    'quote " and backslash \\ with literal \\u007f',
    "".join(chr(code) for code in range(1, 128)),
    "\x7f\x80\u00ff\u0100\u07ff\u0800\ud7ff\ue000\uffff",
    "\U00010000\U000103ff\U00010400\U0010ffff",
    "\u4e2d\u6587 caf\u00e9 e\u0301 \U0001f600\n\t\r\b\f",
    "a" * 8192 + '\n"\\\x7f\u4e2d\U0001f600' + "z" * 8192,
], ids=["empty", "ascii", "escapes", "controls-and-del", "bmp-boundaries",
        "surrogate-pair-boundaries", "mixed-unicode", "long-ascii-runs"])
def test_string_bytes_match_previous_sql_and_python(value, previous_serializer):
    expected = json.dumps(value, ensure_ascii=True).encode("ascii")
    with connection.cursor() as cursor:
        cursor.execute("SELECT representation_string_v1(%s), "
                       "pg_temp.representation_string_previous_v1(%s)", [value, value])
        current, previous = cursor.fetchone()
    assert current.encode("ascii") == previous.encode("ascii") == expected


def test_string_null_keeps_strict_contract(previous_serializer):
    with connection.cursor() as cursor:
        cursor.execute("SELECT representation_string_v1(NULL), "
                       "pg_temp.representation_string_previous_v1(NULL)")
        assert cursor.fetchone() == (None, None)


def test_nested_json_bytes_and_digest_match_python():
    value = {"\U0001f600": ["\x7f", True, None, 17], "z": {"\u00e9": '\\"\n'},
             "a": "\U0010ffff", "": []}
    expected = json.dumps(value, ensure_ascii=True, sort_keys=True,
                          separators=(",", ":")).encode("ascii")
    with connection.cursor() as cursor:
        cursor.execute("""SELECT canonical,
            encode(sha256(convert_to(canonical, 'UTF8')), 'hex')
            FROM (SELECT representation_json_v1(%s::jsonb) AS canonical) encoded""",
                       [json.dumps(value)])
        canonical, digest = cursor.fetchone()
    assert canonical.encode("ascii") == expected
    assert digest == sha256(expected).hexdigest()


def test_large_mixed_source_serializes_within_snapshot_statement_timeout():
    unit = ('Source paragraph: "quoted" \\ path; '
            'caf\u00e9 \u4e2d\u6587 \U0001f600.\n\t\x01\x7f')
    value = (unit * (320_000 // len(unit) + 1))[:320_000]
    expected = json.dumps(value, ensure_ascii=True).encode("ascii")
    with transaction.atomic(), connection.cursor() as cursor:
        cursor.execute("SET LOCAL statement_timeout = '20s'")
        cursor.execute("SELECT representation_string_v1(%s)", [value])
        assert cursor.fetchone()[0].encode("ascii") == expected


def test_serializer_migration_roundtrip_preserves_captured_history(retained):
    doc, _, _ = retained
    _, snapshot = capture(doc)
    original = type(snapshot).objects.values().get(pk=snapshot.pk)
    leaves = MigrationExecutor(connection).loader.graph.leaf_nodes()
    with connection.cursor() as cursor:
        cursor.execute("SELECT 'representation_string_v1(text)'::regprocedure::oid")
        original_oid = cursor.fetchone()[0]
    try:
        MigrationExecutor(connection).migrate(BEFORE)
        with connection.cursor() as cursor:
            cursor.execute("""SELECT oid, prosrc, provolatile, proisstrict FROM pg_proc
                WHERE oid = 'representation_string_v1(text)'::regprocedure""")
            oid, body, volatility, strict = cursor.fetchone()
        assert (oid, volatility, strict) == (original_oid, "i", True)
        assert body == HISTORICAL_STRING_SQL.split("$$", 2)[1]
        MigrationExecutor(connection).migrate(leaves)
        with connection.cursor() as cursor:
            cursor.execute("""SELECT oid, provolatile, proisstrict FROM pg_proc
                WHERE oid = 'representation_string_v1(text)'::regprocedure""")
            assert cursor.fetchone() == (original_oid, "i", True)
        assert type(snapshot).objects.values().get(pk=snapshot.pk) == original
    finally:
        MigrationExecutor(connection).migrate(leaves)
