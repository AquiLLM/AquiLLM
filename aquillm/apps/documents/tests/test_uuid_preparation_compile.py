"""Optimized single-UUID SQL stays byte-identical to the generic oracle."""

import json
from dataclasses import FrozenInstanceError
from uuid import UUID

import pytest
from django.db import connections
from django.db.models import Exists, OuterRef, UUIDField
from django.db.models.expressions import Col
from django.db.models.sql.compiler import SQLCompiler

from apps.documents.models import RawTextDocument
from apps.documents.services._uuid_membership import uuid_membership
from apps.documents.services.representation_tuple_predicate import ExactTupleMembership

from .uuid_preparation_support import (
    arms,
    compile_query,
    fields,
    forbid_cursor,
    membership_in,
    predicate_in,
    union,
)


@pytest.mark.parametrize("count", [1, 4, 530])
@pytest.mark.parametrize("using", ["default", "uuid_compile"])
def test_all_arms_and_union_match_generic_without_queries(count, using, monkeypatch):
    if using != "default":
        monkeypatch.setitem(
            connections.databases, using, connections.databases["default"].copy()
        )
    forbid_cursor(monkeypatch, using)
    values = tuple(UUID(int=i) for i in range(count))
    rows = tuple((value,) for value in (*values, values[0]))
    generic = ExactTupleMembership(("id",), rows)
    optimized = uuid_membership(("id",), rows, using=using, fields=fields())
    for actual, expected in zip(arms(optimized, using), arms(generic, using)):
        assert compile_query(actual) == compile_query(expected)
    actual = compile_query(union(optimized, using))
    assert actual == compile_query(union(generic, using))
    assert actual[0].count("UNION ALL") == 7
    assert all(
        json.loads(value) == [{"c0": str(row[0])} for row in rows]
        for value in actual[1]
    )


def test_eager_preparation_and_packing_once_eight_independent_lhs_and_fresh_payload(
    monkeypatch,
):
    forbid_cursor(monkeypatch)
    rows = ((UUID(int=0),), (UUID(int=7),), (UUID(int=7),))
    preparation, packing, columns = [], [], []
    original_prepare, original_pack = UUIDField.get_db_prep_value, json.dumps
    original_compile = SQLCompiler.compile

    def prepare(field, value, connection, prepared=False):
        preparation.append((field, value, prepared))
        return original_prepare(field, value, connection, prepared=prepared)

    def pack(value, **kwargs):
        packing.append(value)
        return original_pack(value, **kwargs)

    def compile_node(compiler, node):
        if type(node) is Col:
            columns.append(node)
        return original_compile(compiler, node)

    monkeypatch.setattr(UUIDField, "get_db_prep_value", prepare)
    monkeypatch.setattr(json, "dumps", pack)
    first = uuid_membership(("id",), rows, using="default", fields=fields())
    assert [value for _, value, _ in preparation] == [row[0] for row in rows]
    assert all(prepared is False for _, _, prepared in preparation)
    assert len(packing) == 1
    queries = arms(first)
    monkeypatch.setattr(SQLCompiler, "compile", compile_node)
    outputs = []
    for query in queries:
        resolved = predicate_in(query)
        compiler = query.query.get_compiler(using="default")
        outputs.append(resolved.as_sql(compiler, connections["default"]))
        assert resolved._payload is first._payload
        assert predicate_in(query.all())._payload is first._payload
    assert len(columns) == 8 and len({id(column) for column in columns}) == 8
    assert len(preparation) == 3 and len(packing) == 1
    assert len({id(params) for _, params in outputs}) == 8
    outputs[0][1].append("caller-local mutation")
    assert all(len(params) == 1 for _, params in outputs[1:])
    with pytest.raises(FrozenInstanceError):
        first._payload.json = "changed"
    combined = union(first).all()
    assert all(
        membership_in(query.where)._payload is first._payload
        for query in combined.query.combined_queries
    )
    compile_query(combined)
    second = uuid_membership(("id",), rows, using="default", fields=fields())
    assert second._payload is not first._payload
    assert len(preparation) == 6 and len(packing) == 2


@pytest.mark.parametrize("negated", [False, True])
def test_subquery_relabeling_clones_preserve_sql_and_payload(negated, monkeypatch):
    forbid_cursor(monkeypatch)
    rows = ((UUID(int=9),),)
    optimized = uuid_membership(("id",), rows, using="default", fields=fields())
    generic = ExactTupleMembership(("id",), rows)

    def nested(predicate):
        inner = RawTextDocument.objects.filter(
            ~predicate if negated else predicate, id=OuterRef("id")
        ).order_by()
        return (
            RawTextDocument.objects.annotate(found=Exists(inner))
            .order_by()
            .values("id", "found")
        )

    actual, expected = nested(optimized), nested(generic)
    assert compile_query(actual) == compile_query(expected)
    assert compile_query(actual.all()) == compile_query(expected)
    nested_predicate = membership_in(actual.query.annotations["found"].query.where)
    assert nested_predicate._payload is optimized._payload


def test_every_compilation_preserves_current_lhs_parameters(monkeypatch):
    forbid_cursor(monkeypatch)
    rows = ((UUID(int=5),),)
    optimized = uuid_membership(("id",), rows, using="default", fields=fields())
    actual = RawTextDocument.objects.filter(optimized)
    generic = RawTextDocument.objects.filter(ExactTupleMembership(("id",), rows))
    compiler = actual.query.get_compiler(using="default")
    original = compiler.compile
    calls = []

    def bound_column(node):
        sql, params = original(node)
        assert not params
        calls.append(node)
        return f"COALESCE(%s, {sql})", [UUID(int=12)]

    monkeypatch.setattr(compiler, "compile", bound_column)
    expected = predicate_in(generic).as_sql(compiler, connections["default"])
    result = predicate_in(actual).as_sql(compiler, connections["default"])
    assert result == expected and result[1][0] == UUID(int=12)
    assert len(calls) == 2
