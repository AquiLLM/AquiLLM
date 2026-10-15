"""Unsupported preparations retain the unmodified generic SQL/error path."""

from uuid import UUID

import pytest
from django.db import connections
from django.db.backends.postgresql.base import DatabaseWrapper as StockPostgres
from django.db.models import F, UUIDField, Value
from django.db.models.expressions import Col

from apps.documents.models import RawTextDocument, TextChunk
from apps.documents.services._uuid_membership import uuid_membership
from apps.documents.services.representation_tuple_predicate import ExactTupleMembership

from .uuid_preparation_support import compile_query, fields, forbid_cursor, predicate_in


def outcome(predicate, compiler, connection):
    try:
        return predicate.as_sql(compiler, connection)
    except (TypeError, ValueError) as error:
        return type(error), error.args


@pytest.mark.parametrize(
    ("columns", "rows"),
    [
        ((), ()),
        (("id",), []),
        (("id",), ([UUID(int=0)],)),
        (("id",), ((None,),)),
        (("id",), ((True,),)),
        (("id", "pkid"), ((UUID(int=0),),)),
    ],
)
def test_malformed_rows_retain_constructor_errors(columns, rows):
    with pytest.raises((TypeError, ValueError)) as generic:
        ExactTupleMembership(columns, rows)
    with pytest.raises(type(generic.value), match=str(generic.value)):
        uuid_membership(columns, rows, using="default", fields=fields())


def test_uuid_subclass_retains_exact_scalar_rejection():
    class CustomUUID(UUID):
        pass

    rows = ((CustomUUID(int=1),),)
    with pytest.raises(TypeError, match="invalid_exact_tuple_scalar"):
        uuid_membership(("id",), rows, using="default", fields=fields())


def test_eager_preparation_exception_is_never_replaced_by_membership(monkeypatch):
    rows = ((UUID(int=5),),)

    def rejected(field, value, connection, prepared=False):
        raise ValueError("preparation rejected")

    monkeypatch.setattr(UUIDField, "get_db_prep_value", rejected)
    with pytest.raises(ValueError, match="preparation rejected"):
        uuid_membership(("id",), rows, using="default", fields=fields())
    with pytest.raises(ValueError, match="preparation rejected"):
        compile_query(
            RawTextDocument.objects.filter(ExactTupleMembership(("id",), rows))
        )


def test_multiple_columns_keep_generic_tuple_preparation(monkeypatch):
    forbid_cursor(monkeypatch)
    columns, rows = ("id", "pkid"), ((UUID(int=5), 7),)
    predicate = uuid_membership(columns, rows, using="default", fields=fields())
    assert type(predicate) is ExactTupleMembership
    assert compile_query(RawTextDocument.objects.filter(predicate)) == compile_query(
        RawTextDocument.objects.filter(ExactTupleMembership(columns, rows))
    )


def test_truthy_native_feature_still_requires_exact_true(monkeypatch):
    monkeypatch.setattr(connections["default"].features, "has_native_uuid_field", 1)
    rows = ((UUID(int=5),),)
    predicate = uuid_membership(("id",), rows, using="default", fields=fields())
    assert type(predicate) is ExactTupleMembership
    assert compile_query(RawTextDocument.objects.filter(predicate)) == compile_query(
        RawTextDocument.objects.filter(ExactTupleMembership(("id",), rows))
    )


@pytest.mark.parametrize("rows", [(), (("wrong UUID scalar",),), ((1,),)])
def test_empty_or_wrong_scalar_preserves_compiler_outcome(rows, monkeypatch):
    forbid_cursor(monkeypatch)
    actual = RawTextDocument.objects.filter(
        uuid_membership(
            ("id",),
            rows,
            using="default",
            fields=fields(),
        )
    )
    expected = RawTextDocument.objects.filter(ExactTupleMembership(("id",), rows))
    compiler = actual.query.get_compiler(using="default")
    assert outcome(predicate_in(actual), compiler, connections["default"]) == outcome(
        predicate_in(expected),
        compiler,
        connections["default"],
    )


@pytest.mark.parametrize(
    "change",
    [
        "different_wrapper",
        "different_alias",
        "custom_backend",
        "nonnative",
        "vendor",
        "instance_override",
        "field_subclass",
        "foreign_key",
        "value",
        "transform",
        "other_field",
    ],
)
def test_failed_guard_really_prepares_original_rows(change, monkeypatch):
    forbid_cursor(monkeypatch)
    connection = connections["default"]
    rows = ((UUID(int=3),), (UUID(int=3),))
    actual = uuid_membership(("id",), rows, using="default", fields=fields())
    generic = ExactTupleMembership(("id",), rows)
    query = RawTextDocument.objects.filter(actual)
    resolved = predicate_in(query)
    compiler = query.query.get_compiler(using="default")
    generic = generic.resolve_expression(query.query)

    if change == "different_wrapper":
        connection = type(connection)(
            connection.settings_dict.copy(), alias=connection.alias
        )
    elif change == "different_alias":
        monkeypatch.setitem(
            connections.databases, "uuid_fallback", connection.settings_dict.copy()
        )
        changed = query.using("uuid_fallback")
        compiler = changed.query.get_compiler(using="uuid_fallback")
        connection = connections["uuid_fallback"]
    elif change == "custom_backend":

        class CustomBackend(type(connection)):
            pass

        connection = CustomBackend(
            connection.settings_dict.copy(), alias=connection.alias
        )
    elif change == "nonnative":
        monkeypatch.setattr(connection.features, "has_native_uuid_field", False)
    elif change == "vendor":
        monkeypatch.setattr(connection, "vendor", "generic")
    elif change == "instance_override":
        field = RawTextDocument._meta.get_field("id")

        def override(value, connection, prepared=False):
            return UUID(int=11)

        monkeypatch.setitem(field.__dict__, "get_db_prep_value", override)
    else:
        if change == "field_subclass":

            class CustomField(UUIDField):
                pass

            field = CustomField()
            field.set_attributes_from_name("id")
            expression = Col("custom", field)
        elif change == "foreign_key":
            expression = Col(
                "custom", TextChunk._meta.get_field("representation_generation")
            )
        elif change == "other_field":
            expression = Col("custom", TextChunk._meta.get_field("doc_id"))
        elif change == "transform":
            from django.db.models.functions import Cast

            expression = Cast(F("id"), output_field=UUIDField()).resolve_expression(
                query.query
            )
        else:
            expression = Value(UUID(int=3), output_field=UUIDField())
        resolved.set_source_expressions([expression])
        generic.set_source_expressions([expression])

    preparation = []
    original = ExactTupleMembership._prepare

    def prepare(expression, value, database):
        preparation.append(value)
        return original(expression, value, database)

    monkeypatch.setattr(ExactTupleMembership, "_prepare", staticmethod(prepare))
    result = outcome(resolved, compiler, connection)
    assert preparation == [row[0] for row in rows]
    assert result == outcome(generic, compiler, connection)


@pytest.mark.parametrize(
    "change",
    [
        "instance_override",
        "field_subclass",
        "custom_backend",
        "nonnative",
        "foreign_key",
        "unbound_uuid",
    ],
)
def test_factory_rejects_unsupported_metadata_before_eager_preparation(
    change, monkeypatch
):
    connection = connections["default"]
    supplied = fields()
    if change == "instance_override":
        original = supplied[0].get_db_prep_value
        monkeypatch.setitem(supplied[0].__dict__, "get_db_prep_value", original)
    elif change == "field_subclass":

        class CustomField(UUIDField):
            pass

        supplied = (CustomField(),)
    elif change == "custom_backend":

        class CustomBackend(type(connection)):
            pass

        monkeypatch.setattr(connection, "__class__", CustomBackend)
    elif change == "foreign_key":
        supplied = (TextChunk._meta.get_field("representation_generation"),)
    elif change == "unbound_uuid":
        supplied = (UUIDField(),)
    else:
        monkeypatch.setattr(connection.features, "has_native_uuid_field", False)
    predicate = uuid_membership(
        ("id",), ((UUID(int=3),),), using="default", fields=supplied
    )
    assert type(predicate) is ExactTupleMembership
    prepared = []
    original_prepare = ExactTupleMembership._prepare

    def prepare(expression, value, database):
        prepared.append(value)
        return original_prepare(expression, value, database)

    monkeypatch.setattr(ExactTupleMembership, "_prepare", staticmethod(prepare))
    actual = compile_query(RawTextDocument.objects.filter(predicate))
    assert prepared == [UUID(int=3)]
    assert actual == compile_query(
        RawTextDocument.objects.filter(ExactTupleMembership(("id",), ((UUID(int=3),),)))
    )


def test_stock_backend_is_also_eligible_without_connecting(monkeypatch):
    connection = connections["default"]
    stock = StockPostgres(connection.settings_dict.copy(), alias="uuid_stock")
    monkeypatch.setattr(connections._connections, "uuid_stock", stock, raising=False)
    forbid_cursor(monkeypatch, "uuid_stock")
    rows = ((UUID(int=5),),)
    predicate = uuid_membership(("id",), rows, using="uuid_stock", fields=fields())
    assert type(predicate) is not ExactTupleMembership
    actual = RawTextDocument.objects.using("uuid_stock").filter(predicate)
    expected = RawTextDocument.objects.using("uuid_stock").filter(
        ExactTupleMembership(("id",), rows)
    )
    assert compile_query(actual) == compile_query(expected)
