"""The portable OR branch repeats bound column parameters in SQL order."""
from types import SimpleNamespace

import pytest
from django.db import connection
from django.db.models import CharField, IntegerField, Value

from apps.documents.models import RawTextDocument
from apps.documents.services.representation_tuple_predicate import ExactTupleMembership

pytestmark = pytest.mark.django_db(databases=["default"])


def test_fallback_prepares_and_orders_each_repeated_lhs_parameter(monkeypatch):
    expression = ExactTupleMembership(("pkid", "title"), ((7, "L'HS"), (8, "right")))
    # Real parameterized Django expressions exercise the fallback's repeated
    # LHS slots; ordinary F/Col children have no LHS parameters to reveal this.
    expression.set_source_expressions([
        Value(7, output_field=IntegerField()), Value("L'HS", output_field=CharField()),
    ])
    before = expression.get_source_expressions()
    compiler = RawTextDocument.objects.all().query.get_compiler(using="default")
    original = compiler.compile
    visits = []

    def delegated(node):
        visits.append(node)
        return original(node)

    monkeypatch.setattr(compiler, "compile", delegated)
    # Only SQL dialect selection changes. Field preparation and the compiler
    # are real installed backend components; execute the standard OR on PG.
    dialect = SimpleNamespace(vendor="generic", ops=connection.ops)
    sql, params = expression.as_sql(compiler, dialect)
    assert visits == before
    assert params == [7, 7, "L'HS", "L'HS", 7, 8, "L'HS", "right"]
    assert "L'HS" not in sql and "right" not in sql
    with connection.cursor() as cursor:
        cursor.execute("SELECT " + sql, params)
        assert cursor.fetchone() == (True,)
    assert expression.get_source_expressions() == before
    assert expression.as_sql(compiler, dialect) == (sql, params)
