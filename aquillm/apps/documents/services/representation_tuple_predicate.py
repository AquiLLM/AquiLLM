"""Private exact non-null tuple predicates with normal ORM alias resolution."""
from uuid import UUID

from django.db.models import (
    BooleanField, CharField, Expression, F, IntegerField, TextField, UUIDField,
)


class ExactTupleMembership(Expression):
    def __init__(self, columns: tuple[str, ...], rows: tuple[tuple, ...]):
        if (type(columns) is not tuple or not columns
                or any(type(name) is not str or not name for name in columns)
                or type(rows) is not tuple):
            raise ValueError("invalid_exact_tuple_shape")
        for row in rows:
            if type(row) is not tuple or len(row) != len(columns) or any(v is None for v in row):
                raise ValueError("invalid_exact_tuple_shape")
            if any(type(v) not in (str, int, UUID) for v in row):
                raise TypeError("invalid_exact_tuple_scalar")
        super().__init__(output_field=BooleanField())
        self.expressions = tuple(F(name) for name in columns)
        self.rows = rows

    def get_source_expressions(self):
        return list(self.expressions)

    def set_source_expressions(self, expressions):
        self.expressions = tuple(expressions)

    @staticmethod
    def _prepare(expression, value, connection):
        field = expression.output_field
        field = getattr(field, "target_field", field)
        expected = (UUID if isinstance(field, UUIDField) else
                    int if isinstance(field, IntegerField) else
                    str if isinstance(field, (CharField, TextField)) else None)
        if expected is None or type(value) is not expected:
            raise TypeError("invalid_exact_tuple_field_value")
        return field.get_db_prep_value(value, connection, prepared=False)

    def as_sql(self, compiler, connection):
        if not self.rows:
            return "FALSE", []
        columns = [compiler.compile(expression) for expression in self.expressions]
        rows = tuple(tuple(self._prepare(expression, value, connection)
                           for expression, value in zip(self.expressions, row))
                     for row in self.rows)
        if connection.vendor == "postgresql":
            left = ", ".join(sql for sql, _ in columns)
            slot = "(" + ", ".join("%s" for _ in columns) + ")"
            right = ", ".join(slot for _ in rows)
            params = [value for _, values in columns for value in values]
            params.extend(value for row in rows for value in row)
            return f"({left}) IN ({right})", params
        # Equivalent fallback keeps nullable current rows ineligible and retains
        # exact tuple pairing. Repeat LHS parameters in their textual SQL order.
        clauses, params = [], []
        for row in rows:
            fields = []
            for (sql, values), value in zip(columns, row):
                fields.append(f"{sql} = %s")
                params.extend(values)
                params.append(value)
            clauses.append("(" + " AND ".join(fields) + ")")
        return "(" + " OR ".join(clauses) + ")", params
