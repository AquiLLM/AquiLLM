


from __future__ import annotations


class PRQLValidationError(ValueError):
    """Raised when PRQL cannot be compiled."""


def validate_prql(prql_query: str, *, dialect: str = "postgres") -> str:
    """Compile PRQL and return generated SQL for validation/debugging only.

    The returned SQL is not executed here. Callers should use this only to
    verify that generated PRQL is accepted by prqlc.
    """
    try:
        import prql_python as prql
    except ImportError as exc:
        raise PRQLValidationError(
            "prql-python is not installed. Add prql-python to project dependencies."
        ) from exc

    try:
        options = prql.CompileOptions(
            target=f"sql.{dialect}",
            signature_comment=False,
            format=False,
        )
        return prql.compile(prql_query, options)
    except Exception as exc:
        raise PRQLValidationError(f"PRQL validation failed: {exc}") from exc
