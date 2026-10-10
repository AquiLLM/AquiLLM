"""Finite metadata-only database work within an existing publication lease."""

from contextlib import contextmanager
from time import monotonic

from django.db import connections, transaction
from django.db.models import Func, TextField

from lib.llm.answer_publication import AnswerPublicationRejected


class _SourceSHA256(Func):
    """PostgreSQL's built-in bytea SHA-256, requiring no pgcrypto extension."""

    template = "encode(sha256(convert_to(%(expressions)s, 'UTF8')), 'hex')"
    output_field = TextField()


class _StrippedSource(Func):
    template = "regexp_replace(%(expressions)s, '^[[:space:]]+|[[:space:]]+$', '', 'g')"
    output_field = TextField()


@contextmanager
def _bounded_database(alias, seconds):
    if seconds <= 0:
        raise AnswerPublicationRejected("authority_unavailable")
    with transaction.atomic(using=alias):
        connection = connections[alias]
        if connection.vendor != "postgresql":
            raise AnswerPublicationRejected("authority_unavailable")
        with connection.cursor() as cursor:
            cursor.execute("SELECT current_setting('statement_timeout')")
            previous = cursor.fetchone()[0]
            cursor.execute(
                "SELECT set_config('statement_timeout', %s, true)",
                [f"{max(1, int(seconds * 1000))}ms"],
            )
        deadline = monotonic() + seconds

        def limit_query(execute, sql, params, many, context):
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise AnswerPublicationRejected("authority_unavailable")
            execute(
                "SELECT set_config('statement_timeout', %s, true)",
                [f"{max(1, int(remaining * 1000))}ms"],
                False,
                context,
            )
            return execute(sql, params, many, context)

        try:
            with connection.execute_wrapper(limit_query):
                yield
        except BaseException:
            # Atomic rollback also restores SET LOCAL; preserve the real failure.
            raise
        else:
            if not connection.needs_rollback:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT set_config('statement_timeout', %s, true)", [previous]
                    )
