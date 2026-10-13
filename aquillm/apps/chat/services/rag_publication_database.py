"""Finite metadata-only database work within an existing publication lease."""

from contextlib import contextmanager
from math import isfinite
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
def _control_cursor(connection):
    # Raw control cursors do not recurse through application execute wrappers or
    # accidentally reuse a named/server-side cursor. Keep Django error mapping.
    with connection.wrap_database_errors, connection.connection.cursor() as cursor:
        yield cursor


def _read_limits(connection):
    with _control_cursor(connection) as cursor:
        cursor.execute("SELECT name, setting::bigint FROM pg_settings "
                       "WHERE name IN ('statement_timeout', 'lock_timeout')")
        return dict(cursor.fetchall())


def _remaining_ms(frames):
    remaining = min(frame[0] for frame in frames) - monotonic()
    if remaining <= 0:
        raise AnswerPublicationRejected("authority_unavailable")
    return max(1, int(remaining * 1000))


def _apply_limits(connection, frames):
    remaining = _remaining_ms(frames)
    limits = [min([remaining, *(frame[1][name] for frame in frames
                                if frame[1][name] > 0)])
              for name in ("statement_timeout", "lock_timeout")]
    # Current ambient limits may have been tightened after scope entry. Clamping
    # them in this same control query also protects a nested atomic SAVEPOINT
    # from an outer execute wrapper widening the operator's newer limit.
    with _control_cursor(connection) as cursor:
        cursor.execute(
            "WITH limits AS (SELECT name, setting::bigint AS value FROM pg_settings "
            "WHERE name IN ('statement_timeout', 'lock_timeout')) "
            "SELECT set_config('statement_timeout', "
            "LEAST(%s, COALESCE(NULLIF((SELECT value FROM limits WHERE name = "
            "'statement_timeout'), 0), %s))::text || 'ms', true), "
            "set_config('lock_timeout', "
            "LEAST(%s, COALESCE(NULLIF((SELECT value FROM limits WHERE name = "
            "'lock_timeout'), 0), %s))::text || 'ms', true)",
            [limits[0], limits[0], limits[1], limits[1]],
        )
    _remaining_ms(frames)


def _restore_limits(connection, previous):
    try:
        with _control_cursor(connection) as cursor:
            cursor.execute(
                "SELECT set_config('statement_timeout', %s, true), "
                "set_config('lock_timeout', %s, true)",
                [f"{previous['statement_timeout']}ms", f"{previous['lock_timeout']}ms"],
            )
    except BaseException:
        # A pooled connection must never retain an unknown timeout policy.
        connection.close()
        raise


@contextmanager
def _bounded_database(alias, seconds):
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not isfinite(seconds) or seconds <= 0:
        raise AnswerPublicationRejected("authority_unavailable")
    deadline = monotonic() + seconds
    connection = connections[alias]
    if connection.vendor != "postgresql":
        raise AnswerPublicationRejected("authority_unavailable")
    connection.ensure_connection()
    # Capture before entering atomic: an outer execute wrapper runs for the
    # inner SAVEPOINT and must not replace the inherited operator limit first.
    previous = _read_limits(connection)
    frame = (deadline, previous)
    _remaining_ms((frame,))
    frames = getattr(connection, "_rag_publication_db_frames", None)
    if frames is None:
        frames = []
        connection._rag_publication_db_frames = frames

    def limit_query(execute, sql, params, many, context):
        # An inner atomic rollback can run while an outer wrapper is active.
        # Aborted transactions cannot execute control SELECTs until rollback;
        # cleanup must also work after a deadline. This admits no data query.
        if isinstance(sql, str) and sql.lstrip().upper().startswith((
            "ROLLBACK TO SAVEPOINT ", "RELEASE SAVEPOINT ",
        )):
            return execute(sql, params, many, context)
        _apply_limits(connection, frames)
        result = execute(sql, params, many, context)
        _remaining_ms(frames)
        return result

    try:
        with transaction.atomic(using=alias):
            frames.append(frame)
            try:
                with connection.execute_wrapper(limit_query):
                    _apply_limits(connection, frames)
                    yield
                    _remaining_ms(frames)
            finally:
                frames.pop()
            if not connection.needs_rollback:
                _restore_limits(connection, previous)
    finally:
        # No expired inner wrapper/frame surrounds rollback or savepoint release.
        # Atomic rollback restores SET LOCAL on failure; success restores above.
        if not frames:
            del connection._rag_publication_db_frames
