"""Bound execute and server-side FETCH against all enclosing DB deadlines."""

from contextlib import contextmanager
from contextvars import ContextVar

_SCOPES = ContextVar("knowledge_graph_database_cursor_scopes", default=())
_CALLS = ContextVar("knowledge_graph_database_cursor_calls", default=())
_MISSING = object()


@contextmanager
def bounded_cursor_calls(connection, active, previous):
    token = _SCOPES.set((*_SCOPES.get(), (connection, active, previous)))
    originals = {}

    def remaining():
        return min(
            [
                *(
                    operation.remaining_ms()
                    for current, operation, _ in _SCOPES.get()
                    if current is connection
                ),
                *(call.remaining_ms() for call in _CALLS.get()),
            ]
        )

    @contextmanager
    def call_scope():
        calls = tuple(
            operation.call(settings["statement_timeout"])
            for current, operation, settings in _SCOPES.get()
            if current is connection and hasattr(operation, "call")
        )
        token = _CALLS.set((*_CALLS.get(), *calls))
        try:
            remaining()
            yield calls
            remaining()
        finally:
            _CALLS.reset(token)

    def before(calls):
        allowance = remaining()
        limits = []
        for name in ("lock_timeout", "statement_timeout"):
            inherited = [
                settings[name]
                for current, _, settings in _SCOPES.get()
                if current is connection and settings[name]
            ]
            limits.append(min(allowance, *inherited) if inherited else allowance)
        with connection.wrap_database_errors, connection.connection.cursor() as control:
            # A caller may tighten a setting inside an enclosing scope. Keep
            # that limit even on SAVEPOINT, before a nested scope can read it.
            control.execute(
                "WITH prior AS MATERIALIZED (SELECT "
                "MAX(setting::bigint) FILTER (WHERE name='lock_timeout') AS lock_ms, "
                "MAX(setting::bigint) FILTER (WHERE name='statement_timeout') "
                "AS statement_ms "
                "FROM pg_settings WHERE name IN ('lock_timeout','statement_timeout')) "
                "SELECT lock_ms, statement_ms, "
                "set_config('lock_timeout', LEAST(%s, COALESCE(NULLIF(lock_ms,"
                "0),%s))::text || 'ms', true), "
                "set_config('statement_timeout', LEAST(%s, "
                "COALESCE(NULLIF(statement_ms,"
                "0),%s))::text || 'ms', true) FROM prior",
                [limits[0], limits[0], limits[1], limits[1]],
            )
            prior = control.fetchone()[:2]
            installed = tuple(
                min(limit, inherited or limit)
                for limit, inherited in zip(limits, prior)
            )
            for call in calls:
                call.limit(installed[1])
        remaining()
        return prior, installed

    def restore(settings):
        if not _CALLS.get():
            return  # Preserve the canonical reconciliation scope's policy.
        prior, installed = settings
        with connection.wrap_database_errors, connection.connection.cursor() as control:
            # Preserve any explicit caller change made by the actual query.
            control.execute(
                "SELECT set_config(name, (CASE WHEN setting::bigint="
                "CASE name WHEN 'lock_timeout' THEN %s ELSE %s END THEN "
                "CASE name WHEN 'lock_timeout' THEN %s ELSE %s END "
                "ELSE setting::bigint END)::text || 'ms', true) "
                "FROM pg_settings WHERE name IN ('lock_timeout','statement_timeout')",
                [*installed, *prior],
            )

    def install(cursor):
        if cursor in originals:
            return
        saved = originals[cursor] = {}
        for name in ("fetchone", "fetchmany", "fetchall"):
            saved[name] = cursor.__dict__.get(name, _MISSING)
            # Keep Django's error translator and any enclosing fetch guard.
            original = getattr(cursor, name)

            def fetch(*args, _original=original, **kwargs):
                with call_scope() as calls:
                    settings = before(calls)
                    result = _original(*args, **kwargs)
                    remaining()
                    restore(settings)
                    return result

            setattr(cursor, name, fetch)

    def execute(executor, sql, params, many, context):
        install(context["cursor"])
        with call_scope() as calls:
            settings = before(calls)
            result = executor(sql, params, many, context)
            remaining()
            restore(settings)
            return result

    try:
        yield execute
    finally:
        for cursor, methods in originals.items():
            for name, original in methods.items():
                if original is _MISSING:
                    delattr(cursor, name)
                else:
                    setattr(cursor, name, original)
        _SCOPES.reset(token)
