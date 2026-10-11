"""Bound execute and server-side FETCH against all enclosing DB deadlines."""
from contextlib import contextmanager
from contextvars import ContextVar

_SCOPES = ContextVar("knowledge_graph_database_cursor_scopes", default=())
_MISSING = object()


@contextmanager
def bounded_cursor_calls(connection, active, previous):
    token = _SCOPES.set((*_SCOPES.get(), (connection, active, previous)))
    originals = {}

    def remaining():
        return min(operation.remaining_ms() for current, operation, _ in _SCOPES.get()
                   if current is connection)

    def before():
        allowance = remaining()
        limits = []
        for name in ("lock_timeout", "statement_timeout"):
            inherited = [settings[name] for current, _, settings in _SCOPES.get()
                         if current is connection and settings[name]]
            limits.append(min(allowance, *inherited) if inherited else allowance)
        with connection.wrap_database_errors, connection.connection.cursor() as control:
            # A caller may tighten a setting inside an enclosing scope. Keep
            # that limit even on SAVEPOINT, before a nested scope can read it.
            control.execute(
                "SELECT set_config('lock_timeout', LEAST(%s, COALESCE(NULLIF("
                "(SELECT setting::bigint FROM pg_settings WHERE name='lock_timeout'),"
                "0),%s))::text || 'ms', true), "
                "set_config('statement_timeout', LEAST(%s, COALESCE(NULLIF("
                "(SELECT setting::bigint FROM pg_settings WHERE name='statement_timeout'),"
                "0),%s))::text || 'ms', true)",
                [limits[0], limits[0], limits[1], limits[1]],
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
                before()
                result = _original(*args, **kwargs)
                remaining()
                return result

            setattr(cursor, name, fetch)

    def execute(executor, sql, params, many, context):
        install(context["cursor"])
        before()
        result = executor(sql, params, many, context)
        remaining()
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
