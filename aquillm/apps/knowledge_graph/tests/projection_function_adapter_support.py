"""Pure fixed-function adapter recorder; it supplies no source authority proof."""
from contextlib import contextmanager


def recording_connection(monkeypatch, module):
    calls, scopes = [], []

    class Cursor:
        description = (("projection_id",), ("owner",), ("expires_at",), ("attempt_count",))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def execute(self, sql, params):
            calls.append((sql, tuple(params)))

        def fetchone(self):
            return None

    class Connection:
        def cursor(self):
            return Cursor()

    @contextmanager
    def scope(alias, *, operation):
        scopes.append((alias, operation))
        yield operation

    monkeypatch.setattr(module, "connections", {"projection_state": Connection()})
    monkeypatch.setattr(module, "projection_database", scope)
    return calls, scopes
