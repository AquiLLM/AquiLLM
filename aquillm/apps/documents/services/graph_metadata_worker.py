"""One private scalar lexical job; connections and publication never cross threads."""
from concurrent.futures import Future
from contextvars import copy_context
from dataclasses import dataclass
from math import isfinite
from threading import BoundedSemaphore, Event, Thread
from time import monotonic, perf_counter

from django.db import connections

_SLOT = BoundedSemaphore(1)


@dataclass(frozen=True, slots=True)
class MetadataSessionProfile:
    isolation: str
    default_isolation: str
    user: str
    session_user: str
    role: str
    readonly: bool
    default_readonly: bool
    row_security: bool
    search_path: str
    schemas: tuple
    temp_schema: int
    database: str
    database_oid: int
    address: object
    port: object
    statement_ms: int
    lock_ms: int
    pid: int


@dataclass(frozen=True, slots=True)
class LexicalMetadataSpec:
    alias: str
    representation: object
    original_ids: frozenset
    requested_ids: frozenset
    query: str
    trigram_limit: int
    exact_limit: int
    similarity: float


@dataclass(frozen=True, slots=True)
class LexicalMetadataResult:
    trigram: tuple
    exact: tuple
    terms: tuple
    trigram_ms: float
    exact_ms: float


@dataclass(frozen=True, slots=True)
class PreDataIneligible:
    pass


_PROFILE_SQL = """SELECT current_setting('transaction_isolation'),
 current_setting('default_transaction_isolation'), current_user::text,
 session_user::text, current_setting('role'),
 current_setting('transaction_read_only')::boolean,
 current_setting('default_transaction_read_only')::boolean,
 current_setting('row_security')::boolean, current_setting('search_path'),
 current_schemas(true)::text[], pg_my_temp_schema(), current_database(),
 (SELECT oid FROM pg_database WHERE datname=current_database()),
 inet_server_addr()::text, inet_server_port(),
 (SELECT setting::bigint FROM pg_settings WHERE name='statement_timeout' AND unit='ms'),
 (SELECT setting::bigint FROM pg_settings WHERE name='lock_timeout' AND unit='ms'),
 pg_backend_pid()"""


def _failure():
    from .representation_contracts import SourceScopeUnavailable
    return SourceScopeUnavailable("source_metadata_worker_unavailable")


def _profile(connection):
    with connection.cursor() as cursor:
        cursor.execute(_PROFILE_SQL)
        row = list(cursor.fetchone())
    row[9] = tuple(row[9])
    result = MetadataSessionProfile(*row)
    if any(type(value) is not int or value < 0
           for value in (result.statement_ms, result.lock_ms)):
        raise _failure()
    return result


def _supported(profile):
    return (profile.isolation == profile.default_isolation == "read committed"
            and profile.temp_schema == 0)


def _policy(profile):
    return tuple(getattr(profile, field) for field in profile.__dataclass_fields__
                 if field not in {"statement_ms", "lock_ms", "pid"})


def _equivalent(parent, worker):
    return (_supported(parent) and _supported(worker)
            and _policy(parent) == _policy(worker) and parent.pid != worker.pid)


def _timeout_cap(remaining, *caps):
    if type(remaining) is not int or remaining <= 0 or not isfinite(remaining):
        raise ValueError("expired metadata allowance")
    if any(type(cap) is not int or cap < 0 for cap in caps):
        raise ValueError("invalid metadata timeout")
    return min([remaining, *(cap for cap in caps if cap)])


def capture_parent_profile(connection, *, check):
    check()
    if connection.execute_wrappers:
        return None
    profile = _profile(connection)
    check()
    return profile if _supported(profile) and not connection.execute_wrappers else None


def accept_parent_profile(connection, profile, *, check):
    check()
    if connection.execute_wrappers:
        raise _failure()
    current = _profile(connection)
    if (connection.execute_wrappers or current.pid != profile.pid
            or _policy(current) != _policy(profile)
            or any(old and (not new or new > old) for old, new in (
                (profile.statement_ms, current.statement_ms),
                (profile.lock_ms, current.lock_ms)))):
        raise _failure()
    check()


class LexicalMetadataJob:
    def __init__(self, spec, profile, runtime, budget, check):
        self.spec, self.profile = spec, profile
        self.runtime, self.budget, self.check = runtime, budget, check
        self.invalid = Event()
        self.data_started = False
        self._future = Future()
        self._dispatch, self._dispatched = Event(), False
        before = monotonic()
        self.deadline = before + self._original_remaining() / 1000

    def _original_remaining(self):
        from .source_deadline import source_remaining_ms
        self.check()
        if self.runtime.budget is not self.budget:
            raise _failure()
        self.budget.check_active()
        remaining = source_remaining_ms(self.budget.remaining_ms())
        if type(remaining) is not int or remaining <= 0:
            raise _failure()
        return remaining

    def remaining(self):
        if self.invalid.is_set():
            raise _failure()
        remaining = min(self._original_remaining(),
                        int((self.deadline - monotonic()) * 1000))
        if remaining <= 0:
            raise _failure()
        return remaining

    def invalidate(self):
        self.invalid.set()

    def result(self):
        try:
            result = (self._future.result() if self._future.done() else
                      self._future.result(timeout=self.remaining() / 1000))
            self.remaining()
            return result
        except BaseException:
            self.invalidate()
            raise

    def _gate(self, connection, retained):
        remaining = self.remaining()
        if connection.execute_wrappers:
            return None
        current = _profile(connection)
        if current.pid != retained.pid:
            return None
        statement = _timeout_cap(remaining, self.profile.statement_ms,
                                 current.statement_ms, retained.statement_ms)
        lock = _timeout_cap(remaining, self.profile.lock_ms,
                            current.lock_ms, retained.lock_ms)
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT set_config('statement_timeout', LEAST(%s, COALESCE(NULLIF("
                "(SELECT setting::bigint FROM pg_settings WHERE name='statement_timeout' "
                "AND unit='ms'), 0), %s))::text || 'ms', true), "
                "set_config('lock_timeout', LEAST(%s, COALESCE(NULLIF("
                "(SELECT setting::bigint FROM pg_settings WHERE name='lock_timeout' "
                "AND unit='ms'), 0), %s))::text || 'ms', true)",
                [statement, statement, lock, lock])
        observed = _profile(connection)
        self.remaining()
        if (connection.execute_wrappers or not _equivalent(self.profile, observed)
                or observed.pid != retained.pid
                or not observed.statement_ms or observed.statement_ms > statement
                or not observed.lock_ms or observed.lock_ms > lock):
            return None
        return observed

    def _read(self):
        self.remaining()
        from .source_loading import bounded_source_database
        from .graph_candidate_metadata import _lexical_text, _trigram_refs, _exact_refs
        connection = connections[self.spec.alias]
        with bounded_source_database(self.runtime, self.spec.alias):
            self.remaining()
            if connection.execute_wrappers:
                return PreDataIneligible()
            retained = _profile(connection)
            retained = self._gate(connection, retained)
            if retained is None:
                return PreDataIneligible()
            text = _lexical_text(self.spec)
            started = perf_counter()
            self.remaining()
            self.data_started = True
            trigram = _trigram_refs(text, self.spec.query, self.spec.trigram_limit,
                                    self.spec.similarity)
            trigram_ms = (perf_counter() - started) * 1000
            retained = self._gate(connection, retained)
            if retained is None:
                raise _failure()
            started = perf_counter()
            exact, terms = _exact_refs(text, self.spec.query, self.spec.exact_limit)
            exact_ms = (perf_counter() - started) * 1000
            if self._gate(connection, retained) is None:
                raise _failure()
            return LexicalMetadataResult(trigram, exact, terms, trigram_ms, exact_ms)

    def run(self):
        self._dispatch.wait()
        if not self._dispatched:
            return  # Startup failed before permission to open a worker connection.
        result, failure = None, None
        try:
            result = self._read()
        except BaseException as exc:
            failure = exc
        finally:
            try:
                connections.close_all()
                self.remaining()
            except BaseException as exc:
                failure = exc
            finally:
                _SLOT.release()
        if failure is not None:
            self.invalidate()
            self._future.set_exception(failure)
        else:
            self._future.set_result(result)


def start_lexical_metadata(spec, *, parent_profile, runtime, budget, check):
    check()
    if not _SLOT.acquire(blocking=False):
        return None
    job = None
    try:
        job = LexicalMetadataJob(spec, parent_profile, runtime, budget, check)
        context = copy_context()
        job.remaining()
        Thread(target=context.run, args=(job.run,), daemon=True,
               name="source-lexical-metadata").start()
        job._dispatched = True
        return job
    except BaseException as exc:
        if job is not None:
            job.invalidate()
            if not job._dispatched:
                job._future.set_exception(exc)
        if job is None or not job._dispatched:
            _SLOT.release()
        raise
    finally:
        if job is not None:
            job._dispatch.set()
