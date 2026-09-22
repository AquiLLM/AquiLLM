from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from threading import Event, Thread
from uuid import UUID

from django.db import (
    InterfaceError,
    OperationalError,
    close_old_connections,
    connections,
)
from django.utils import timezone

from .memgraph_driver import MemgraphDriverError
from .memgraph_repository import MemgraphProjectionRepository
from .records import (
    ProjectionFailureCode,
    ProjectionGenerationManifestV1,
    ProjectionLifecycleState,
)
from .runtime import (
    ProjectionDatabaseAliases,
    load_projection_runtime_settings,
    memgraph_projection_repository,
    postgres_projection_repository,
)
from .serialization import projection_checksum
from .state_repository import FunctionProjectionStateRepository

_STATE_REPOSITORY: ContextVar[FunctionProjectionStateRepository | None] = ContextVar(
    "projection_state_repository", default=None
)


@dataclass(frozen=True, slots=True)
class ProjectionRunOutcomeV1:
    projection_id: UUID
    ready: bool
    failure_code: str | None


class _ProjectionLeaseLost(RuntimeError):
    pass


class _ProjectionLeaseHeartbeat:
    """Renew the exact state-function lease throughout slow external work."""

    def __init__(self, repository, lease, *, lease_seconds: int):
        self.repository = repository
        self.lease = lease
        self.lease_seconds = lease_seconds
        self.interval = lease_seconds / 4.0
        self._stop = Event()
        self._thread = None
        self._failure = None

    def pulse(self):
        previous = self.lease
        try:
            lease = self.repository.renew(
                projection_id=previous.projection_id,
                owner=previous.owner,
                now=timezone.now(),
                lease_seconds=self.lease_seconds,
            )
        except Exception as exc:
            if _backend_transient(exc):
                raise TimeoutError("projection_backend_transient") from None
            raise _ProjectionLeaseLost(
                "projection lease renewal was rejected"
            ) from None
        if (
            lease.projection_id != previous.projection_id
            or lease.owner != previous.owner
            or lease.attempt_count != previous.attempt_count
            or lease.expires_at <= timezone.now()
        ):
            raise _ProjectionLeaseLost("projection lease renewal was rejected")
        self.lease = lease

    def check(self):
        if self._failure is not None:
            raise self._failure
        if self.lease.expires_at <= timezone.now():
            raise _ProjectionLeaseLost("projection lease expired")

    def _run(self):
        close_old_connections()
        try:
            while not self._stop.wait(self.interval):
                try:
                    self.pulse()
                except BaseException as exc:
                    self._failure = exc
                    self._stop.set()
                    return
        finally:
            # Django connections belong to this thread; never close them from
            # the worker thread or leave a persistent heartbeat connection open.
            connections.close_all()

    def __enter__(self):
        self.pulse()
        self._thread = Thread(
            target=self._run,
            name=f"projection-lease-{self.lease.projection_id}",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        self._thread.join(timeout=min(5.0, self.interval))
        if self._thread.is_alive():
            # An unavailable state backend must not hang publication indefinitely.
            # The daemon still closes its own connection when the call returns.
            raise _ProjectionLeaseLost("projection lease renewal did not finish")
        if exc_type is None or self._failure is not None:
            self.check()
        return False


def _identifier(value: object) -> UUID:
    if type(value) is not UUID:
        raise TypeError("projection_id must be an exact UUID")
    return value


def _postgres_repository():
    state_repository = _STATE_REPOSITORY.get()
    if state_repository is None:
        return postgres_projection_repository(_projection_settings())
    return postgres_projection_repository(
        _projection_settings(), state_repository=state_repository
    )


def _state_using() -> str:
    return ProjectionDatabaseAliases().state


def _projection_settings():
    return load_projection_runtime_settings()


def _memgraph_repository():
    return memgraph_projection_repository(_projection_settings())


def _function_state_repository() -> FunctionProjectionStateRepository:
    repository = _STATE_REPOSITORY.get()
    if repository is None:
        raise RuntimeError("projection state repository is not bound")
    return repository


def claim_projection_lease(**kwargs):
    kwargs.pop("using", None)
    return _function_state_repository().claim(**kwargs)


def record_projection_private_mapping_checksum(**kwargs):
    kwargs.pop("using", None)
    return _function_state_repository().record_private_mapping(**kwargs)


def mark_projection_failed(**kwargs):
    kwargs.pop("using", None)
    return _function_state_repository().fail(**kwargs)


def publish_projection_ready_compare_and_set(**kwargs):
    kwargs.pop("using", None)
    settings = _projection_settings()
    kwargs["versions"] = (
        settings.projection_schema_version,
        settings.projection_format_version,
        settings.projection_identifier_key_version,
    )
    return _function_state_repository().ready(**kwargs)


def _backend_transient(exc: BaseException) -> bool:
    return isinstance(
        exc, (ConnectionError, TimeoutError, OperationalError, InterfaceError)
    ) or (
        isinstance(exc, MemgraphDriverError)
        and exc.code in {"memgraph_read_failed", "memgraph_write_failed"}
    )


def _expected_manifest(bundle, private_mapping_checksum: str):
    checksum = projection_checksum(bundle)
    return ProjectionGenerationManifestV1(
        bundle.generation.generation_key,
        bundle.generation.schema_version,
        bundle.generation.projection_version,
        bundle.generation.identifier_key_version,
        checksum,
        checksum,
        private_mapping_checksum,
        bundle.counts,
        ProjectionLifecycleState.BUILDING,
    )


def project_generation(
    *, projection_id: UUID, lease_owner: str
) -> ProjectionRunOutcomeV1:
    identifier = _identifier(projection_id)
    settings = _projection_settings()
    using = _state_using()
    state_repository = FunctionProjectionStateRepository(
        state_using=using,
        source_using=ProjectionDatabaseAliases().source,
        owner=lease_owner,
    )
    token = _STATE_REPOSITORY.set(state_repository)
    graph_ready_transition_started = False
    try:
        lease = claim_projection_lease(
            projection_id=identifier,
            owner=lease_owner,
            now=timezone.now(),
            lease_seconds=settings.projection_lease_seconds,
            using=using,
        )
        if lease is None:
            return ProjectionRunOutcomeV1(identifier, False, "lease_lost")
        with _ProjectionLeaseHeartbeat(
            state_repository, lease, lease_seconds=settings.projection_lease_seconds
        ) as heartbeat:
            postgres = _postgres_repository()
            bundle = postgres.load_projection_bundle(
                projection_id=identifier,
                batch_size=settings.projection_batch_size,
                purpose="build",
            )
            heartbeat.check()
            private_rows = postgres.load_private_chunk_references(
                projection_id=identifier,
                batch_size=settings.projection_batch_size,
            )
            heartbeat.check()
            private_checksum = postgres.persist_chunk_references(
                projection_id=identifier,
                rows=private_rows,
                batch_size=settings.projection_batch_size,
            )
            heartbeat.check()
            record_projection_private_mapping_checksum(
                projection_id=identifier,
                owner=lease_owner,
                checksum=private_checksum,
                now=timezone.now(),
                using=using,
            )
            graph = _memgraph_repository()
            timeout = settings.graph_overall_timeout_ms / 1_000.0
            graph.write_staging_generation(
                bundle=bundle,
                private_mapping_checksum=private_checksum,
                batch_size=settings.projection_batch_size,
                timeout_seconds=timeout,
            )
            heartbeat.check()
            expected = _expected_manifest(bundle, private_checksum)
            validation = graph.validate_generation(
                expected=expected,
                timeout_seconds=timeout,
            )
            heartbeat.check()
            if not validation.valid:
                raise ValueError("projection_validation_failed")
            graph_ready_transition_started = True
            graph.mark_generation_ready(
                generation_key=MemgraphProjectionRepository.opaque_generation_key(
                    bundle.generation.generation_key
                ),
                validation_checksum=validation.validation_checksum,
                timeout_seconds=timeout,
            )
        # The heartbeat is joined before a fresh renewal and the authoritative
        # CAS. A successful CAS clears the lease and must never race another pulse.
        heartbeat.pulse()
        outcome = publish_projection_ready_compare_and_set(
            projection_id=identifier,
            owner=lease_owner,
            validation=validation,
            expected_generation_key=bundle.generation.generation_key,
            expected_graph_checksum=expected.graph_checksum,
            expected_private_mapping_checksum=private_checksum,
            now=timezone.now(),
            using=using,
        )
        return ProjectionRunOutcomeV1(
            identifier,
            outcome.published,
            outcome.failure_code,
        )
    except Exception as exc:
        if isinstance(exc, _ProjectionLeaseLost):
            return ProjectionRunOutcomeV1(identifier, False, "lease_lost")
        if not graph_ready_transition_started and _backend_transient(exc):
            raise TimeoutError("projection_backend_transient") from None
        code = (
            ProjectionFailureCode.VALIDATION_FAILED
            if str(exc) == "projection_validation_failed"
            else ProjectionFailureCode.WRITE_FAILED
        )
        try:
            mark_projection_failed(
                projection_id=identifier,
                owner=lease_owner,
                failure_code=code,
                now=timezone.now(),
                using=using,
            )
        except Exception as failure_exc:
            if graph_ready_transition_started:
                # Memgraph may already be ready, so retrying staging on the same
                # generation is unsafe. Reconciliation replaces failed/expired
                # occurrences; the state function never overwrites another owner
                # or an already-successful ready CAS.
                return ProjectionRunOutcomeV1(identifier, False, "lease_lost")
            if _backend_transient(failure_exc):
                raise TimeoutError("projection_backend_transient") from None
        return ProjectionRunOutcomeV1(identifier, False, code.value)
    finally:
        _STATE_REPOSITORY.reset(token)


__all__ = ["ProjectionRunOutcomeV1", "project_generation"]
