"""Bounded retry classification for graph build tasks."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from django.db import InterfaceError as DjangoInterfaceError
from django.db import OperationalError as DjangoOperationalError
from kombu.exceptions import OperationalError as KombuOperationalError

_TRANSIENT_RETRY_BASE_SECONDS = 30
_TRANSIENT_RETRY_MAX_SECONDS = 60
_MAX_EXCEPTION_CHAIN_DEPTH = 16
_OPENAI_TRANSIENT_ERRORS = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
    }
)
_BOTOCORE_TRANSIENT_ERRORS = frozenset(
    {
        "ConnectionClosedError",
        "ConnectTimeoutError",
        "EndpointConnectionError",
        "HTTPClientError",
        "ReadTimeoutError",
    }
)
_BOTOCORE_TRANSIENT_ERROR_CODES = frozenset(
    {
        "internalerror",
        "requestlimitexceeded",
        "requesttimeout",
        "requesttimeoutexception",
        "serviceunavailable",
        "slowdown",
    }
)
def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    for _ in range(_MAX_EXCEPTION_CHAIN_DEPTH):
        if current is None or id(current) in seen:
            return
        seen.add(id(current))
        yield current
        if current.__cause__ is not None:
            current = current.__cause__
        elif current.__suppress_context__:
            return
        else:
            current = current.__context__


def _response_status_code(exc: BaseException) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if type(status_code) is int:
        return status_code
    response = getattr(exc, "response", None)
    if type(response) is not dict:
        return None
    metadata = response.get("ResponseMetadata")
    if type(metadata) is not dict:
        return None
    status_code = metadata.get("HTTPStatusCode")
    return status_code if type(status_code) is int else None


def _is_known_provider_transient(exc: BaseException) -> bool:
    exception_type = type(exc)
    if (
        exception_type.__module__ == "lib.knowledge_graph.extractors.gliner2_local"
        and exception_type.__name__ == "ExtractionBackendError"
    ):
        return True
    module_root = exception_type.__module__.partition(".")[0]
    name = exception_type.__name__
    status_code = _response_status_code(exc)
    if module_root == "openai":
        return name in _OPENAI_TRANSIENT_ERRORS or (
            name == "APIStatusError"
            and status_code is not None
            and (status_code == 429 or status_code >= 500)
        )
    if module_root != "botocore":
        return False
    if name in _BOTOCORE_TRANSIENT_ERRORS:
        return True
    if name != "ClientError":
        return False
    response = getattr(exc, "response", None)
    error = response.get("Error") if type(response) is dict else None
    error_code = error.get("Code") if type(error) is dict else None
    return (status_code is not None and (status_code == 429 or status_code >= 500)) or (
        type(error_code) is str
        and (
            error_code.lower().startswith("throttl")
            or error_code.lower() in _BOTOCORE_TRANSIENT_ERROR_CODES
        )
    )


def _retry_countdown(
    exc: BaseException,
    *,
    builds: Any | None,
    retry_count: int,
) -> int | None:
    if builds is not None and isinstance(
        exc,
        (builds.StaleBuildError, builds.CorruptBuildError),
    ):
        return None
    if builds is not None and isinstance(
        exc,
        builds.BuildInProgressError,
    ):
        return builds.BUILD_LEASE_RETRY_SECONDS
    if builds is not None and isinstance(exc, builds.BuildLeaseLostError):
        return min(
            _TRANSIENT_RETRY_BASE_SECONDS * (2**retry_count),
            _TRANSIENT_RETRY_MAX_SECONDS,
        )
    for candidate in _exception_chain(exc):
        if isinstance(
            candidate,
            (
                ConnectionError,
                TimeoutError,
                DjangoInterfaceError,
                DjangoOperationalError,
                KombuOperationalError,
            ),
        ) or _is_known_provider_transient(candidate):
            return min(
                _TRANSIENT_RETRY_BASE_SECONDS * (2**retry_count),
                _TRANSIENT_RETRY_MAX_SECONDS,
            )
    return None
