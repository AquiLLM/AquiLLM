"""Explicit ownership for projection-job graph resources."""
from dataclasses import dataclass
import logging

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OwnedProjectionRepository:
    repository: object
    close: object


def release_owned_repository(handle):
    """Cleanup cannot replace a completed job outcome or its original error."""
    try:
        handle.close()
    except Exception:
        try:
            _LOGGER.warning("obs.kg.projection_driver_close_failed")
        except Exception:
            pass


def close_owned_client(self):
    """Close only a lazily allocated client; constructor injection is borrowed."""
    with self._client_lock:
        if self._closed:
            return
        self._closed = True
        client = self._client if self._owns_client else None
        self._client = None
    if client is not None:
        client.close()
