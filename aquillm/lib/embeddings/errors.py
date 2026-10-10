"""Classify transport failures without treating application bugs as outages."""
import httpx
import openai
import requests
from cohere.core.api_error import ApiError

from .utils import EmbeddingContractError

REQUEST_TIMEOUT_SECONDS = 30


class EmbeddingUpstreamUnavailableError(RuntimeError):
    """A retryable embedding connection, timeout, rate-limit, or server failure."""


def require_transient(exc: Exception) -> None:
    """Return only for retryable failures; raise all other failures immediately."""
    if isinstance(exc, (EmbeddingUpstreamUnavailableError, ConnectionError, TimeoutError,
                        httpx.TransportError, openai.APIConnectionError,
                        requests.ConnectionError, requests.Timeout)):
        return
    if isinstance(exc, (openai.APIStatusError, ApiError, requests.HTTPError)):
        status = getattr(exc, "status_code", None)
        if status is None and getattr(exc, "response", None) is not None:
            status = exc.response.status_code
        if status in (408, 429) or (isinstance(status, int) and 500 <= status <= 599):
            return
        raise EmbeddingContractError("Embedding provider rejected the request") from None
    raise exc
