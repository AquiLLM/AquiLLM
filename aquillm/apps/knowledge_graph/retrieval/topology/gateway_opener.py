"""Fresh gateway openers initialize only the configured transport's handler."""

from urllib.parse import urlsplit
from urllib.request import (
    HTTPDefaultErrorHandler,
    HTTPErrorProcessor,
    HTTPHandler,
    HTTPSHandler,
    OpenerDirector,
    UnknownHandler,
)


def build_opener(endpoint, *handlers):
    """Avoid loading an unused TLS trust store for every internal HTTP read."""
    scheme = urlsplit(endpoint).scheme
    if scheme not in ("http", "https"):
        raise ValueError("gateway transport scheme is invalid")
    transport = HTTPHandler if scheme == "http" else HTTPSHandler
    opener = OpenerDirector()
    for handler in (
        UnknownHandler(),
        HTTPDefaultErrorHandler(),
        HTTPErrorProcessor(),
        transport(),
        *handlers,
    ):
        opener.add_handler(handler)
    return opener
