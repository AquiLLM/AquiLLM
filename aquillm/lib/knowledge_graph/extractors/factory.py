"""Lazy extraction-provider selection."""

from __future__ import annotations

import importlib
from typing import cast

from ..config import ExtractionSettings, load_extraction_settings
from .base import ExtractionBackend

_PROVIDERS = {
    "gliner2_local": (
        "lib.knowledge_graph.extractors.gliner2_local",
        "GLiNER2LocalBackend",
    ),
}


class UnsupportedExtractionProviderError(ValueError):
    """Raised before import when a provider is not configured."""


def get_extraction_backend(
    *, settings: ExtractionSettings | None = None
) -> ExtractionBackend:
    """Construct the configured provider, importing it only for this call."""

    resolved_settings = settings or load_extraction_settings()
    try:
        module_name, backend_name = _PROVIDERS[resolved_settings.provider]
    except KeyError as exc:
        raise UnsupportedExtractionProviderError(
            f"Unsupported knowledge-graph extraction provider: "
            f"{resolved_settings.provider!r}"
        ) from exc

    provider_module = importlib.import_module(module_name)
    backend_type = getattr(provider_module, backend_name)
    return cast(ExtractionBackend, backend_type(settings=resolved_settings))
