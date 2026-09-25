"""Provider-neutral graph extraction backend contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from ..types import ExtractionBatchResult


class OntologyDefinition(Protocol):
    """Structural ontology boundary consumed by provider implementations."""

    version: str
    entity_types: Mapping[str, object]
    relations: Mapping[str, object]
    checksum: str


class ExtractionBackend(Protocol):
    """A batch extractor implemented by an optional provider runtime."""

    def extract_batch(
        self,
        texts: tuple[str, ...],
        *,
        ontology: OntologyDefinition,
    ) -> tuple[ExtractionBatchResult, ...]:
        raise NotImplementedError
