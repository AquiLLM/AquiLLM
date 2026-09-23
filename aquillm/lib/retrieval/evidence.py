"""Exact source offsets and private provenance shared by retrieval consumers.

Fingerprints may be opaque storage revisions. ``fingerprint_source`` is available
when a producer needs a deterministic content revision instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from json import dumps
from typing import Literal

Coverage = Literal["complete", "partial", "unknown"]
_COVERAGE = frozenset(("complete", "partial", "unknown"))


def fingerprint_source(text: str) -> str:
    """Hash exact UTF-8 source content; no Unicode normalization or clipping."""
    if not isinstance(text, str):
        raise TypeError("source text must be str")
    return sha256(text.encode("utf-8")).hexdigest()


def fingerprint_prepared_evidence(source: SourceEvidence, spans: tuple[SourceSpan, ...]) -> str:
    """Versioned identity for an exact ordered source/span representation."""
    fields = [
        source.chunk_id, source.document_id, source.chunk_number,
        source.source_fingerprint, source.text,
        [[s.chunk_id, s.source_fingerprint, s.start, s.end, s.text] for s in spans],
    ]
    encoded = dumps(fields, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return sha256(b"aquillm.prepared-evidence.v1\x00" + encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    chunk_id: int
    source_fingerprint: str

    def __post_init__(self) -> None:
        if isinstance(self.chunk_id, bool) or not isinstance(self.chunk_id, int):
            raise ValueError("chunk_id must be an integer")
        if not isinstance(self.source_fingerprint, str) or not self.source_fingerprint:
            raise ValueError("source_fingerprint must be nonempty")


@dataclass(frozen=True, slots=True)
class SourceEvidence:
    chunk_id: int
    document_id: str
    chunk_number: int
    source_fingerprint: str
    text: str

    def __post_init__(self) -> None:
        SourceIdentity(self.chunk_id, self.source_fingerprint)
        if not isinstance(self.document_id, str) or not self.document_id:
            raise ValueError("document_id must be nonempty")
        if isinstance(self.chunk_number, bool) or not isinstance(self.chunk_number, int) or self.chunk_number < 0:
            raise ValueError("chunk_number must be a nonnegative integer")
        if not isinstance(self.text, str):
            raise ValueError("source text must be str")

    @property
    def identity(self) -> SourceIdentity:
        return SourceIdentity(self.chunk_id, self.source_fingerprint)


@dataclass(frozen=True, slots=True)
class SourceSpan:
    chunk_id: int
    source_fingerprint: str
    start: int
    end: int
    text: str

    def __post_init__(self) -> None:
        SourceIdentity(self.chunk_id, self.source_fingerprint)
        if (
            isinstance(self.start, bool) or not isinstance(self.start, int)
            or isinstance(self.end, bool) or not isinstance(self.end, int)
            or self.start < 0 or self.end <= self.start
        ):
            raise ValueError("span offsets must be an increasing code-point range")
        if not isinstance(self.text, str):
            raise ValueError("span text must be str")


@dataclass(frozen=True, slots=True)
class PreparedEvidence:
    source: SourceEvidence
    spans: tuple[SourceSpan, ...]
    evidence_fingerprint: str
    estimated_tokens: int
    source_coverage: Coverage

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceEvidence):
            raise ValueError("source must be SourceEvidence")
        if not isinstance(self.spans, tuple):
            raise ValueError("spans must be a tuple")
        if not isinstance(self.evidence_fingerprint, str) or not self.evidence_fingerprint:
            raise ValueError("evidence_fingerprint must be nonempty")
        if isinstance(self.estimated_tokens, bool) or not isinstance(self.estimated_tokens, int) or self.estimated_tokens < 0:
            raise ValueError("estimated_tokens must be nonnegative")
        if not isinstance(self.source_coverage, str) or self.source_coverage not in _COVERAGE:
            raise ValueError("invalid source_coverage")
        last_end = 0
        for span in self.spans:
            if not isinstance(span, SourceSpan):
                raise ValueError("spans must contain SourceSpan")
            if span.chunk_id != self.source.chunk_id or span.source_fingerprint != self.source.source_fingerprint:
                raise ValueError("span identity does not match source")
            if span.start < last_end or span.end > len(self.source.text):
                raise ValueError("span is overlapping or outside source")
            if self.source.text[span.start:span.end] != span.text:
                raise ValueError("span text does not match exact source slice")
            last_end = span.end
        if self.source_coverage == "complete":
            if not self.spans and self.source.text:
                raise ValueError("complete coverage needs source spans")
            next_offset = 0
            for span in self.spans:
                if span.start != next_offset:
                    raise ValueError("complete coverage cannot omit source text")
                next_offset = span.end
            if next_offset != len(self.source.text):
                raise ValueError("complete coverage cannot omit source tail")
        if self.evidence_fingerprint != fingerprint_prepared_evidence(self.source, self.spans):
            raise ValueError("evidence_fingerprint does not match exact prepared evidence")
