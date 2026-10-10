"""Closed, reproducible inputs for weighted lexical projections."""

from dataclasses import dataclass
from hashlib import sha256
from uuid import UUID

from django.db.models import F, JSONField, Value
from django.db.models.functions import Cast, JSONObject

from lib.embeddings.document_contracts import require_digest

from .representation_contracts import DocumentRefV1
from .representation_store import SourceSHA256, document_payload

POLICY = "fts-english-titleA-sectionB-bodyD-v1"


def text_hash(text):
    return sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class LexicalProjectionV1:
    document: DocumentRefV1
    canonical_sha256: str
    representation_key: UUID | None
    source_revision_sha256: str | None
    title: str

    def __post_init__(self):
        document_payload(self.document)
        require_digest(self.canonical_sha256)
        if (
            self.representation_key is not None
            and type(self.representation_key) is not UUID
        ):
            raise ValueError("invalid_lexical_representation")
        if self.source_revision_sha256 is not None:
            require_digest(self.source_revision_sha256)
        if (self.representation_key is None) != (self.source_revision_sha256 is None):
            raise ValueError("invalid_lexical_revision")
        if type(self.title) is not str or len(self.title) > 200:
            raise ValueError("invalid_lexical_title")

    def values(self):
        return {
            "version": "chunk-search-projection-v1",
            "policy": POLICY,
            "document": document_payload(self.document),
            "canonical_sha256": self.canonical_sha256,
            "representation_key": str(self.representation_key)
            if self.representation_key
            else None,
            "source_revision_sha256": self.source_revision_sha256,
            "title_sha256": text_hash(self.title),
        }


def binding_expression(projection):
    from django.db.models import TextField

    # JSON equality checks the complete closed payload, including extra keys.
    constants = {
        name: Value(value, output_field=JSONField())
        if isinstance(value, dict) or value is None
        else Value(value)
        for name, value in projection.values().items()
    }
    return JSONObject(
        **constants,
        body_sha256=SourceSHA256(F("content")),
        start=F("start_position"),
        end=F("end_position"),
        chunk_number=F("chunk_number"),
        section_path=F("section_path"),
        source_kind=F("source_kind"),
        vector_sha256=SourceSHA256(Cast("search_vector", TextField())),
    )


def structural_projection(chunk, canonical_text, extraction):
    if extraction is None:
        return "unknown", []
    from lib.parsers.canonical_projection import validate_extraction

    validate_extraction(extraction)
    if extraction.canonical_text != canonical_text:
        raise ValueError("invalid_lexical_extraction")
    covering = [
        e
        for e in extraction.elements
        if e.start <= chunk.start_position and e.end >= chunk.end_position
    ]
    if len(covering) != 1:
        raise ValueError("unbound_lexical_section")
    element = covering[0]
    headings = [
        e
        for e in extraction.elements
        if e.kind == "heading" and e.start <= element.start
    ]
    section = (
        [canonical_text[headings[-1].start : headings[-1].end].rstrip("\n")]
        if headings
        else []
    )
    kind = {
        "paragraph": "body",
        "heading": "heading",
        "table": "table",
        "formula": "formula",
        "figure": "figure",
    }.get(element.kind, "unknown")
    return kind, section
