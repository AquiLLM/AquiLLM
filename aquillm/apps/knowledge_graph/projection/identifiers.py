from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from hmac import new as hmac_new
from typing import Protocol
from uuid import UUID

_SIGNED_64_BIT_MIN = -(2**63)
_SIGNED_64_BIT_MAX = 2**63 - 1


class ProjectionIdentifierDomain(StrEnum):
    COLLECTION = "collection"
    ARTIFACT = "artifact"
    ENTITY = "entity"
    RELATION = "relation"
    EVIDENCE = "evidence"
    RELATION_MENTION = "relation_mention"
    ENTITY_MENTION = "entity_mention"
    ENTITY_MAPPING = "entity_mapping"
    DOCUMENT = "document"
    CHUNK = "chunk"
    CANONICAL_LINK_DECISION = "canonical_link_decision"
    AUTOMATIC_CANONICAL_IDENTITY = "automatic_canonical_identity"


@dataclass(frozen=True, slots=True)
class OpaqueProjectionKey:
    domain: ProjectionIdentifierDomain
    value: str

    def __post_init__(self) -> None:
        if type(self.domain) is not ProjectionIdentifierDomain:
            raise TypeError("domain must be a ProjectionIdentifierDomain")
        if type(self.value) is not str:
            raise TypeError("value must be a built-in str")
        if len(self.value) != 64 or any(
            character not in "0123456789abcdef" for character in self.value
        ):
            raise ValueError("value must be a lowercase SHA-256 hexadecimal digest")

    def __str__(self) -> str:
        return self.value


class ProjectionIdentifierCodec(Protocol):
    @property
    def codec_version(self) -> str: ...

    @property
    def key_version(self) -> str: ...

    def frame(
        self,
        domain: ProjectionIdentifierDomain,
        *,
        generation: str | UUID | None = None,
        source: str | int | UUID,
    ) -> bytes: ...

    def encode(
        self,
        domain: ProjectionIdentifierDomain,
        *,
        generation: str | UUID | None = None,
        source: str | int | UUID,
    ) -> OpaqueProjectionKey: ...


@dataclass(frozen=True, slots=True)
class _ProjectionIdentifierInput:
    codec_version: str
    key_version: str
    domain: ProjectionIdentifierDomain
    generation: str
    source_kind: str
    canonical_source: str


@dataclass(frozen=True, slots=True, init=False)
class HmacSha256ProjectionIdentifierCodec:
    _key: bytes = field(repr=False)
    key_version: str
    codec_version: str

    def __init__(
        self,
        key: bytes,
        *,
        key_version: str,
        codec_version: str = "projection-id-v1",
    ) -> None:
        if type(key) is not bytes:
            raise TypeError("key must be bytes")
        if not key:
            raise ValueError("key must not be empty")
        object.__setattr__(self, "_key", key)
        object.__setattr__(
            self,
            "key_version",
            _canonical_token(key_version, field_name="key_version"),
        )
        object.__setattr__(
            self,
            "codec_version",
            _canonical_token(codec_version, field_name="codec_version"),
        )

    def frame(
        self,
        domain: ProjectionIdentifierDomain,
        *,
        generation: str | UUID | None = None,
        source: str | int | UUID,
    ) -> bytes:
        identifier_input = self._identifier_input(
            domain,
            generation=generation,
            source=source,
        )
        fields = (
            identifier_input.codec_version,
            identifier_input.key_version,
            identifier_input.domain.value,
            identifier_input.generation,
            identifier_input.source_kind,
            identifier_input.canonical_source,
        )
        return b"".join(_frame_field(value) for value in fields)

    def encode(
        self,
        domain: ProjectionIdentifierDomain,
        *,
        generation: str | UUID | None = None,
        source: str | int | UUID,
    ) -> OpaqueProjectionKey:
        framed = self.frame(domain, generation=generation, source=source)
        return OpaqueProjectionKey(
            domain=domain,
            value=hmac_new(self._key, framed, sha256).hexdigest(),
        )

    def _identifier_input(
        self,
        domain: ProjectionIdentifierDomain,
        *,
        generation: str | UUID | None,
        source: str | int | UUID,
    ) -> _ProjectionIdentifierInput:
        if type(domain) is not ProjectionIdentifierDomain:
            raise TypeError("domain must be a ProjectionIdentifierDomain")
        canonical_generation = _canonical_generation(domain, generation)
        source_kind, canonical_source = _canonicalize_source(source)
        return _ProjectionIdentifierInput(
            codec_version=self.codec_version,
            key_version=self.key_version,
            domain=domain,
            generation=canonical_generation,
            source_kind=source_kind,
            canonical_source=canonical_source,
        )


def _canonical_generation(
    domain: ProjectionIdentifierDomain,
    generation: str | UUID | None,
) -> str:
    if generation is not None:
        canonical_generation = _canonical_uuid_or_text(
            generation,
            field_name="generation",
        )
    elif domain is not ProjectionIdentifierDomain.AUTOMATIC_CANONICAL_IDENTITY:
        raise ValueError("generation is required for this domain")

    if domain is ProjectionIdentifierDomain.AUTOMATIC_CANONICAL_IDENTITY:
        return "-"
    if canonical_generation == "-":
        raise ValueError("generation must not use the reserved '-' sentinel")
    return canonical_generation


def _canonicalize_source(source: str | int | UUID) -> tuple[str, str]:
    if type(source) is int:
        if not _SIGNED_64_BIT_MIN <= source <= _SIGNED_64_BIT_MAX:
            raise ValueError("integer source must fit in the signed 64-bit range")
        return "integer", str(source)
    if type(source) is UUID:
        return "uuid", str(source)
    if type(source) is str:
        canonical_source = _canonical_text(source, field_name="source")
        try:
            parsed_uuid = UUID(canonical_source)
        except ValueError:
            return "utf8", canonical_source
        if canonical_source != str(parsed_uuid):
            raise ValueError("source must use the canonical UUID encoding")
        return "uuid", canonical_source
    raise TypeError("source must be an int, str, or UUID")


def _canonical_uuid_or_text(value: str | UUID, *, field_name: str) -> str:
    if type(value) is UUID:
        return str(value)
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a str or UUID")
    canonical_value = _canonical_text(value, field_name=field_name)
    try:
        parsed_uuid = UUID(canonical_value)
    except ValueError:
        return canonical_value
    if canonical_value != str(parsed_uuid):
        raise ValueError(f"{field_name} must use the canonical UUID encoding")
    return canonical_value


def _canonical_token(value: object, *, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a str")
    canonical_value = _canonical_text(value, field_name=field_name)
    if canonical_value != canonical_value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace")
    return canonical_value


def _canonical_text(value: str, *, field_name: str) -> str:
    if not value or value.isspace():
        raise ValueError(f"{field_name} must not be empty")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{field_name} must be valid UTF-8") from error
    return value


def _frame_field(value: str) -> bytes:
    encoded = value.encode("utf-8")
    if len(encoded) > 0xFFFFFFFF:
        raise ValueError("identifier field exceeds the four-byte length limit")
    return len(encoded).to_bytes(4, "big", signed=False) + encoded
