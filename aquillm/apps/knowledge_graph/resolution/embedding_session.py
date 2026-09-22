"""Signature-bound collection embedding session and result values."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256

from .scoring import validate_embedding

EMBEDDING_PREPROCESSING_VERSION = "kg-entity-v1"
MAX_TEXT_CHARACTERS = 8_192
DEFAULT_EMBEDDING_BATCH_SIZE = 64
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
def _bounded_text(
    value: object,
    label: str,
    *,
    maximum: int = MAX_TEXT_CHARACTERS,
    allow_empty: bool = False,
) -> str:
    if type(value) is not str:
        raise ValueError(f"{label} must be an exact string")
    normalized = " ".join(unicodedata.normalize("NFC", value).split())
    if not normalized and not allow_empty:
        raise ValueError(f"{label} must be nonempty")
    if len(normalized) > maximum or "\x00" in normalized:
        raise ValueError(f"{label} is unsafe or exceeds {maximum} characters")
    return normalized
def _require_hash(value: object, label: str) -> str:
    if type(value) is not str or not _HASH_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value
def embedding_text_hash(text: object) -> str:
    """Hash the exact normalized input covered by the preprocessing signature."""

    normalized = _bounded_text(text, "embedding text")
    return sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SignedEmbeddingBatch:
    """Backend response binding vector order to the actual provider/model."""

    vectors: tuple[tuple[float, ...], ...]
    text_hashes: tuple[str, ...]
    indices: tuple[int, ...]
    model_signature: str

    def __post_init__(self) -> None:
        if type(self.vectors) is not tuple or any(
            type(vector) is not tuple for vector in self.vectors
        ):
            raise ValueError("embedding vectors must be an exact tuple of tuples")
        if type(self.text_hashes) is not tuple:
            raise ValueError("embedding text hashes must be an exact tuple")
        if type(self.indices) is not tuple or any(
            type(index) is not int or index < 0 for index in self.indices
        ):
            raise ValueError(
                "embedding provider indices must be exact nonnegative ints"
            )
        if not (len(self.vectors) == len(self.text_hashes) == len(self.indices)):
            raise ValueError("embedding batch audit fields must have equal counts")
        if len(self.indices) != len(set(self.indices)):
            raise ValueError("embedding provider indices must be unique")
        object.__setattr__(
            self,
            "vectors",
            tuple(validate_embedding(vector) for vector in self.vectors),
        )
        for value in self.text_hashes:
            _require_hash(value, "embedding text hash")
        object.__setattr__(
            self,
            "model_signature",
            _bounded_text(
                self.model_signature, "embedding model signature", maximum=512
            ),
        )


@dataclass(frozen=True, slots=True)
class EmbeddedText:
    text: str
    input_hash: str
    vector: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "text",
            _bounded_text(self.text, "embedded text", maximum=MAX_TEXT_CHARACTERS),
        )
        _require_hash(self.input_hash, "embedded input hash")
        if self.input_hash != embedding_text_hash(self.text):
            raise ValueError("embedded input hash does not bind exact text")
        object.__setattr__(self, "vector", validate_embedding(self.vector))


EmbeddingBackend = Callable[[tuple[str, ...]], SignedEmbeddingBatch]


class CollectionEmbeddingSession:
    """One build-scoped, signature-locked embedding session with stable caching."""

    __slots__ = (
        "expected_model_signature",
        "batch_size",
        "preprocessing_version",
        "max_text_characters",
        "_backend",
        "_cache",
        "_successful_batch_count",
    )

    def __init__(
        self,
        *,
        expected_model_signature: str,
        backend: EmbeddingBackend,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        preprocessing_version: str = EMBEDDING_PREPROCESSING_VERSION,
        max_text_characters: int = MAX_TEXT_CHARACTERS,
    ) -> None:
        signature = _bounded_text(
            expected_model_signature, "embedding model signature", maximum=512
        )
        if type(batch_size) is not int or not 1 <= batch_size <= 1_000:
            raise ValueError("embedding batch size must be a bounded positive integer")
        if (
            type(max_text_characters) is not int
            or not 1 <= max_text_characters <= 1_000_000
        ):
            raise ValueError(
                "embedding max characters must be a bounded positive integer"
            )
        preprocessing = _bounded_text(
            preprocessing_version, "embedding preprocessing version", maximum=128
        )
        required_tokens = (
            "dims=1024",
            f"prep={preprocessing}",
            f"max_chars={max_text_characters}",
            f"batch={batch_size}",
        )
        if any(token not in signature.split(":") for token in required_tokens):
            raise ValueError(
                "embedding model signature must lock dimensions, preprocessing, "
                "maximum characters, and batch size"
            )
        endpoint_tokens = tuple(
            token.removeprefix("endpoint=")
            for token in signature.split(":")
            if token.startswith("endpoint=")
        )
        if len(endpoint_tokens) != 1 or not _HASH_PATTERN.fullmatch(endpoint_tokens[0]):
            raise ValueError(
                "embedding model signature must bind one provider endpoint digest"
            )
        if not callable(backend):
            raise ValueError("embedding backend must be callable")
        self.expected_model_signature = signature
        self.batch_size = batch_size
        self.preprocessing_version = preprocessing
        self.max_text_characters = max_text_characters
        self._backend = backend
        self._cache: dict[str, tuple[str, tuple[float, ...]]] = {}
        self._successful_batch_count = 0

    @property
    def successful_batch_count(self) -> int:
        return self._successful_batch_count

    @property
    def cached_text_count(self) -> int:
        return len(self._cache)

    def embed(self, texts: Sequence[str]) -> tuple[EmbeddedText, ...]:
        if not isinstance(texts, (tuple, list)):
            raise ValueError("embedding inputs must be a concrete sequence")
        normalized = tuple(
            _bounded_text(text, "embedding text", maximum=self.max_text_characters)
            for text in texts
        )
        missing = tuple(sorted(set(normalized).difference(self._cache)))
        if missing:
            staged: dict[str, tuple[str, tuple[float, ...]]] = {}
            successful_batches = 0
            for start in range(0, len(missing), self.batch_size):
                inputs = missing[start : start + self.batch_size]
                batch = self._backend(inputs)
                if type(batch) is not SignedEmbeddingBatch:
                    raise ValueError(
                        "embedding backend must return SignedEmbeddingBatch"
                    )
                batch.__post_init__()
                if batch.model_signature != self.expected_model_signature:
                    raise ValueError(
                        "embedding provider/model signature drift detected"
                    )
                if len(batch.vectors) != len(inputs):
                    raise ValueError(
                        "embedding backend must return one vector per input"
                    )
                if set(batch.indices) != set(range(len(inputs))):
                    raise ValueError(
                        "embedding provider indices do not prove exact input binding"
                    )
                for index, input_hash, vector in zip(
                    batch.indices,
                    batch.text_hashes,
                    batch.vectors,
                    strict=True,
                ):
                    text = inputs[index]
                    if input_hash != embedding_text_hash(text):
                        raise ValueError(
                            "embedding backend output index/hash does not match input"
                        )
                    staged[text] = (input_hash, vector)
                successful_batches += 1
            if set(staged) != set(missing):
                raise ValueError("embedding response did not bind every exact input")
            self._cache.update(staged)
            self._successful_batch_count += successful_batches
        return tuple(
            EmbeddedText(
                text=text,
                input_hash=self._cache[text][0],
                vector=self._cache[text][1],
            )
            for text in normalized
        )
