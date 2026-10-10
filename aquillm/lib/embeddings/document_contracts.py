"""Strict document-space declarations; none of these values attest a response."""

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from math import isfinite
import re
from time import monotonic
from typing import Callable, Literal
from uuid import UUID


def digest(payload):
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def require_digest(value):
    if type(value) is not str or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("invalid_digest")
    return value


@dataclass(frozen=True)
class VerifiedEmbeddingSpaceV1:
    space_key: str
    provider: str
    model: str
    verified_revision: str
    tokenizer_revision: str
    dimensions: int
    query_prep_key: str
    document_prep_key: str
    dimension_policy: str
    normalization: str
    compute_precision: Literal["fp32", "fp16", "bf16"]
    quantization_policy: str
    pooling_policy: str
    template_policy_sha256: str
    engine_signature_sha256: str
    max_input_tokens: int
    max_batch_tokens: int
    permitted_modalities: tuple[Literal["text", "image", "caption_only"], ...]


ENGINE_KEYS = frozenset(
    (
        "weights_revision",
        "tokenizer_revision",
        "compute_precision",
        "quantization_algorithm",
        "quantization_revision",
        "pooling_policy",
        "query_template_sha256",
        "document_template_sha256",
        "image_processor_revision",
        "dimension_policy",
        "normalization",
        "engine_binary_sha256",
        "engine_config_sha256",
        "response_route",
    )
)
_DECLARATIONS = {}


def declare_space(
    engine_manifest,
    *,
    provider,
    model,
    query_template,
    document_template,
    permitted_modalities,
    max_input_tokens,
    max_batch_tokens,
):
    engine = dict(engine_manifest)
    if set(engine) != ENGINE_KEYS or any(
        type(v) is not str or not v for v in engine.values()
    ):
        raise ValueError("incomplete_engine_manifest")
    for key in (
        "query_template_sha256",
        "document_template_sha256",
        "engine_binary_sha256",
        "engine_config_sha256",
    ):
        require_digest(engine[key])
    if engine["query_template_sha256"] != digest(query_template) or engine[
        "document_template_sha256"
    ] != digest(document_template):
        raise ValueError("template_manifest_mismatch")
    template_digest = digest(
        {"query_template": query_template, "document_template": document_template}
    )
    values = dict(
        provider=provider,
        model=model,
        verified_revision=engine["weights_revision"],
        tokenizer_revision=engine["tokenizer_revision"],
        dimensions=1024,
        query_prep_key=digest({"version": "query-prep-v1", "template": query_template}),
        document_prep_key=digest(
            {
                "version": "document-prep-v1",
                "template": document_template,
                "image_processor_revision": engine["image_processor_revision"],
            }
        ),
        dimension_policy=engine["dimension_policy"],
        normalization=engine["normalization"],
        compute_precision=engine["compute_precision"],
        quantization_policy=digest(
            {
                "algorithm": engine["quantization_algorithm"],
                "revision": engine["quantization_revision"],
            }
        ),
        pooling_policy=engine["pooling_policy"],
        template_policy_sha256=template_digest,
        engine_signature_sha256=digest(engine),
        max_input_tokens=max_input_tokens,
        max_batch_tokens=max_batch_tokens,
        permitted_modalities=tuple(permitted_modalities),
    )
    space = VerifiedEmbeddingSpaceV1(space_key=digest(values), **values)
    validate_space(space)
    _DECLARATIONS[space.space_key] = dict(engine)
    return space


def validate_space(space):
    if type(space) is not VerifiedEmbeddingSpaceV1:
        raise ValueError("invalid_embedding_space")
    values = asdict(space)
    key = values.pop("space_key")
    if digest(values) != key or space.provider != "local-openai" or not space.model:
        raise ValueError("invalid_embedding_space")
    if space.dimensions != 1024 or type(space.dimensions) is not int:
        raise ValueError("unsupported_dimensions")
    if space.compute_precision not in ("fp32", "fp16", "bf16"):
        raise ValueError("unsupported_precision")
    if (
        space.dimension_policy not in ("native-1024-v1", "vendor-dimensions-1024-v1")
        or space.normalization != "none"
    ):
        raise ValueError("unsupported_dimension_policy")
    if (
        type(space.permitted_modalities) is not tuple
        or not space.permitted_modalities
        or len(set(space.permitted_modalities)) != len(space.permitted_modalities)
        or not set(space.permitted_modalities) <= {"text", "image", "caption_only"}
    ):
        raise ValueError("invalid_modalities")
    for n in (space.max_input_tokens, space.max_batch_tokens):
        if type(n) is not int or n <= 0:
            raise ValueError("invalid_token_limit")
    return space


@dataclass(frozen=True)
class EmbeddingBuildBudgetV1:
    deadline_at: float
    max_items: int = 32
    max_input_bytes: int = 262144
    clock: Callable[[], float] = field(default=monotonic, repr=False, compare=False)

    def __post_init__(self):
        if type(self.deadline_at) not in (int, float) or not isfinite(self.deadline_at):
            raise ValueError("invalid_deadline")
        if (
            type(self.max_items) is not int
            or not 0 < self.max_items <= 32
            or type(self.max_input_bytes) is not int
            or not 0 < self.max_input_bytes <= 262144
        ):
            raise ValueError("invalid_batch_limit")

    def remaining_seconds(self):
        return max(0.0, self.deadline_at - self.clock())


def build_budget(timeout_ms=120000, *, clock=monotonic):
    if type(timeout_ms) is not int or not 0 < timeout_ms <= 120000:
        raise ValueError("invalid_build_timeout")
    return EmbeddingBuildBudgetV1(clock() + timeout_ms / 1000, clock=clock)


@dataclass(frozen=True)
class QueryEmbeddingObservationV1:
    vector: tuple[float, ...]
    receipt: dict
    prepared_input_sha256: str
    compatibility_key: str | None


@dataclass(frozen=True)
class EmbeddingLeaseV1:
    generation_key: UUID
    lease_token: UUID


@dataclass(frozen=True)
class RankedChunkV1:
    chunk_id: int
    distance: float
    embedding_generation_key: UUID
    representation_key: UUID | None
    source_md5: str


@dataclass(frozen=True)
class DenseGenerationSelectionV1:
    document_id: UUID
    model_label: str
    document_pkid: int
    representation_key: UUID | None
    source_hash: str
    generation_key: UUID
    manifest_sha256: str


@dataclass(frozen=True)
class DenseGenerationSnapshotV1:
    space_key: str
    generation_keys: tuple[UUID, ...]
    manifest_sha256: str
    selections: tuple[DenseGenerationSelectionV1, ...]
