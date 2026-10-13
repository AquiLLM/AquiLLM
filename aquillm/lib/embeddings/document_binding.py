"""Internal serving evidence, kept separate from immutable v1 response receipts.

The envelope is outside DTO dataclass fields. Generic reconstruction does not
mint a binding; internal caches must use the checked serializer/hydrator below.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
from math import isfinite

from .document_contracts import (
    QueryEmbeddingObservationV1,
    VerifiedEmbeddingSpaceV1,
    digest,
)
from .document_preparation import verified_profile
from .provenance import vector_digest
from .utils import EmbeddingContractError


@dataclass(frozen=True)
class ServingBindingV1:
    space_key: str
    verification_sha256: str
    prepared_input_sha256: str
    vector_sha256: str
    role: str
    modality: str
    media_sha256: str | None


def _binding(space, receipt, *, modality, media_sha256):
    if type(receipt) is not dict or not {
        "prepared_input_sha256",
        "vector_sha256",
        "role",
    } <= set(receipt):
        raise EmbeddingContractError("serving_binding_mismatch")
    verified = verified_profile(space)
    evidence = {key: value for key, value in verified.items() if key != "space"}
    return ServingBindingV1(
        space.space_key,
        digest(evidence),
        receipt["prepared_input_sha256"],
        receipt["vector_sha256"],
        receipt["role"],
        modality,
        media_sha256,
    )


def bind_response(result, *, space, modality, media_sha256=None):
    """Only the strict transport calls this, after an endpoint-bound response."""
    object.__setattr__(
        result,
        "_serving_binding",
        _binding(
            space, result.provenance, modality=modality, media_sha256=media_sha256
        ),
    )
    return result


def stored_binding(result, *, space, modality, media_sha256):
    actual = getattr(result, "_serving_binding", None)
    expected = _binding(
        space, result.provenance, modality=modality, media_sha256=media_sha256
    )
    if (
        type(actual) is not ServingBindingV1
        or actual != expected
        or vector_digest(result.vector) != actual.vector_sha256
    ):
        raise EmbeddingContractError("serving_binding_mismatch")
    return asdict(actual)


def validate_stored_binding(binding, vector, receipt, *, space, modality, media_sha256):
    expected = asdict(
        _binding(space, receipt, modality=modality, media_sha256=media_sha256)
    )
    if (
        type(binding) is not dict
        or binding != expected
        or vector_digest(vector) != expected["vector_sha256"]
    ):
        raise EmbeddingContractError("serving_binding_mismatch")


def bind_query_observation(observation, result):
    binding = getattr(result, "_serving_binding", None)
    if type(binding) is not ServingBindingV1:
        raise EmbeddingContractError("serving_binding_mismatch")
    object.__setattr__(observation, "_serving_binding", binding)
    return observation


def validate_query_observation(
    observation: QueryEmbeddingObservationV1, *, space: VerifiedEmbeddingSpaceV1
) -> str | None:
    from .document_provider import compatible_receipt

    if type(observation) is not QueryEmbeddingObservationV1:
        raise EmbeddingContractError("serving_binding_mismatch")
    actual = getattr(observation, "_serving_binding", None)
    expected = _binding(space, observation.receipt, modality="text", media_sha256=None)
    if type(actual) is not ServingBindingV1 or actual != expected:
        raise EmbeddingContractError("serving_binding_mismatch")
    key = compatible_receipt(
        observation.vector,
        observation.receipt,
        space=space,
        role="search_query",
        prepared_input_sha256=observation.prepared_input_sha256,
    )
    if key != observation.compatibility_key:
        raise EmbeddingContractError("serving_binding_mismatch")
    return key


def serialize_query_observation(
    observation: QueryEmbeddingObservationV1, *, space: VerifiedEmbeddingSpaceV1
) -> dict:
    validate_query_observation(observation, space=space)
    return {
        "version": "query-embedding-observation-v1",
        "vector": [float(v) for v in observation.vector],
        "receipt": deepcopy(observation.receipt),
        "prepared_input_sha256": observation.prepared_input_sha256,
        "compatibility_key": observation.compatibility_key,
        "serving_binding": asdict(observation._serving_binding),
    }


def restore_query_observation(
    payload: dict, *, space: VerifiedEmbeddingSpaceV1
) -> QueryEmbeddingObservationV1:
    keys = {
        "version",
        "vector",
        "receipt",
        "prepared_input_sha256",
        "compatibility_key",
        "serving_binding",
    }
    try:
        if (
            type(payload) is not dict
            or set(payload) != keys
            or payload["version"] != "query-embedding-observation-v1"
        ):
            raise ValueError()
        vector = payload["vector"]
        if (
            type(vector) is not list
            or len(vector) != 1024
            or any(type(v) is not float or not isfinite(v) for v in vector)
        ):
            raise ValueError()
        binding = ServingBindingV1(**payload["serving_binding"])
        observation = QueryEmbeddingObservationV1(
            tuple(vector),
            deepcopy(payload["receipt"]),
            payload["prepared_input_sha256"],
            payload["compatibility_key"],
        )
        object.__setattr__(observation, "_serving_binding", binding)
        validate_query_observation(observation, space=space)
        return observation
    except (ValueError, TypeError, KeyError, AttributeError):
        raise EmbeddingContractError("serving_binding_mismatch") from None
