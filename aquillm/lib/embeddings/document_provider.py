"""Strict response-bound document/query adapter: one exact request, no fallback."""

from copy import deepcopy
from .config import get_local_embed_config
from .document_contracts import QueryEmbeddingObservationV1
from .document_preparation import (
    engine_manifest_for as engine_manifest_for,
    register_verified_profile as register_verified_profile,
    verified_profile as verified_profile,
    prepare_query as prepare_query,
    prepare_document as prepare_document,
)
from .errors import EmbeddingUpstreamUnavailableError, require_transient
from .local import _get_local_openai_client, _response_vectors
from .provenance import input_digest, make_result, valid_provenance
from .utils import EmbeddingContractError
from lib.retrieval.turn_budget import TurnBudget
from .document_binding import bind_response, bind_query_observation


def _endpoint():
    return get_local_embed_config()[0].rstrip("/")


def _client():
    base, key, _ = get_local_embed_config()
    return _get_local_openai_client(base, key)


def _remaining(budget):
    value = (
        budget.remaining_seconds()
        if hasattr(budget, "remaining_seconds")
        else budget.remaining_ms() / 1000
    )
    if value <= 0:
        raise TimeoutError("embedding_deadline")
    return min(30.0, value)


def compatible_receipt(vector, receipt, *, space, role, prepared_input_sha256):
    verified_profile(space)
    checked = valid_provenance(vector, receipt)
    if (
        checked is None
        or len(vector) != 1024
        or checked["provider"] != space.provider
        or checked["route"] != "openai-embeddings"
        or checked["role"] != role
        or checked["prepared_input_sha256"] != prepared_input_sha256
        or checked["dimensions"] != {"raw": 1024, "fitted": 1024, "adaptation": "none"}
        or checked["input_transformations"] != []
        or checked["declared_identity"]["model"] != space.model
    ):
        raise EmbeddingContractError("incompatible_embedding_receipt")
    observed = checked["observed_identity"]["model"]
    if observed is not None and observed != space.model:
        raise EmbeddingContractError("response_model_mismatch")
    return space.space_key if observed == space.model else None


def _observe(prepared, *, space, budget, role):
    binding = verified_profile(space)
    _remaining(budget)
    if _endpoint() != binding["endpoint"]:
        raise EmbeddingContractError("endpoint_binding_mismatch")
    client = _client()
    # UTF-8 byte count is a conservative token upper bound for supported byte
    # tokenizers; no text is shortened to fit an endpoint context.
    sizes = [
        len((p if isinstance(p, str) else p["text"]).encode("utf-8")) for p in prepared
    ]
    if (
        any(n > space.max_input_tokens for n in sizes)
        or sum(sizes) > space.max_batch_tokens
    ):
        raise EmbeddingContractError("prepared_input_token_limit")
    kwargs = (
        {"dimensions": 1024}
        if space.dimension_policy == "vendor-dimensions-1024-v1"
        else {}
    )
    timeout = _remaining(budget)
    try:
        response = client.embeddings.create(
            model=space.model,
            input=list(prepared),
            timeout=timeout,
            **kwargs,
        )
    except Exception as exc:
        require_transient(exc)
        raise EmbeddingUpstreamUnavailableError(
            "embedding_upstream_unavailable"
        ) from None
    _remaining(budget)
    vectors = _response_vectors(response, len(prepared))
    results = []
    for vector, wire in zip(vectors, prepared):
        result = make_result(
            vector,
            provider=space.provider,
            route="openai-embeddings",
            role=role,
            prepared_input=wire,
            original_input=wire,
            model=space.model,
            response_model=getattr(response, "model", None),
        )
        compatible_receipt(
            vector,
            result.provenance,
            space=space,
            role=role,
            prepared_input_sha256=input_digest(wire),
        )
        results.append(result)
    return tuple(results)


def observe_query_embedding(query, *, space, budget):
    if not isinstance(budget, TurnBudget):
        raise TypeError("query_requires_turn_budget")
    prepared, sha = prepare_query(query, space=space)
    (result,) = _observe((prepared,), space=space, budget=budget, role="search_query")
    key = compatible_receipt(
        result.vector,
        result.provenance,
        space=space,
        role="search_query",
        prepared_input_sha256=sha,
    )
    bind_response(result, space=space, modality="text")
    observation = QueryEmbeddingObservationV1(
        tuple(float(v) for v in result.vector), deepcopy(result.provenance), sha, key
    )
    return bind_query_observation(observation, result)


def embed_document_batch(chunks, *, space, budget):
    _remaining(budget)
    if type(chunks) is not tuple or not 0 < len(chunks) <= budget.max_items:
        raise EmbeddingContractError("document_batch_limit")
    prepared = tuple(prepare_document(chunk, space=space)[0] for chunk in chunks)
    size = sum(
        len((p if isinstance(p, str) else p["text"]).encode("utf-8")) for p in prepared
    )
    if size > budget.max_input_bytes:
        raise EmbeddingContractError("document_batch_bytes")
    results = _observe(prepared, space=space, budget=budget, role="search_document")
    return tuple(
        bind_response(
            result,
            space=space,
            modality=chunk.modality,
            media_sha256=chunk.media_artifact.sha256 if chunk.media_artifact else None,
        )
        for result, chunk in zip(results, chunks)
    )
