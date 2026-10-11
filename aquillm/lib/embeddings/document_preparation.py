"""Explicit immutable serving verification and deterministic input preparation."""

from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import base64
from threading import RLock
from .document_contracts import (
    _DECLARATIONS,
    declare_space,
    require_digest,
    validate_space,
)
from .provenance import input_digest
from .utils import EmbeddingContractError

_PROFILES = {}
_LOCK = RLock()


def engine_manifest_for(space):
    return deepcopy(_DECLARATIONS[space.space_key])


def register_verified_profile(
    space,
    *,
    endpoint,
    deployment_artifact_sha256,
    engine_manifest,
    query_template,
    document_template,
):
    """Install independently verified immutable evidence; never alter a receipt."""
    validate_space(space)
    require_digest(deployment_artifact_sha256)
    rebuilt = declare_space(
        engine_manifest,
        provider=space.provider,
        model=space.model,
        query_template=query_template,
        document_template=document_template,
        permitted_modalities=space.permitted_modalities,
        max_input_tokens=space.max_input_tokens,
        max_batch_tokens=space.max_batch_tokens,
    )
    if (
        rebuilt != space
        or type(endpoint) is not str
        or not endpoint.startswith(("http://", "https://"))
    ):
        raise EmbeddingContractError("invalid_endpoint_binding")
    if engine_manifest["response_route"] != "openai-embeddings":
        raise EmbeddingContractError("unsupported_response_route")
    for template in (query_template, document_template):
        if (
            type(template) is not str
            or template.count("{text}") != 1
            or "{" in template.replace("{text}", "")
            or "}" in template.replace("{text}", "")
        ):
            raise EmbeddingContractError("unsupported_template")
    binding = dict(
        space=space,
        endpoint=endpoint.rstrip("/"),
        deployment_artifact_sha256=deployment_artifact_sha256,
        engine_manifest=deepcopy(engine_manifest),
        query_template=query_template,
        document_template=document_template,
    )
    with _LOCK:
        old = _PROFILES.get(space.space_key)
        if old is not None and old != binding:
            raise EmbeddingContractError("immutable_endpoint_binding_changed")
        _PROFILES[space.space_key] = binding
    return deepcopy(binding)


def verified_profile(space):
    validate_space(space)
    with _LOCK:
        binding = _PROFILES.get(space.space_key)
        if binding is None or binding["space"] != space:
            raise EmbeddingContractError("embedding_space_unverified")
        return deepcopy(binding)


def prepare_query(query, *, space):
    binding = verified_profile(space)
    if type(query) is not str:
        raise EmbeddingContractError("query_text_required")
    prepared = binding["query_template"].replace("{text}", query)
    return prepared, input_digest(prepared)


def prepare_document(chunk, *, space):
    binding = verified_profile(space)
    if (
        chunk.source_sha256 != sha256(chunk.source_text.encode("utf-8")).hexdigest()
        or chunk.input_sha256 != input_digest(chunk.embedding_text)
        or chunk.modality not in space.permitted_modalities
    ):
        raise EmbeddingContractError("source_projection_mismatch")
    text = binding["document_template"].replace("{text}", chunk.embedding_text)
    if chunk.modality == "image":
        from django.core.files.storage import default_storage
        from lib.parsers.source_contracts import validate_artifact_ref
        from PIL import Image

        ref = chunk.media_artifact
        validate_artifact_ref(ref)
        if ref.size_bytes > 32 * 1024 * 1024:
            raise EmbeddingContractError("media_too_large")
        try:
            with default_storage.open(ref.storage_key, "rb") as stream:
                pixels = stream.read(ref.size_bytes + 1)
            validate_artifact_ref(ref, data=pixels)
            if (
                binding["engine_manifest"]["image_processor_revision"]
                != "verified-png-v1"
            ):
                raise EmbeddingContractError("unsupported_image_policy")
            with Image.open(BytesIO(pixels)) as image:
                image.load()
                output = BytesIO()
                image.convert("RGB").save(output, format="PNG")
            encoded = base64.b64encode(output.getvalue()).decode("ascii")
        except Exception as exc:
            raise EmbeddingContractError("media_unavailable") from exc
        prepared = {"text": text, "image": "data:image/png;base64," + encoded}
    else:
        # Caption-only is an explicit actual modality, not error fallback.
        if chunk.modality == "text" and chunk.media_artifact is not None:
            raise EmbeddingContractError("text_media_mismatch")
        prepared = text
    return prepared, input_digest(prepared)


def prepared_input_bytes(wire):
    """Count prepared text plus the complete encoded image data URI, if present."""
    values = (wire,) if isinstance(wire, str) else (wire["text"], wire["image"])
    return sum(len(value.encode("utf-8")) for value in values)
