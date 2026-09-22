# ruff: noqa: E402, I001 - candidate/evidence helpers follow adapter definitions
"""Candidate parsing and evidence aggregation for local schema generation."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .schema_generation import (
    _MAX_ENTITY_TYPES,
    _MAX_RELATION_TYPES,
    _MIN_ENTITY_TYPES,
    _MIN_RELATION_TYPES,
    InvalidSchemaCandidate,
    SchemaSample,
    load_schema_generation_config,
)

_MAX_VLLM_RESPONSE_BYTES = 1_000_000
_SEMANTIC_CORRECTION = (
    "\n\nThe previous candidate failed semantic validation. Return a corrected, "
    "different candidate. Entity and relation names must remain unique after "
    "lowercase snake_case "
    "normalization. Every relation endpoint must exactly match an entity name in this "
    "same response. Keep each entity's aliases at no more than 1024 total characters."
)


def _schema_response_format() -> dict:
    text = {"type": "string", "minLength": 1, "maxLength": 512}
    name = {"type": "string", "minLength": 1, "maxLength": 64}
    entity = {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "description", "aliases"],
        "properties": {
            "name": name,
            "description": text,
            "aliases": {
                "type": "array",
                "maxItems": _MAX_ALIASES_PER_ENTITY,
                "items": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": _MAX_ALIAS_CHARACTERS,
                },
            },
        },
    }
    endpoints = {
        "type": "array",
        "minItems": 1,
        "maxItems": _MAX_ENTITY_TYPES,
        "items": name,
    }
    relation = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "name",
            "description",
            "direction",
            "allowed_head_types",
            "allowed_tail_types",
        ],
        "properties": {
            "name": name,
            "description": text,
            "direction": {"type": "string", "enum": ["directed", "undirected"]},
            "allowed_head_types": endpoints,
            "allowed_tail_types": endpoints,
        },
    }
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["entities", "relations"],
        "properties": {
            "entities": {
                "type": "array",
                "minItems": _MIN_ENTITY_TYPES,
                "maxItems": _MAX_ENTITY_TYPES,
                "items": entity,
            },
            "relations": {
                "type": "array",
                "minItems": _MIN_RELATION_TYPES,
                "maxItems": _MAX_RELATION_TYPES,
                "items": relation,
            },
        },
    }
    return {
        "type": "json_schema",
        "json_schema": {"name": "collection_schema", "strict": True, "schema": schema},
    }


class LocalVLLMTransportError(RuntimeError):
    """The local adapter did not receive a bounded, successful JSON response."""


class _RejectRedirects(HTTPRedirectHandler):
    """A local vLLM request must not be redirected to another host."""

    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


_LOCAL_URL_OPENER = build_opener(ProxyHandler({}), _RejectRedirects())


def _open_local_request(request: Request, timeout: int):
    return _LOCAL_URL_OPENER.open(request, timeout=timeout)


def _sample_text(sample: object) -> str:
    if isinstance(sample, SchemaSample):
        return sample.text
    if isinstance(sample, dict) and isinstance(sample.get("text"), str):
        return sample["text"]
    raise InvalidSchemaCandidate("samples must contain text")


def _post_local_vllm_json(
    url: str, payload: dict, headers: dict[str, str], timeout: int
) -> dict:
    """Post JSON directly without an SDK logger that can render request bodies."""

    request = Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with _open_local_request(request, timeout=timeout) as response:  # noqa: S310 - validated local Docker URL
            status = getattr(response, "status", None)
            if status is None:
                status = response.getcode()
            if not 200 <= status < 300:
                raise LocalVLLMTransportError(
                    "local vLLM returned a non-success status"
                )
            content_length = response.headers.get("Content-Length")
            if (
                content_length is not None
                and int(content_length) > _MAX_VLLM_RESPONSE_BYTES
            ):
                raise LocalVLLMTransportError(
                    "local vLLM response exceeded the size limit"
                )
            body = response.read(_MAX_VLLM_RESPONSE_BYTES + 1)
    except LocalVLLMTransportError:
        raise
    except (HTTPError, URLError, OSError, TimeoutError, ValueError) as exc:
        raise LocalVLLMTransportError("local vLLM transport failed") from exc
    if len(body) > _MAX_VLLM_RESPONSE_BYTES:
        raise LocalVLLMTransportError("local vLLM response exceeded the size limit")
    try:
        decoded = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LocalVLLMTransportError("local vLLM response was malformed") from exc
    if not isinstance(decoded, dict):
        raise LocalVLLMTransportError("local vLLM response must be a JSON object")
    return decoded


def generate_schema_candidate(samples, client=None) -> dict:
    """Ask local vLLM for a strict proposal, with one bounded semantic repair."""

    config = load_schema_generation_config()
    texts = [_sample_text(sample) for sample in samples]
    if not texts:
        raise InvalidSchemaCandidate("at least one sample is required")
    prompt = (
        "Produce only JSON with entities and relations. Use 2-24 entity types "
        "and 1-32 relation types. Every entity needs name, description, aliases. "
        "Every relation needs name, description, direction, allowed_head_types, "
        "and allowed_tail_types. Names must be concise and endpoints must name "
        "entities.\n\n"
        + "\n\n".join(texts)
    )
    local_client = client or _post_local_vllm_json
    for attempt in range(2):
        try:
            response = local_client(
                f"{config.base_url}/chat/completions",
                {
                    "model": config.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                            + (_SEMANTIC_CORRECTION if attempt else ""),
                        }
                    ],
                    "response_format": _schema_response_format(),
                    "chat_template_kwargs": {"enable_thinking": False},
                    "temperature": 0,
                },
                {
                    "Authorization": f"Bearer {config.api_key}",
                    "Content-Type": "application/json",
                },
                config.timeout_seconds,
            )
        except Exception as exc:
            raise RuntimeError("local vLLM inference failed") from exc
        try:
            return normalize_schema_candidate(
                json.loads(response["choices"][0]["message"]["content"])
            )
        except (IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            if attempt == 1:
                raise InvalidSchemaCandidate(
                    "local vLLM returned an invalid candidate"
                ) from exc
    raise AssertionError("unreachable schema generation attempt state")


from .schema_candidate import (
    _MAX_ALIAS_CHARACTERS,
    _MAX_ALIASES_PER_ENTITY,
    normalize_schema_candidate,
)
from .schema_generation_evidence import (
    _default_backend as _default_backend,
    collect_candidate_evidence as collect_candidate_evidence,
)
