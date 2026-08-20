from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from hashlib import sha256
from hmac import compare_digest
from os import environ
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from .config import QueryExtractorSettings, load_query_extractor_settings
from .contracts import (
    QueryEntitySpanV1,
    QueryExtractionResponseV1,
    QueryExtractorProvenanceV1,
    canonical_query_extraction_response_bytes,
    parse_query_extraction_request,
)

_JSON_HEADERS = [(b"content-type", b"application/json")]


@dataclass(frozen=True, slots=True)
class QueryExtractorRuntime:
    settings: QueryExtractorSettings
    ontology: object
    backend: object


_runtime: QueryExtractorRuntime | None = None


@dataclass(frozen=True, slots=True)
class _ActivatedOntology:
    version: str
    entity_types: object
    relations: object
    checksum: str


def _records(value: object, name: str) -> list[dict[str, object]]:
    if type(value) is not list or not value:
        raise RuntimeError(f"activated ontology {name} must be a nonempty list")
    records: list[dict[str, object]] = []
    for item in value:
        if type(item) is not dict or type(item.get("name")) is not str:
            raise RuntimeError(f"activated ontology {name} is invalid")
        records.append(item)
    if len({row["name"] for row in records}) != len(records):
        raise RuntimeError(f"activated ontology {name} contains duplicate names")
    return records


def _load_activated_ontology(path: Path) -> _ActivatedOntology:
    """Load the immutable sidecar copy of the activated ontology YAML."""

    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise RuntimeError("activated ontology YAML is unavailable") from error
    if type(document) is not dict or set(document) != {
        "version",
        "entity_types",
        "relations",
    }:
        raise RuntimeError("activated ontology YAML has an invalid field set")
    if type(document["version"]) is not str or not document["version"]:
        raise RuntimeError("activated ontology version is invalid")
    entities = _records(document["entity_types"], "entity_types")
    relations = _records(document["relations"], "relations")
    entity_content = []
    for row in sorted(entities, key=lambda item: item["name"]):
        entity_content.append(
            {
                "name": row["name"],
                "description": row["description"],
                "aliases": sorted(row["aliases"]),  # type: ignore[arg-type]
                "default_retrieval_weight": float(
                    row["default_retrieval_weight"]  # type: ignore[arg-type]
                ),
                "default_suppression_policy": row["default_suppression_policy"],
                "default_suppression_threshold": float(
                    row["default_suppression_threshold"]  # type: ignore[arg-type]
                ),
                "extension_enabled": row.get("extension_enabled", False),
            }
        )
    relation_content = []
    for row in sorted(relations, key=lambda item: item["name"]):
        relation_content.append(
            {
                "name": row["name"],
                "description": row["description"],
                "direction": row["direction"],
                "allowed_head_types": sorted(  # type: ignore[arg-type]
                    row["allowed_head_types"]
                ),
                "allowed_tail_types": sorted(  # type: ignore[arg-type]
                    row["allowed_tail_types"]
                ),
            }
        )
    content = {
        "version": document["version"],
        "entity_types": entity_content,
        "relations": relation_content,
    }
    encoded = json.dumps(
        content, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return _ActivatedOntology(
        version=document["version"],
        entity_types=MappingProxyType({row["name"]: row for row in entities}),
        relations=MappingProxyType({row["name"]: row for row in relations}),
        checksum=sha256(encoded).hexdigest(),
    )


def _get_runtime(
    settings: QueryExtractorSettings | None = None,
) -> QueryExtractorRuntime:
    global _runtime
    if _runtime is None:
        from lib.knowledge_graph.config import load_extraction_settings
        from lib.knowledge_graph.extractors.gliner2_local import GLiNER2LocalBackend

        if settings is None:
            settings = load_query_extractor_settings(environ)
        ontology = _load_activated_ontology(settings.ontology_path)
        if ontology.checksum != settings.ontology_checksum:
            raise RuntimeError("activated ontology provenance mismatch")
        extraction = replace(
            load_extraction_settings(environ),
            model_id=settings.model_identifier,
            model_revision=settings.model_revision,
        )
        _runtime = QueryExtractorRuntime(
            settings=settings,
            ontology=ontology,
            backend=GLiNER2LocalBackend(settings=extraction),
        )
    return _runtime


async def _respond(
    send: Callable[..., Awaitable[None]], status: int, body: bytes
) -> None:
    await send(
        {"type": "http.response.start", "status": status, "headers": _JSON_HEADERS}
    )
    await send({"type": "http.response.body", "body": body})


def _authorization(scope: dict[str, Any]) -> str:
    values = [
        value
        for name, value in scope.get("headers", ())
        if name.lower() == b"authorization"
    ]
    if len(values) != 1:
        return ""
    try:
        return values[0].decode("ascii")
    except UnicodeDecodeError:
        return ""


async def _read_body(
    receive: Callable[..., Awaitable[dict[str, Any]]], maximum: int
) -> bytes | None:
    chunks: list[bytes] = []
    size = 0
    while True:
        message = await receive()
        if message.get("type") != "http.request":
            return None
        chunk = message.get("body", b"")
        if type(chunk) is not bytes:
            return None
        size += len(chunk)
        if size > maximum:
            return None
        chunks.append(chunk)
        if not message.get("more_body", False):
            return b"".join(chunks)


def _canonical_spans(entities: object, maximum: int) -> tuple[QueryEntitySpanV1, ...]:
    candidates: dict[tuple[int, int, str], QueryEntitySpanV1] = {}
    for entity in entities:  # type: ignore[union-attr]
        span = QueryEntitySpanV1(
            entity.entity_type,
            entity.start,
            entity.end,
            float(entity.confidence),
        )
        key = (span.start, span.end, span.ontology_type)
        previous = candidates.get(key)
        if previous is None or span.confidence > previous.confidence:
            candidates[key] = span
    selected: list[QueryEntitySpanV1] = []
    for span in sorted(
        candidates.values(), key=lambda row: (row.start, row.end, row.ontology_type)
    ):
        if selected and selected[-1].end > span.start:
            continue
        selected.append(span)
        if len(selected) == maximum:
            break
    return tuple(selected)


async def healthz(scope, receive, send) -> None:
    del scope, receive
    await _respond(send, 200, b'{"status":"ok"}')


async def extract_v1(scope, receive, send) -> None:
    try:
        settings = load_query_extractor_settings(environ)
    except Exception:
        await _respond(send, 503, b'{"reason":"extractor_provenance"}')
        return
    expected_auth = "Bearer " + settings.bearer_token.get_secret_value()
    if not compare_digest(_authorization(scope), expected_auth):
        await _respond(send, 401, b'{"reason":"extractor_auth"}')
        return
    try:
        runtime = _get_runtime(settings)
        if runtime.settings != settings:
            raise RuntimeError("runtime configuration drift")
    except Exception:
        await _respond(send, 503, b'{"reason":"extractor_provenance"}')
        return
    body = await _read_body(receive, settings.max_request_body_bytes)
    if body is None:
        await _respond(send, 413, b'{"reason":"request_too_large"}')
        return
    try:
        request = parse_query_extraction_request(body)
        if (
            request.ontology_checksum != settings.ontology_checksum
            or request.max_query_utf8_bytes != settings.max_query_utf8_bytes
            or request.max_query_code_points != settings.max_query_code_points
            or request.max_spans != settings.max_spans
        ):
            raise ValueError("request provenance or caps mismatch")
        result = runtime.backend.extract_batch(  # type: ignore[union-attr]
            (request.query,), ontology=runtime.ontology
        )
        if type(result) is not tuple or len(result) != 1:
            raise ValueError("invalid extraction batch")
        response = QueryExtractionResponseV1(
            provenance=QueryExtractorProvenanceV1(
                model_identifier=settings.model_identifier,
                model_revision=settings.model_revision,
                schema_version=settings.schema_version,
                schema_checksum=settings.schema_checksum,
                ontology_checksum=settings.ontology_checksum,
                build_hash=settings.build_hash,
            ),
            query_utf8_bytes=len(request.query.encode()),
            query_code_points=len(request.query),
            spans=_canonical_spans(result[0].entities, settings.max_spans),
        )
        payload = canonical_query_extraction_response_bytes(response)
        if len(payload) > settings.max_response_body_bytes:
            raise ValueError("response body exceeds cap")
    except Exception:
        await _respond(send, 422, b'{"reason":"extractor_provenance"}')
        return
    await _respond(send, 200, payload)


async def app(scope, receive, send) -> None:
    if scope.get("type") != "http":
        return
    route = (scope.get("method"), scope.get("path"))
    if route == ("GET", "/healthz"):
        await healthz(scope, receive, send)
    elif route == ("POST", "/v1/extract"):
        await extract_v1(scope, receive, send)
    else:
        await _respond(send, 404, b'{"reason":"not_found"}')


def run() -> None:
    import uvicorn

    uvicorn.run("lib.knowledge_graph.query_extractor.service:app", access_log=False)


if __name__ == "__main__":
    run()
