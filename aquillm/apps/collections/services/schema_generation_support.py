"""Candidate parsing and evidence aggregation for local schema generation."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import replace
from math import isfinite
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

import yaml

from .schema_generation import (
    InvalidSchemaCandidate,
    SchemaGenerationConfig,
    SchemaSample,
    _MAX_ENTITY_TYPES,
    _MAX_RELATION_TYPES,
    _MIN_ENTITY_TYPES,
    _MIN_RELATION_TYPES,
    _SNAKE_CASE,
    load_schema_generation_config,
)

_MAX_VLLM_RESPONSE_BYTES = 1_000_000
_MAX_ALIASES_PER_ENTITY = 16
_MAX_ALIAS_CHARACTERS = 128
_MAX_ALIAS_TOTAL_CHARACTERS = 1_024


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
            "name", "description", "direction",
            "allowed_head_types", "allowed_tail_types",
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
                "type": "array", "minItems": _MIN_ENTITY_TYPES,
                "maxItems": _MAX_ENTITY_TYPES, "items": entity,
            },
            "relations": {
                "type": "array", "minItems": _MIN_RELATION_TYPES,
                "maxItems": _MAX_RELATION_TYPES, "items": relation,
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


def _snake_case(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidSchemaCandidate(f"{field} must be a nonempty string")
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if not _SNAKE_CASE.fullmatch(normalized):
        raise InvalidSchemaCandidate(f"{field} must normalize to lowercase snake case")
    return normalized


def _description(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 512:
        raise InvalidSchemaCandidate(f"{field} must be a nonempty string up to 512 characters")
    return value.strip()


def _aliases(value: object) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(alias, str) or not alias.strip() for alias in value):
        raise InvalidSchemaCandidate("aliases must be a list of nonempty strings")
    aliases = sorted(set(alias.strip() for alias in value))
    if len(aliases) > _MAX_ALIASES_PER_ENTITY:
        raise InvalidSchemaCandidate("aliases exceed the per-entity count limit")
    if any(len(alias) > _MAX_ALIAS_CHARACTERS for alias in aliases):
        raise InvalidSchemaCandidate("aliases exceed the per-alias character limit")
    if sum(len(alias) for alias in aliases) > _MAX_ALIAS_TOTAL_CHARACTERS:
        raise InvalidSchemaCandidate("aliases exceed the per-entity character limit")
    return aliases


def _candidate_ontology_yaml(definitions: dict) -> str:
    entity_types = [
        {
            "name": item["key"], "description": item["values"]["description"],
            "aliases": item["values"]["aliases"],
            "default_retrieval_weight": item["values"]["default_retrieval_weight"],
            "default_suppression_policy": item["values"]["default_suppression_policy"],
            "default_suppression_threshold": item["values"]["default_suppression_threshold"],
        } for item in definitions["entities"]
    ]
    relations = [
        {
            "name": item["key"], "description": item["values"]["description"],
            "direction": item["values"]["direction"],
            "allowed_head_types": item["values"]["allowed_head_types"],
            "allowed_tail_types": item["values"]["allowed_tail_types"],
        } for item in definitions["relations"]
    ]
    return yaml.safe_dump({"version": "1.0.0", "entity_types": entity_types, "relations": relations}, sort_keys=True)


def normalize_schema_candidate(candidate: object) -> dict:
    """Convert strict model JSON into the canonical editor definition shape."""

    if not isinstance(candidate, dict):
        raise InvalidSchemaCandidate("candidate must be a JSON object")
    entity_records, relation_records = candidate.get("entities"), candidate.get("relations")
    if not isinstance(entity_records, list) or not (_MIN_ENTITY_TYPES <= len(entity_records) <= _MAX_ENTITY_TYPES):
        raise InvalidSchemaCandidate("candidate must contain 2-24 entities")
    if not isinstance(relation_records, list) or not (_MIN_RELATION_TYPES <= len(relation_records) <= _MAX_RELATION_TYPES):
        raise InvalidSchemaCandidate("candidate must contain 1-32 relations")
    entities, entity_keys = [], set()
    for record in entity_records:
        if not isinstance(record, dict):
            raise InvalidSchemaCandidate("entity definitions must be objects")
        key = _snake_case(record.get("name"), "entity name")
        if key in entity_keys:
            raise InvalidSchemaCandidate("duplicate entity name")
        entity_keys.add(key)
        entities.append({
            "key": key, "origin": "generated", "change_state": "added",
            "capabilities": {"editable_fields": ["description", "aliases", "default_retrieval_weight", "default_suppression_policy", "default_suppression_threshold"], "removable": True, "renameable": False},
            "values": {"name": key, "description": _description(record.get("description"), "entity description"), "aliases": _aliases(record.get("aliases")), "default_retrieval_weight": 0.5, "default_suppression_policy": "none", "default_suppression_threshold": 0.0},
        })
    relations, relation_keys = [], set()
    for record in relation_records:
        if not isinstance(record, dict):
            raise InvalidSchemaCandidate("relation definitions must be objects")
        key = _snake_case(record.get("name"), "relation name")
        if key in relation_keys:
            raise InvalidSchemaCandidate("duplicate relation name")
        relation_keys.add(key)
        direction = record.get("direction")
        if direction not in {"directed", "undirected"}:
            raise InvalidSchemaCandidate("relation direction must be directed or undirected")
        heads = [_snake_case(value, "relation head endpoint") for value in record.get("allowed_head_types", [])]
        tails = [_snake_case(value, "relation tail endpoint") for value in record.get("allowed_tail_types", [])]
        if not heads or not tails or set(heads).union(tails).difference(entity_keys):
            raise InvalidSchemaCandidate("relation has an unknown endpoint type")
        relations.append({
            "key": key, "origin": "generated", "change_state": "added",
            "capabilities": {"editable_fields": ["description", "direction", "allowed_head_types", "allowed_tail_types"], "removable": True, "renameable": False},
            "values": {"name": key, "description": _description(record.get("description"), "relation description"), "direction": direction, "allowed_head_types": sorted(set(heads)), "allowed_tail_types": sorted(set(tails))},
        })
    definitions = {"entities": sorted(entities, key=lambda item: item["key"]), "relations": sorted(relations, key=lambda item: item["key"])}
    try:
        from apps.knowledge_graph.services.ontology import load_ontology_yaml
        load_ontology_yaml(_candidate_ontology_yaml(definitions))
    except Exception as exc:
        raise InvalidSchemaCandidate("candidate fails ontology validation") from exc
    return definitions


def _sample_text(sample: object) -> str:
    if isinstance(sample, SchemaSample):
        return sample.text
    if isinstance(sample, dict) and isinstance(sample.get("text"), str):
        return sample["text"]
    raise InvalidSchemaCandidate("samples must contain text")


def _post_local_vllm_json(url: str, payload: dict, headers: dict[str, str], timeout: int) -> dict:
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
                raise LocalVLLMTransportError("local vLLM returned a non-success status")
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) > _MAX_VLLM_RESPONSE_BYTES:
                raise LocalVLLMTransportError("local vLLM response exceeded the size limit")
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
    """Ask the configured local vLLM server for one strict JSON proposal."""

    config = load_schema_generation_config()
    texts = [_sample_text(sample) for sample in samples]
    if not texts:
        raise InvalidSchemaCandidate("at least one sample is required")
    prompt = (
        "Produce only JSON with entities and relations. Use 2-24 entity types and 1-32 relation types. "
        "Every entity needs name, description, aliases. Every relation needs name, description, direction, "
        "allowed_head_types, and allowed_tail_types. Names must be concise and endpoints must name entities.\n\n"
        + "\n\n".join(texts)
    )
    try:
        response = (client or _post_local_vllm_json)(
            f"{config.base_url}/chat/completions",
            {
                "model": config.model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": _schema_response_format(),
                "chat_template_kwargs": {"enable_thinking": False},
                "temperature": 0,
            },
            {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"},
            config.timeout_seconds,
        )
        return normalize_schema_candidate(json.loads(response["choices"][0]["message"]["content"]))
    except (IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise InvalidSchemaCandidate("local vLLM returned an invalid candidate") from exc
    except Exception as exc:
        raise RuntimeError("local vLLM inference failed") from exc


def _sample_reference(sample: object) -> dict[str, object]:
    if isinstance(sample, SchemaSample):
        return {"document_id": sample.document_id, "chunk_id": sample.chunk_id}
    if isinstance(sample, dict) and isinstance(sample.get("document_id"), str) and type(sample.get("chunk_id")) is int:
        return {"document_id": sample["document_id"], "chunk_id": sample["chunk_id"]}
    raise InvalidSchemaCandidate("samples must contain source references")


def _definitions_for_evidence(candidate: object) -> dict:
    if isinstance(candidate, dict) and all(isinstance(candidate.get(kind), list) for kind in ("entities", "relations")):
        if all(isinstance(item, dict) and "key" in item and "values" in item for kind in ("entities", "relations") for item in candidate[kind]):
            return candidate
    return normalize_schema_candidate(candidate)


def _default_backend():
    from lib.knowledge_graph.config import load_extraction_settings
    from lib.knowledge_graph.extractors.factory import get_extraction_backend

    settings = replace(
        load_extraction_settings(),
        provider="gliner2_local",
        local_files_only=True,
        fail_open=False,
    )
    return get_extraction_backend(settings=settings)


def collect_candidate_evidence(candidate, samples, backend=None) -> tuple[dict, dict]:
    """Keep evidence-backed definitions and aggregate text-free statistics only."""

    definitions, samples = _definitions_for_evidence(candidate), list(samples)
    from apps.knowledge_graph.services.ontology import load_ontology_yaml
    ontology = load_ontology_yaml(_candidate_ontology_yaml(definitions))
    results = (backend or _default_backend()).extract_batch(tuple(_sample_text(sample) for sample in samples), ontology=ontology)
    if len(results) != len(samples):
        raise RuntimeError("local GLiNER2 returned an invalid result batch")
    entities, relations = defaultdict(list), defaultdict(list)
    entity_sources, relation_sources = defaultdict(list), defaultdict(list)
    for sample, result in zip(samples, results, strict=True):
        reference = _sample_reference(sample)
        for mention in result.entities:
            if mention.entity_type in ontology.entity_types and isfinite(mention.confidence):
                entities[mention.entity_type].append(float(mention.confidence))
                if reference not in entity_sources[mention.entity_type] and len(entity_sources[mention.entity_type]) < 3:
                    entity_sources[mention.entity_type].append(reference)
        for mention in result.relations:
            if mention.relation_type in ontology.relations and isfinite(mention.confidence):
                relations[mention.relation_type].append(float(mention.confidence))
                if reference not in relation_sources[mention.relation_type] and len(relation_sources[mention.relation_type]) < 3:
                    relation_sources[mention.relation_type].append(reference)
    kept_entities = [item for item in definitions["entities"] if entities[item["key"]]]
    keys = {item["key"] for item in kept_entities}
    kept_relations = [item for item in definitions["relations"] if relations[item["key"]] and set(item["values"]["allowed_head_types"]).union(item["values"]["allowed_tail_types"]).issubset(keys)]
    if len(kept_entities) < _MIN_ENTITY_TYPES or len(kept_relations) < _MIN_RELATION_TYPES:
        raise InvalidSchemaCandidate("candidate has insufficient local evidence")
    def stats(values, sources, allowed):
        return {key: {"count": len(confidences), "mean_confidence": sum(confidences) / len(confidences), "sources": sources[key]} for key, confidences in sorted(values.items()) if key in allowed}
    return {"entities": kept_entities, "relations": kept_relations}, {"entities": stats(entities, entity_sources, keys), "relations": stats(relations, relation_sources, {item["key"] for item in kept_relations})}
