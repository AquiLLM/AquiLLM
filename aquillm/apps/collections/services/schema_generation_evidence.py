"""Aggregate local extraction evidence without returning source text."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from math import isfinite

from .schema_candidate import _candidate_ontology_yaml, normalize_schema_candidate
from .schema_generation import (
    _MIN_ENTITY_TYPES,
    _MIN_RELATION_TYPES,
    InvalidSchemaCandidate,
    SchemaSample,
)
from .schema_generation_support import _sample_text


def _sample_reference(sample: object) -> dict[str, object]:
    if isinstance(sample, SchemaSample):
        return {"document_id": sample.document_id, "chunk_id": sample.chunk_id}
    if (
        isinstance(sample, dict)
        and isinstance(sample.get("document_id"), str)
        and type(sample.get("chunk_id")) is int
    ):
        return {"document_id": sample["document_id"], "chunk_id": sample["chunk_id"]}
    raise InvalidSchemaCandidate("samples must contain source references")


def _definitions_for_evidence(candidate: object) -> dict:
    if isinstance(candidate, dict) and all(
        isinstance(candidate.get(kind), list) for kind in ("entities", "relations")
    ):
        if all(
            isinstance(item, dict) and "key" in item and "values" in item
            for kind in ("entities", "relations")
            for item in candidate[kind]
        ):
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
    results = (backend or _default_backend()).extract_batch(
        tuple(_sample_text(sample) for sample in samples), ontology=ontology
    )
    if len(results) != len(samples):
        raise RuntimeError("local GLiNER2 returned an invalid result batch")
    entities, relations = defaultdict(list), defaultdict(list)
    entity_sources, relation_sources = defaultdict(list), defaultdict(list)
    for sample, result in zip(samples, results, strict=True):
        reference = _sample_reference(sample)
        for mention in result.entities:
            if mention.entity_type in ontology.entity_types and isfinite(
                mention.confidence
            ):
                entities[mention.entity_type].append(float(mention.confidence))
                if (
                    reference not in entity_sources[mention.entity_type]
                    and len(entity_sources[mention.entity_type]) < 3
                ):
                    entity_sources[mention.entity_type].append(reference)
        for mention in result.relations:
            if mention.relation_type in ontology.relations and isfinite(
                mention.confidence
            ):
                relations[mention.relation_type].append(float(mention.confidence))
                if (
                    reference not in relation_sources[mention.relation_type]
                    and len(relation_sources[mention.relation_type]) < 3
                ):
                    relation_sources[mention.relation_type].append(reference)
    kept_entities = [item for item in definitions["entities"] if entities[item["key"]]]
    keys = {item["key"] for item in kept_entities}
    kept_relations = [
        item
        for item in definitions["relations"]
        if relations[item["key"]]
        and set(item["values"]["allowed_head_types"])
        .union(item["values"]["allowed_tail_types"])
        .issubset(keys)
    ]
    if (
        len(kept_entities) < _MIN_ENTITY_TYPES
        or len(kept_relations) < _MIN_RELATION_TYPES
    ):
        raise InvalidSchemaCandidate("candidate has insufficient local evidence")

    def stats(values, sources, allowed):
        return {
            key: {
                "count": len(confidences),
                "mean_confidence": sum(confidences) / len(confidences),
                "sources": sources[key],
            }
            for key, confidences in sorted(values.items())
            if key in allowed
        }

    return {"entities": kept_entities, "relations": kept_relations}, {
        "entities": stats(entities, entity_sources, keys),
        "relations": stats(
            relations, relation_sources, {item["key"] for item in kept_relations}
        ),
    }
