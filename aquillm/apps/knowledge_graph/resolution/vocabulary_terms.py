"""Negative endpoint footprints from already validated source memberships."""
from dataclasses import replace

from apps.collections.services.vocabulary_contracts import MAX_BYTES, MAX_ENTRIES, normalized
from .vocabulary_capture import VocabularyGraphUnavailable


def validate_terms(terms):
    if (type(terms) is not tuple or len(terms) > MAX_ENTRIES * 2
            or any(type(t) is not str or not t or len(t) > 512 or t != normalized(t) for t in terms)
            or tuple(sorted(set(terms))) != terms or len(repr(terms).encode()) > MAX_BYTES):
        raise VocabularyGraphUnavailable("invalid negative vocabulary footprint")


def terms_for(vocabulary, surfaces, *, budget=None):
    if vocabulary is None:
        return ()
    applicable = {t for _, pairs in vocabulary.scopes for pair in pairs for t in pair}
    terms = tuple(sorted({normalized(s) for s in surfaces if normalized(s) in applicable}))
    validate_terms(terms)
    if budget is not None:
        budget[0] += sum(len(t.encode()) for t in terms)
        if budget[0] > MAX_BYTES:
            raise VocabularyGraphUnavailable("aggregate negative vocabulary footprint limit")
    return terms


def endpoint_labels(entity):
    yield entity.label
    yield from entity.vocabulary_terms
    for alias in getattr(entity, "alias_evidence", ()):
        yield alias.alias
    for alias in getattr(entity, "aliases", ()):
        yield alias.alias
    for binding in getattr(entity, "acronym_expansions", ()):
        yield binding.acronym
        yield binding.full_form


def decorate_canonical(inputs, mention_rows, source_owners, vocabulary):
    if vocabulary is None:
        return inputs
    applicable = {t for _, pairs in vocabulary.scopes for pair in pairs for t in pair}
    found, by_entity = set(), {item.entity_id: set() for item in inputs}
    total_bytes = 0
    for row in mention_rows:
        found.add(row.document_entity_id)
        owner = source_owners[row.document_entity_id]
        term = normalized(row.mention.raw_text)
        if term in applicable and term not in by_entity[owner]:
            total_bytes += len(term.encode())
            if total_bytes > MAX_BYTES:
                raise VocabularyGraphUnavailable("aggregate negative vocabulary footprint limit")
            by_entity[owner].add(term)
    if found != set(source_owners) or set(source_owners.values()) != set(by_entity):
        raise VocabularyGraphUnavailable("incomplete negative vocabulary memberships")
    return tuple(replace(item, vocabulary_terms=tuple(sorted(by_entity[item.entity_id]))) for item in inputs)


def input_payload(value):
    if not value.get("vocabulary_terms"):
        value.pop("vocabulary_terms", None)
    return value
