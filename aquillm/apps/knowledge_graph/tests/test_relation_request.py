"""Relation cues cannot disappear, invent predicates, or force ambiguous intent."""

from dataclasses import replace

import pytest

from apps.knowledge_graph.tests.relation_v2_fixtures import api

ALLOWED = frozenset({"uses_dataset", "supports", "contradicts"})


def parse(query, allowed=ALLOWED):
    return api("relation_request").parse_relation_request_v2(
        query, ontology_checksum="a" * 64, allowed_relations=allowed
    )


def test_same_entity_questions_keep_distinct_predicates():
    a, b = (
        parse("Which dataset does Atlas use?"),
        parse("Which finding contradicts Atlas?"),
    )
    assert dict(a.predicate_weights) == {
        "uses_dataset": 1.0,
        "supports": 0.2,
        "contradicts": 0.2,
    }
    assert dict(b.predicate_weights) == {
        "uses_dataset": 0.2,
        "supports": 0.2,
        "contradicts": 1.0,
    }
    digest = api("relation_request").relation_request_digest
    assert digest(a) != digest(b)


@pytest.mark.parametrize(
    "query, preferred",
    [
        ("What supports Atlas?", "supports"),
        ("Which evidence does not support Atlas?", "contradicts"),
        ("Which evidence is not supporting Atlas?", "contradicts"),
        ("Which finding contradicts Atlas?", "contradicts"),
    ],
)
def test_requested_and_negated_support(query, preferred):
    weights = dict(parse(query).predicate_weights)
    assert weights[preferred] == 1.0
    assert sum(w == 1.0 for w in weights.values()) == 1


@pytest.mark.parametrize(
    "query",
    [
        "Tell me about Atlas",
        "What supports or contradicts Atlas?",
        "Which dataset supports Atlas?",
        "What does not contradict Atlas?",
        "What evidence is not contradicting Atlas?",
    ],
)
def test_unknown_or_ambiguous_intent_preserves_broad_recall(query):
    assert set(dict(parse(query).predicate_weights).values()) == {1.0}


def test_casing_and_unicode_are_deterministic_without_source_text():
    first = parse("ＷＨＡＴ supports ATLAS?")
    second = parse("what SUPPORTS atlas?")
    assert first == second
    assert "ATLAS" not in repr(first)
    assert "query" not in first.__slots__


def test_cue_cannot_invent_an_unapproved_predicate():
    request = parse("What contradicts Atlas?", frozenset({"supports"}))
    assert request.predicate_weights == (("supports", 1.0),)


@pytest.mark.parametrize("query", ["x" * 2049, "界" * 1366, "\ud800", 1])
def test_oversized_and_invalid_queries_are_refused(query):
    with pytest.raises((TypeError, ValueError)):
        parse(query)


def test_request_digest_binds_ontology_and_rejects_invalid_weights():
    request = parse("What supports Atlas?")
    digest = api("relation_request").relation_request_digest
    assert digest(request) != digest(replace(request, ontology_checksum="b" * 64))
    with pytest.raises((TypeError, ValueError)):
        replace(request, predicate_weights=(("supports", float("nan")),))
    with pytest.raises((TypeError, ValueError)):
        parse("Atlas", {"supports"})
