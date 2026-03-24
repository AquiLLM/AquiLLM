"""Salience scoring for context retention."""
from __future__ import annotations

from lib.llm.utils.context_salience import (
    build_salience_scores,
    entity_citation_boost,
    lexical_overlap_score,
)


def test_salience_ranks_semantically_relevant_turns_higher():
    msgs = [
        {"role": "user", "content": "pasta carbonara recipes olive oil garlic " * 25},
        {"role": "assistant", "content": "Here are cooking ideas."},
        {
            "role": "user",
            "content": "Photosynthesis in plants uses chloroplasts to capture light energy " * 12,
        },
        {"role": "user", "content": "How do chloroplasts relate to sugar production in leaves?"},
    ]
    scores = build_salience_scores(msgs)
    assert scores[2] > scores[0]


def test_salience_lexical_overlap_matches_query_terms():
    q = "mitochondria ATP synthesis"
    a = "The mitochondrion produces ATP through oxidative phosphorylation."
    b = "I like pizza delivery on Fridays."
    assert lexical_overlap_score(q, a) > lexical_overlap_score(q, b)


def test_salience_boosts_citations_entities_numbers():
    plain = "Some background text without special markers."
    rich = "Smith et al. [3] reported DOI:10.1234/example in 2019."
    assert entity_citation_boost(rich) > entity_citation_boost(plain)
