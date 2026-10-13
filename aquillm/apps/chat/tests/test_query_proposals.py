"""Optional query proposals retain constraints and never replace the question."""

from importlib import import_module, util

import pytest


def proposal_module():
    name = "apps.chat.services.retrieval_query_proposals"
    assert util.find_spec(name), "faithful query proposal capability is missing"
    return import_module(name)


@pytest.mark.parametrize("original,rewrite", [
    ("Which models did NOT use dataset X in 2024?", "Which models used dataset X in 2024?"),
    ('Find "mass loss" in source A_12', 'Find mass loss in source A_12'),
    ("Explain A_12", "Explain A_13"),
    ("Compare results in 2024", "Compare results in 2025"),
    ("Measure 12 km from paper A", "Measure 12 m from paper A"),
    ("Explain the second paper", "Explain the paper"),
    ("Which models use X?", "Which models do not use X?"),
])
def test_rewrite_cannot_drop_or_reverse_protected_constraints(original, rewrite):
    rows = proposal_module().build_query_proposals(
        original, vocabulary=None, rewrite=rewrite, max_queries=3
    )
    assert [row.text for row in rows] == [original]


def test_unicode_vocabulary_is_original_first_bounded_and_provenanced():
    p = proposal_module()
    v = import_module("apps.collections.services.vocabulary_contracts")
    alias = v.VocabularyAliasV1("μ_test", tuple(f"phrase {i}" for i in range(6)), "curator", "entry-1")
    snapshot = v.VocabularySnapshotV1(("v1",), (alias,), (), "a" * 64)
    question = "What is μ_test?"
    rows = p.build_query_proposals(question, vocabulary=snapshot,
                                  rewrite="Explain μ_test?", max_queries=9)
    assert [row.kind for row in rows] == ["original", "vocabulary", "rewrite"]
    assert rows[0].text == question
    assert rows[1].source_keys == ("entry-1",)
    assert rows[1].text.count('"phrase ') == 6
    assert p.build_query_proposals(question, vocabulary=snapshot,
                                  rewrite=question, max_queries=1)[0].text == question
    assert len(p.build_query_proposals("What is other_μ_test?", vocabulary=snapshot,
                                      rewrite=None, max_queries=3)) == 1


def test_proposal_content_is_not_semantic_certification():
    p = proposal_module()
    rows = p.build_query_proposals("What is A_12?", vocabulary=None,
                                  rewrite="Explain A_12?", max_queries=3)
    assert [r.text for r in rows] == ["What is A_12?", "Explain A_12?"]
    assert all(r.policy_key for r in rows)
