"""Producer unions cannot establish actual claim identity or evidence allowances."""

from apps.knowledge_graph.tests.graph_association_fixture_v2 import (
    AssociationFixture,
    key,
)


def test_incidental_mentions_cannot_multiply_one_transition_allowance():
    f = AssociationFixture(tuple(str(i) for i in range(4)))
    for i in range(4):
        f.add(str(i), relations=(("a", "supports", "b"),), mentions=("c" + str(i),))
    assert len({r.identity_keys for r in f.topology().evidence_candidates}) == 4
    result = f.select()
    assert len(result.candidates) == 3
    assert "evidence_budget" in result.coverage.reasons


def test_mixed_chunks_debit_real_mentions_alongside_different_relations():
    f = AssociationFixture(("one", "two", "three"))
    for i, label in enumerate(("one", "two", "three")):
        f.add(label, relations=(("a", "supports", "b" + str(i)),), mentions=("c",))
    result = f.select()
    assert len(result.candidates) == 2
    assert "evidence_budget" in result.coverage.reasons


def test_true_opposition_survives_different_incidental_identities():
    f = AssociationFixture(("intro", "support", "oppose"))
    f.add("intro", mentions=("a",), text="Show conflicting Atlas accuracy evidence")
    f.add(
        "support",
        relations=(("a", "supports", "b"),),
        mentions=("c",),
        text="Claim succeeds",
    )
    f.add(
        "oppose",
        relations=(("a", "contradicts", "b"),),
        mentions=("d",),
        text="Claim fails",
    )
    assert {r.chunk_key for r in f.select(2).candidates} == {
        key("support"),
        key("oppose"),
    }


def test_equal_identity_unions_cannot_manufacture_opposition():
    f = AssociationFixture(("answer", "support", "unrelated"))
    f.add("answer", mentions=("a",), text="Show conflicting Atlas accuracy evidence")
    f.add(
        "support",
        relations=(("a", "supports", "b"),),
        mentions=("c",),
        text="Claim succeeds",
    )
    f.add(
        "unrelated",
        relations=(("a", "contradicts", "c"),),
        mentions=("b",),
        text="Other claim fails",
    )
    refs = {r.chunk_key: r for r in f.topology().evidence_candidates}
    assert refs[key("support")].identity_keys == refs[key("unrelated")].identity_keys
    assert f.select(1).candidates[0].chunk_key == key("answer")


def test_every_actual_transition_in_a_multi_relation_chunk_consumes_allowance():
    f = AssociationFixture(("one", "two", "three", "four"), query="Atlas accuracy")
    for i, label in enumerate(("one", "two", "three", "four")):
        f.add(label, relations=(("a", "supports", "b"), ("a", "knows", "d" + str(i))))
    assert len(f.select().candidates) == 3
