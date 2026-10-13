"""Contextual graph identity regression controls extracted from legacy suites."""
import pytest

from apps.knowledge_graph.resolution.coreference import resolve_document_mentions
from test_coreference import _mention, _ontology


@pytest.mark.parametrize("identity_policy", ["legacy", "contextual_v2"])
@pytest.mark.parametrize("conflict", [None, "identifier", "type", "version"])
def test_defined_acronym_repeated_full_forms_keep_sparse_partition(
    monkeypatch,
    identity_policy,
    conflict,
):
    from apps.knowledge_graph.resolution import coreference

    full = "Retrieval Augmented Generation"
    first = full + " v1" if conflict == "version" else full
    second = full + " v2" if conflict == "version" else full
    acronym = "RAGV" if conflict == "version" else "RAG"
    text = f"{first} ({acronym}) works. {second} improves grounding."
    mentions = (
        _mention(
            "full1",
            first,
            start=0,
            source_text=text,
            identifier="doi:10.1000/a" if conflict == "identifier" else "",
        ),
        _mention("acronym", acronym, start=text.index(acronym), source_text=text),
        _mention(
            "full2",
            second,
            start=text.rindex(second),
            source_text=text,
            entity_type="dataset" if conflict == "type" else "method",
            identifier="doi:10.1000/b" if conflict == "identifier" else "",
        ),
    )
    monkeypatch.setattr(coreference, "_EXHAUSTIVE_PAIR_LIMIT", len(mentions))
    exhaustive = resolve_document_mentions(
        mentions,
        _ontology(),
        identity_policy=identity_policy,
    )
    monkeypatch.setattr(coreference, "_EXHAUSTIVE_PAIR_LIMIT", 1)
    sparse = resolve_document_mentions(
        mentions,
        _ontology(),
        identity_policy=identity_policy,
    )
    assert {c.mention_ids for c in sparse.clusters} == {
        c.mention_ids for c in exhaustive.clusters
    }
    if identity_policy == "contextual_v2":
        assert sparse.clusters == exhaustive.clusters
    assert sparse.mention_ids == exhaustive.mention_ids
    if conflict is None:
        assert len(sparse.clusters) == 1
    else:
        assert not any(
            {"full1", "full2"} <= set(c.mention_ids) for c in sparse.clusters
        )
        assert any(not decision.accepted for decision in exhaustive.decisions)


def test_contextual_repeated_full_forms_use_bounded_acronym_connectivity(monkeypatch):
    from apps.knowledge_graph.resolution import coreference

    full = "Retrieval Augmented Generation"
    prefix = f"{full} (RAG)."
    repetitions = 128
    text = prefix + f" {full}." * (repetitions - 1)
    positions = (0,) + tuple(
        len(prefix) + 1 + index * (len(full) + 2) for index in range(repetitions - 1)
    )
    mentions = tuple(
        _mention(f"full-{index}", full, start=start, source_text=text)
        for index, start in enumerate(positions)
    ) + (_mention("acronym", "RAG", start=text.index("RAG"), source_text=text),)
    monkeypatch.setattr(coreference, "_EXHAUSTIVE_PAIR_LIMIT", 1)
    monkeypatch.setattr(coreference, "MAX_DOCUMENT_DECISIONS", 4 * len(mentions))
    result = resolve_document_mentions(
        mentions,
        _ontology(),
        identity_policy="contextual_v2",
    )
    assert len(result.clusters) == 1
    assert len(result.clusters[0].mention_ids) == len(mentions)


def test_contextual_sparse_partition_retains_homonym_singletons_and_anchors(
    monkeypatch,
):
    from apps.knowledge_graph.resolution import coreference

    mentions = (
        _mention("a", "Mercury", "model", start=0),
        _mention("b", "Mercury", "model", start=20),
        _mention("c", "Orion", "model", start=40, identifier="doi:10.1000/a"),
        _mention("d", "Orion", "model", start=60, identifier="doi:10.1000/a"),
    )
    exhaustive = resolve_document_mentions(
        mentions, _ontology(), identity_policy="contextual_v2"
    )
    monkeypatch.setattr(coreference, "_EXHAUSTIVE_PAIR_LIMIT", 1)
    sparse = resolve_document_mentions(
        mentions, _ontology(), identity_policy="contextual_v2"
    )
    assert sparse.clusters == exhaustive.clusters
    assert {cluster.mention_ids for cluster in sparse.clusters} == {
        ("a",),
        ("b",),
        ("c", "d"),
    }
    assert any(decision.outcome == "candidate" for decision in sparse.decisions)
    assert all(
        decision.accepted == (decision.outcome == "automatic")
        for decision in sparse.decisions
    )
