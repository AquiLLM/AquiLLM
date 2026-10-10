"""Question relevance precedes bounded evidence allowance and confidence."""

from hashlib import sha256
from importlib import import_module, util

import pytest

from apps.knowledge_graph.retrieval.graph_handoff import GraphCoverageV2
from apps.knowledge_graph.retrieval.graph_query_contracts import (
    GraphEvidenceRefV2,
    GraphGroupRelationRequestV2,
)
from apps.knowledge_graph.retrieval.relation_request import parse_relation_request_v2

MODULE = "apps.knowledge_graph.retrieval.graph_evidence_selection_v2"


def key(value):
    return sha256(value.encode()).hexdigest()


def selector():
    assert util.find_spec(MODULE) is not None, "query-aware selector is missing"
    return import_module(MODULE)


class SelectionFixture:
    identity = key("atlas")
    key = staticmethod(key)

    def inputs(self, rows):
        module = selector()
        return tuple(
            module.GraphEvidenceSelectionInputV2(
                GraphEvidenceRefV2(
                    key(label),
                    key("generation"),
                    None,
                    key("group"),
                    (self.identity,),
                    (key(label + "-provenance"),),
                    relations,
                    confidence,
                ),
                text,
                key(text),
                key(document),
            )
            for label, text, confidence, relations, document in rows
        )

    def envelope(self, rows, query):
        module = selector()
        request = GraphGroupRelationRequestV2(
            key("group"),
            parse_relation_request_v2(
                query,
                ontology_checksum=key("ontology"),
                allowed_relations=frozenset(
                    {"uses_dataset", "supports", "contradicts"}
                ),
            ),
        )
        return module.GraphEvidenceSelectionEnvelopeV2(
            self.inputs(rows),
            GraphCoverageV2(3, 3, "complete", (), False),
            key("algorithm"),
            ((key("group"), key("ready")),),
        ), (request,)

    def dataset_question(self, introductions, answer):
        rows = [
            (label, "Atlas is a model", confidence, (), label)
            for label, confidence in introductions
        ]
        rows.append(
            (answer[0], "Atlas uses the Cedar dataset", answer[1], (), answer[0])
        )
        return self.envelope(rows, "Which dataset does Atlas use?")


@pytest.fixture
def selection_fixture():
    return SelectionFixture()


def select(fixture, envelope, requests, query, maximum=20):
    return selector().select_graph_evidence_v2(
        query=query,
        requests=requests,
        envelope=envelope,
        identity_scores=((fixture.identity, 1.0),),
        max_candidates=maximum,
        max_mentions_per_identity=2,
        max_evidence_per_transition=3,
    )


def test_lower_confidence_answer_survives_intro_mentions(selection_fixture):
    f = selection_fixture
    envelope, requests = f.dataset_question(
        introductions=(("intro-1", 0.99), ("intro-2", 0.98)), answer=("answer", 0.80)
    )
    result = select(f, envelope, requests, "Which dataset does Atlas use?")
    assert f.key("answer") in {r.chunk_key for r in result.candidates}
    assert len(result.candidates) == 2
    assert result.coverage.exhaustive is False
    assert result.coverage.reasons == ("evidence_budget",)


@pytest.mark.parametrize("maximum", [1, 2, 20])
def test_opposing_extracted_classes_precede_redundant_introductions(
    selection_fixture, maximum
):
    f = selection_fixture
    query = "Show evidence supporting and contradicting Atlas accuracy"
    envelope, requests = f.envelope(
        [
            ("intro", "Atlas accuracy evidence introduction", 1.0, (), "intro"),
            ("positive", "Atlas accuracy is high", 0.7, ("supports",), "paper-one"),
            ("negative", "Atlas accuracy is low", 0.6, ("contradicts",), "paper-two"),
        ],
        query,
    )
    result = select(f, envelope, requests, query, maximum)
    if maximum >= 2:
        assert {"supports", "contradicts"} <= {
            t for r in result.candidates for t in r.relation_types
        }
    else:
        assert result.coverage.status == "partial"
        assert "evidence_budget" in result.coverage.reasons


def test_duplicate_source_has_no_additional_ranking_credit(selection_fixture):
    from dataclasses import replace

    f = selection_fixture
    envelope, requests = f.dataset_question(introductions=(), answer=("answer", 0.8))
    once = select(f, envelope, requests, "Which dataset does Atlas use?")
    duplicated = replace(envelope, inputs=envelope.inputs * 2)
    assert select(f, duplicated, requests, "Which dataset does Atlas use?") == once


def test_request_group_and_query_are_exact(selection_fixture):
    from dataclasses import replace

    f = selection_fixture
    envelope, requests = f.dataset_question(introductions=(), answer=("answer", 0.8))
    for invalid in (
        (),
        requests * 2,
        (replace(requests[0], authority_group_checksum=key("wrong")),),
    ):
        with pytest.raises(ValueError):
            select(f, envelope, invalid, "Which dataset does Atlas use?")
    with pytest.raises(ValueError):
        select(f, envelope, requests, "What supports Atlas?")
