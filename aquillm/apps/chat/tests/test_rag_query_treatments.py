"""Independent synthetic query arms retain real authority, scores and sinks."""
import json
from hashlib import sha256

import pytest

from apps.chat.services.rag_config import rag_preservation_config
from apps.collections.services.vocabulary_contracts import digest
from apps.ingestion.tests.output_support import upload as upload
from lib.embeddings.tests.test_document_provider import transport as transport

from .rag_query_treatment_support import (
    EXPANSION, PRIVATE, QUESTION, REWRITE, controls,
    query_corpus as query_corpus, close_query_worker as close_query_worker,
)
from .rag_query_treatment_transport import QueryProvider, selected_rows, score_transport as score_transport
from .rag_query_treatment_observation import install_observers, new_consumer, run

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("followup", [False, True])
async def test_independent_query_arms_share_corpus_but_not_turn_authority(
    query_corpus, score_transport, monkeypatch, followup, tmp_path,
):
    corpus, runs = query_corpus, []
    for arm, kinds in (("original", ["original"]), ("vocabulary", ["original", "vocabulary"]),
                       ("rewrite", ["original", "rewrite"])):
        with monkeypatch.context() as scoped:
            controls(scoped, arm, followup)
            config = rag_preservation_config()
            assert config.error is None and config.active is followup
            assert config.evidence_text_mode == ("source" if followup else "legacy")
            assert config.rerank_text_mode == config.document_capacity_mode == "legacy"
            assert not config.iterative_retrieval_enabled
            audit = install_observers(scoped)
            provider = QueryProvider(REWRITE)
            result = await run(corpus, provider, new_consumer(corpus),
                               repeat_preparation=arm != "original")
        runs.append(result)
        expected = [QUESTION]
        if arm == "vocabulary":
            expected.append(QUESTION + ' "' + EXPANSION + '"')
        elif arm == "rewrite":
            expected.append(REWRITE)
        assert [row[0] for row in audit.searches] == expected
        assert all(row[1] is result.runtime and row[2] is result.turn.budget
                   and row[3] is result.runtime.authorization for row in audit.searches)
        assert result.runtime.budget.actions_used == len(expected)
        assert result.runtime.budget.planner_calls == int(arm == "rewrite")
        assert len(provider.plans) == int(arm == "rewrite")
        assert len(provider.answers) == 1
        assert audit.dense and all(len(rows) == 2 for _, rows in audit.dense)
        assert audit.scores and all(scored.score_set.status == "complete" for _, _, scored in audit.scores)
        assert all(ranking.rerank_status == "complete" for _, _, ranking in audit.rankings)
        assert audit.packets and all(packet.query == QUESTION for packet in audit.packets)
        assert result.answer._validated_answer.outcome == "partial"
        assert "only below 1 Pa" in result.stored
        proposals = [e for e in result.trace.events if e["event"] == "query_proposals"]
        if arm == "original":
            assert not proposals and not getattr(result.runtime, "_query_vocabulary", None)
        else:
            rows = audit.prepared[0][2]
            assert len(audit.validations) >= len(expected)
            assert all(bound.runtime is result.runtime and budget is result.turn.budget
                       for bound, budget, _ in audit.validations)
            assert [row.kind for row in rows] == kinds
            assert len(proposals) == 1
            assert proposals[0]["query_digests"] == [digest(row.text) for row in rows]
            assert proposals[0]["observed_model"] == proposals[0]["response_revision"] == "unknown"
            assert proposals[0]["observed_policy"] == rows[0].policy_key
            if arm == "vocabulary":
                snapshot = result.runtime._query_vocabulary
                assert snapshot.version_keys == (corpus.version.version_key,)
                assert snapshot.binding.selected_collection_ids == (corpus.collection.pk,)
                assert rows[1].source_keys == (corpus.version.version_key + ":entry-1",)
                assert proposals[0]["vocabulary_digest"] == snapshot.checksum
        runtime, lease, packet, payload = result.dispatches[-1]
        assert runtime is result.runtime and lease.budget is result.turn.budget
        assert packet.query == QUESTION and packet.source_authorization is runtime.authorization
        selected = selected_rows(payload)
        assert selected == json.loads(lease.payload)["result"]
        assert packet._publication_representation is result.frozen
        by_id = {chunk.pk: chunk for chunk in corpus.chunks}
        for row in selected:
            original = by_id[row["chunk_id"]]
            assert row["text"].encode() == original.content.encode()
            assert row["citation"] == f"[doc:{original.doc_id} chunk:{original.pk}]"
            assert any(score.chunk_pk == original.pk and score.source_fingerprint ==
                sha256(original.content.encode()).hexdigest()
                for _, _, scored in audit.scores for score in scored.score_set.scores)
        serialized = json.dumps({"events": result.trace.events, "sdk": provider.requests})
        assert PRIVATE not in serialized and corpus.hidden.version_key not in serialized
        (tmp_path / (arm + ".json")).write_text(serialized, encoding="utf-8")
    assert len({id(result.turn.budget) for result in runs}) == 3
    assert len({id(result.runtime.authorization) for result in runs}) == 3


@pytest.mark.asyncio
async def test_three_real_queries_exhaust_original_action_budget_without_fourth_planner(
    query_corpus, score_transport, monkeypatch,
):
    controls(monkeypatch, "interaction", True, iterative=True)
    audit = install_observers(monkeypatch)
    provider = QueryProvider(REWRITE)
    result = await run(query_corpus, provider, new_consumer(query_corpus), repeat_preparation=True)
    assert [row[0] for row in audit.searches] == [QUESTION, QUESTION + ' "' + EXPANSION + '"', REWRITE]
    assert result.runtime.budget.actions_used == result.runtime.budget.limits.actions == 3
    assert result.runtime.budget.planner_calls == len(provider.plans) == 1
    assert len(provider.answers) == 1
    assert len(audit.acquisitions) == 1 and audit.acquisitions[0].rounds == 3
    starts = [e for e in result.trace.events if e["event"] == "action_start"]
    assert len(starts) == 3 and {e["ledger_id"] for e in starts} == {str(id(result.turn.budget))}
