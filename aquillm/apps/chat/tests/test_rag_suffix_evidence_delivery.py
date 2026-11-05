"""Catch suffix clipping, stale score reuse and changes between SDK and sinks."""

import json
from hashlib import sha256

import pytest
from channels.db import database_sync_to_async

from apps.chat.evals.evidence_quality_live import mode_environment, run_case
from apps.chat.evals.evidence_quality_seed import seed_cases
from apps.chat.evals.evidence_quality_trace import transport_observation
from apps.chat.evals.evidence_support_metrics import publication_matches
from apps.chat.tests.rag_suffix_test_support import (
    MARKER,
    activate_synthetic_index,
    configure,
    fake_answer,
    fake_pair_transport,
    observe_originals,
    suffix_case,
)
from lib.embeddings.tests.test_document_provider import transport as transport


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize("candidate_route", ["lexical_only", "compatible_dense"])
async def test_suffix_after_1600_survives_scoring_seal_sdk_and_persisted_sink(
    tmp_path,
    monkeypatch,
    settings,
    transport,
    candidate_route,
    capsys,
    caplog,
):
    configure(monkeypatch, settings)
    case = suffix_case()
    text = case["sources"][0]["text"]
    assert len(text) == 2048 and text.index(MARKER) == 1701
    manifest = await database_sync_to_async(seed_cases)([case], tmp_path / "seed.json")
    if candidate_route == "compatible_dense":
        await database_sync_to_async(activate_synthetic_index)(
            manifest, case, transport
        )
        settings.DOCUMENT_EMBEDDING_SPACE_KEY = transport[0].space_key
    binding = manifest["cases"][case["case_id"]]["sources"][0]
    calls, counter = fake_pair_transport(monkeypatch)
    observed = observe_originals(monkeypatch)
    sdk, answer, citation = fake_answer(monkeypatch, binding)
    unattributed = []
    with mode_environment("combined"), transport_observation(unattributed):
        result = await run_case(case, manifest, mode="combined", cache_state="cold")

    assert calls and any(MARKER in call["text_2"] for call in calls), result
    assert all(call["text_1"] == case["question"] for call in calls)
    assert all(counter(call["text_1"], call["text_2"]) <= 1024 for call in calls)
    assert all(call["text_2"] in text for call in calls)
    complete = [entry for entry in observed.scores if entry[3].status == "complete"]
    assert complete, result
    scorer, query, chunks, score_set = complete[0]
    assert score_set.query_fingerprint == sha256(query.encode()).hexdigest()
    score = score_set.scores[0]
    assert score.source_fingerprint == sha256(text.encode()).hexdigest()
    assert score.chunk_pk == binding["chunk_id"]
    assert str(score.document_id) == binding["document_id"]
    assert score.window_coverage.prepared_input_scoring_coverage == "complete"
    assert len(score.window_coverage.required_window_ids) == 2
    assert score.window_coverage.required_pair_fingerprints == (
        score.window_coverage.successful_pair_fingerprints
    )
    assert observed.sealed and observed.dispatches and sdk
    runtime, turn, lease, packet, payload = observed.dispatches[-1]
    assert scorer.budget is runtime.budget is turn.budget is lease.budget
    assert observed.sealed[0][2] is lease
    assert packet.selection.score_status == "model"
    selected = packet.selection.candidates[0]
    assert selected.source_fingerprint == score.source_fingerprint
    assert selected.scorer_fingerprint == score_set.scorer_fingerprint
    assert selected.score_interpretation == "uncalibrated"
    assert packet.chunks[0]["citation"] == citation
    assert packet.chunks[0]["text"].encode() == text.encode()
    assert MARKER in lease.payload
    assert payload["messages"][-1]["content"].endswith(lease.payload)
    assert json.loads(lease.payload)["result"][0]["text"].encode() == text.encode()
    assert result["delivered"][0]["text"].encode() == text.encode()
    assert result["delivered"][0]["fingerprint"] == sha256(text.encode()).hexdigest()
    assert publication_matches(result), result
    assert result["persisted_answer"] == result["final_stream"] == answer
    assert citation in result["persisted_answer"]
    assert MARKER in result["persisted_answer"].replace(" " + citation, "")
    assert not unattributed and result["dispatch_accounting_complete"]
    assert result["actions"] == 1
    assert sum(result["pairs"].values()) == len(calls) == 2
    if candidate_route == "compatible_dense":
        assert observed.spaces and all(
            value == transport[0] for value in observed.spaces
        ), observed.spaces
        assert len(transport[1]) > 1, transport[1]
        assert observed.dense and observed.dense[0][1], observed.dense
        assert observed.rankings
        args, _, ranking = observed.rankings[0]
        assert len(ranking.combined_candidates) <= args[2] == 1
        assert ranking.score_set is None
    # A one-candidate pool does not need acquisition reranking. The final
    # window scorer still must score the complete current source in two pairs.
    assert result["pairs"] == {"acquisition": 0, "final": 2}
    assert settings.DEBUG
    assert "CERULEAN-OTTER" not in capsys.readouterr().out
    assert "CERULEAN-OTTER" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "expired",
        "changed_tokenizer",
        "changed_template",
        "provider_disagreement",
        "final_reserve",
    ],
)
async def test_incomplete_windows_use_declared_uncalibrated_fallback(
    tmp_path,
    monkeypatch,
    settings,
    failure,
):
    from apps.documents.services import chunk_rerank_pair_capability as capability

    configure(monkeypatch, settings)
    case = suffix_case()
    manifest = await database_sync_to_async(seed_cases)([case], tmp_path / "seed.json")
    binding = manifest["cases"][case["case_id"]]["sources"][0]
    calls, counter = fake_pair_transport(
        monkeypatch, disagreement=failure == "provider_disagreement"
    )
    if failure == "missing":
        capability.invalidate_pair_capability()
    if failure == "expired":
        monkeypatch.setattr(capability, "monotonic", lambda: counter.expires + 1)
    if failure == "changed_tokenizer":
        # A new configured revision cannot reuse the prior verified identity.
        monkeypatch.setenv("APP_RERANK_TOKENIZER_REVISION", "e" * 40)
    if failure == "changed_template":
        from time import monotonic

        from apps.documents.services.pair_worker_lifecycle import canonical_provider
        from lib.retrieval import TurnBudget, TurnLimits

        # Actual bootstrap verifier revokes the old counter before rejecting a
        # changed template that does not match the independently pinned digest.
        replacement = capability.verify_pair_capability(
            canonical_provider(deadline=monotonic() + 60),
            tokenizer=counter.tokenizer,
            template="unverified replacement template",
            identity={
                "runtime_digest": "sha256:" + "a" * 64,
                "model_revision": "b" * 40,
                "tokenizer_revision": "c" * 40,
                "code_revision": "d" * 40,
                "template_sha256": counter.template_identity,
            },
            budget=TurnBudget(TurnLimits()),
        )
        assert replacement is None and counter.revoked.is_set()
    if failure == "final_reserve":
        from apps.chat.services import rag_turn
        from lib.retrieval import TurnLimits

        monkeypatch.setattr(
            rag_turn, "TurnLimits", lambda **kw: TurnLimits(**kw, final_pairs=0)
        )
    observed = observe_originals(monkeypatch)
    fake_answer(monkeypatch, binding)
    with mode_environment("combined"), transport_observation([]):
        result = await run_case(case, manifest, mode="combined", cache_state="cold")
    if failure == "final_reserve":
        assert observed.scores == []  # Admission rejects all new scoring at zero cap.
    else:
        assert observed.scores
    assert all(item[3].status == "unavailable" for item in observed.scores)
    assert all(
        item[3].scoring_kind == "rank_only" and not item[3].scores
        for item in observed.scores
    )
    assert observed.dispatches, result
    runtime, turn, lease, packet, _ = observed.dispatches[-1]
    assert runtime.budget is turn.budget is lease.budget
    assert packet.selection.score_status == "rank_fallback"
    assert all(
        candidate.raw_score is None and candidate.score_interpretation == "unavailable"
        for candidate in packet.selection.candidates
    )
    assert MARKER in result["delivered"][0]["text"]
    assert publication_matches(result)
    assert len(calls) == (1 if failure == "provider_disagreement" else 0)
    assert sum(result["pairs"].values()) == len(calls)
