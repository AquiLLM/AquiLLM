"""Actual acquisition must retain current window scores and every source condition."""

import json
from copy import deepcopy
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
from lib.llm.providers.rag_citations import find_uncited_numeric_claims

HEADER = "Table 1. Signed results; concentration unit: ug/L."
CONTRADICTION = (
    "Correction: CERULEAN-OTTER was withdrawn; the signed result is unknown."
)


def reuse_case(variant):
    case = suffix_case()
    case["case_id"] += "-reuse-" + variant
    source = case["sources"][0]
    text = source["text"]
    if variant == "table_header":
        # A complete preceding paragraph is independently relevant context.
        prefix = text[:1600] + "\n\n" + HEADER + "\n\n"
        text = prefix.ljust(1701) + text[1701:]
    elif variant == "contradiction":
        text = text[:1900] + "\n\n" + CONTRADICTION
        text = text.ljust(2043) + " End."
    source["text"] = text
    assert len(text) == 2048 and text.index(MARKER) == 1701
    distractor = deepcopy(source)
    distractor.update(
        source_id="distractor",
        document_id="distractor-document",
        chunk_id="distractor",
        text="The signed result in this document is a draft. "
        "It has no approved measurement and does not resolve the question.",
    )
    case["sources"].append(distractor)
    return case


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize("variant", ["ordinary", "table_header", "contradiction"])
async def test_actual_acquisition_scores_reused_without_losing_source_conditions(
    tmp_path,
    monkeypatch,
    settings,
    transport,
    variant,
):
    configure(monkeypatch, settings)
    case = reuse_case(variant)
    source = case["sources"][0]["text"]
    manifest = await database_sync_to_async(seed_cases)([case], tmp_path / "seed.json")
    await database_sync_to_async(activate_synthetic_index)(manifest, case, transport)
    settings.DOCUMENT_EMBEDDING_SPACE_KEY = transport[0].space_key
    binding = manifest["cases"][case["case_id"]]["sources"][0]
    calls, counter = fake_pair_transport(
        monkeypatch, score_value=lambda text: 0.95 if MARKER in text else 0.1
    )
    observed = observe_originals(monkeypatch)
    statements = (MARKER,)
    if variant == "table_header":
        statements = ("Table 1.", "Signed results; concentration unit: ug/L.", MARKER)
    elif variant == "contradiction":
        statements = (f"Earlier finding: {MARKER}", CONTRADICTION)
    sdk, answer, citation = fake_answer(monkeypatch, binding, statements=statements)
    unattributed = []
    with mode_environment("combined"), transport_observation(unattributed):
        result = await run_case(case, manifest, mode="combined", cache_state="cold")

    assert observed.dense and len(observed.dense[0][1]) == 2, result
    assert observed.rankings
    args, _, ranking = observed.rankings[0]
    assert len(ranking.combined_candidates) == 2 > args[2] == 1
    assert ranking.score_set is not None and ranking.score_set.status == "complete"
    assert observed.phases == ["acquisition"]
    scorer, query, chunks, scores = observed.scores[0]
    assert scores is ranking.score_set
    assert query == case["question"]
    scored = next(row for row in scores.scores if row.chunk_pk == binding["chunk_id"])
    assert scored.source_fingerprint == sha256(source.encode()).hexdigest()
    assert str(scored.document_id) == binding["document_id"]
    assert scored.window_coverage.prepared_input_scoring_coverage == "complete"
    assert len(scored.window_coverage.required_window_ids) == 2
    assert scored.window_coverage.required_pair_fingerprints == (
        scored.window_coverage.successful_pair_fingerprints
    )
    assert len(calls) == 3 and any(MARKER in call["text_2"] for call in calls)
    assert all(counter(call["text_1"], call["text_2"]) <= 1024 for call in calls)
    assert all(any(call["text_2"] in row.content for row in chunks) for call in calls)
    assert all(call["text_1"] == case["question"] for call in calls)

    assert observed.prepared and sdk and observed.sealed and observed.dispatches
    preparation, prepared = observed.prepared[-1]
    assert prepared.score_status == "model"
    assert prepared.reused_pairs == 1 and prepared.new_pairs == 0
    runtime, turn, lease, packet, payload = observed.dispatches[-1]
    assert preparation["turn_budget"] is scorer.budget is runtime.budget
    assert runtime.budget is turn.budget is lease.budget
    assert preparation["authorization"] is runtime.authorization
    selected = packet.selection.candidates[0]
    assert selected.source_fingerprint == scored.source_fingerprint
    assert selected.scorer_fingerprint == scores.scorer_fingerprint
    assert selected.score_interpretation == "uncalibrated"
    assert packet.chunks[0]["text"].encode() == source.encode()
    assert packet.chunks[0]["citation"] == citation
    assert json.loads(lease.payload)["result"][0]["text"].encode() == source.encode()
    assert payload["messages"][-1]["content"].endswith(lease.payload)
    assert result["delivered"][0]["text"].encode() == source.encode()
    assert result["delivered"][0]["fingerprint"] == scored.source_fingerprint
    assert publication_matches(result), result
    assert result["persisted_answer"] == result["final_stream"] == answer
    assert citation in result["persisted_answer"]
    assert not find_uncited_numeric_claims(result["persisted_answer"])
    # The model synthesizes the finding and its conditions; the exact whole
    # source is checked independently through scoring, sealing and SDK delivery.
    synthesis = result["persisted_answer"].replace(" " + citation, "")
    assert MARKER in synthesis
    if variant != "ordinary":
        condition = HEADER if variant == "table_header" else CONTRADICTION
        assert condition in source and any(
            condition in call["text_2"] for call in calls
        )
        assert condition in lease.payload and condition in " ".join(synthesis.split())
    assert result["pairs"] == {"acquisition": 3, "final": 0}
    assert result["actions"] == 1 and not unattributed
    assert result["dispatch_accounting_complete"]
