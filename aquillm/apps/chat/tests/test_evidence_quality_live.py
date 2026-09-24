"""Live runner route tested with fake external boundaries, never live quality claims."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from channels.db import database_sync_to_async as sync_to_async

from apps.chat.evals.evidence_quality_eval import load_cases
from apps.chat.evals.evidence_quality_live import mode_environment, run_case
from apps.chat.evals.evidence_quality_seed import seed_cases


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize(
    "mode",
    [
        "baseline",
        "selection",
        "preservation",
        "combined",
        "verified_windows",
        "cancellation",
    ],
)
async def test_actual_asgi_seed_retrieval_and_sdk_observation(
    tmp_path, monkeypatch, settings, mode
):
    from apps.chat.consumers.chat import ChatConsumer
    from aquillm import utils
    from lib.llm.providers.openai import OpenAIInterface

    settings.CHANNEL_LAYERS = {
        "default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}
    }
    monkeypatch.setattr(utils, "get_embedding", lambda *a, **k: [0.1] * 1024)
    monkeypatch.setenv("APP_RERANK_PROVIDER", "none")
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "16000")
    monkeypatch.setenv("RAG_DIRECT_MAX_QUERIES", "1")
    monkeypatch.setenv("MEM0_ENABLED", "0")
    case = load_cases(Path(__file__).parents[1] / "evals/evidence_quality_cases.json")[
        2
    ]
    manifest = await sync_to_async(seed_cases)([case], tmp_path / "seed.json")
    sent = []

    async def create(**kwargs):
        sent.append(kwargs)
        if mode == "cancellation":
            import asyncio

            await asyncio.Event().wait()
        row = manifest["cases"][case["case_id"]]["sources"][0]
        answer = (
            f"The study enrolled 25 sites [doc:{row['document_id']} "
            f"chunk:{row['chunk_id']}]."
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=answer, tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )

    interface = OpenAIInterface(
        SimpleNamespace(
            base_url="http://local.test",
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        ),
        "test",
    )
    monkeypatch.setattr(ChatConsumer, "llm_if", interface)
    # Candidate search remains actual PostgreSQL over seeded chunks; only the
    # external query embedding computation is replaced for this local test.
    verified = mode == "verified_windows"
    if verified:
        mode = "combined"
        monkeypatch.setenv("RAG_DIRECT_TOP_K", "1")
        from time import monotonic

        import requests

        from apps.documents.services import chunk_rerank_pair_capability as capability
        from apps.documents.services.pair_worker_lifecycle import canonical_provider

        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return " ".join(m["content"] for m in messages)

            def encode(self, text, **kwargs):
                return list(text.encode())

        monkeypatch.setenv("APP_RERANK_PROVIDER", "local")
        provider = canonical_provider(deadline=monotonic() + 60)
        counter = capability.VerifiedPairCounter(
            Tokenizer(),
            "template",
            "identity",
            "tokenizer",
            "template",
            monotonic() + 60,
        )
        monkeypatch.setitem(
            capability._registered, provider.scorer_fingerprint, counter
        )

        def score(url, *, json, **kwargs):
            return SimpleNamespace(
                status_code=200,
                json=lambda: {
                    "data": [{"index": 0, "score": 0.9}],
                    "usage": {"prompt_tokens": counter(json["text_1"], json["text_2"])},
                },
            )

        monkeypatch.setattr(requests, "post", score)
    if mode == "cancellation":
        case = {**case, "scenario": "cancellation"}
    with mode_environment("combined" if mode == "cancellation" else mode):
        with monkeypatch.context() as turn_environment:
            if not verified:
                turn_environment.setenv("RAG_RERANK_TEXT_MODE", "legacy")
            turn_environment.setenv("RAG_ITERATIVE_RETRIEVAL_ENABLED", "0")
            result = await run_case(case, manifest, mode=mode, cache_state="cold")
    if mode == "cancellation":
        assert result["closed"] and result["late_publications"] == 0
        assert result["answer"] == "" and result["published_after_disconnect"] == []
        assert result["delivered"]  # the SDK saw evidence before disconnect
        assert any(
            e["event"] == "ledger" and e["after"]["terminal"] == "cancelled"
            for e in result["events"]
        )
        return
    assert sent, result
    assert result["sdk_payloads"]
    assert result["delivered"], result
    assert (
        result["timings_ms"]["completion"] >= result["timings_ms"]["first_grounded"] > 0
    )
    if mode in ("preservation", "combined"):
        from apps.chat.evals.evidence_quality_safety import actual_safety

        result["dispatch_accounting_complete"] = True
        assert actual_safety(result)["passed"], actual_safety(result)
        assert result["actions"] == 1
        assert len(result["delivered"]) == (1 if verified else 5)
        if verified:
            assert any(e["event"] == "window_preparation" for e in result["events"])
            assert (
                len([e for e in result["events"] if e["event"] == "synthesis_sealed"])
                == 1
            )
            assert 0 < result["pairs"]["acquisition"] <= 5
            assert result["pairs"]["final"] == 0
    else:
        assert "actions" not in result
        assert len(result["delivered"]) == 3
