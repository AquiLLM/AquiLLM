"""Synthetic external transport; real authorization, scoring and publication."""

import json
from hashlib import sha256
from time import monotonic
from types import SimpleNamespace

import requests

from apps.chat.evals.evidence_quality_trace import Trace
from apps.documents.services import chunk_rerank_pair_capability as capability
from apps.documents.services.chunk_rerank_window_adapter import WindowSelectionScorer
from apps.documents.services.pair_worker_lifecycle import canonical_provider
from lib.retrieval import TurnBudget, TurnLimits

MARKER = "The signed result is CERULEAN-OTTER at 7.3 ug/L."
QUESTION = "What is the signed result in the selected document?"


def suffix_case():
    prefix = "Draft: the result is AMBER-FOX at 99 ug/L; not a signed finding.\n"
    text = (prefix + "Background context. " * 100)[:1700] + "\n" + MARKER
    text = text.ljust(2043, " ") + " End."
    source = {
        "source_id": "suffix-source",
        "revision": "synthetic-revision-1",
        "text": text,
        "document_id": "suffix-document",
        "chunk_id": "suffix-source",
        "chunk_number": 0,
        "display_order": None,
        "authorized_at_answer": True,
        "origin": "synthetic",
    }
    return {
        "case_id": "r5-suffix-2048",
        "question": QUESTION,
        "turns": [],
        "split": "development",
        "genre": "research_paper",
        "depth": "routine",
        "scenario": "tail",
        "sources": [source],
        "quality_aggregation": "normal",
        "gold_support": [
            {
                "support_id": "signed",
                "source_id": "suffix-source",
                "revision": source["revision"],
                "start": 1701,
                "end": 1701 + len(MARKER),
                "quote": MARKER,
            }
        ],
        "required_claims": [
            {"claim_id": "result", "support_ids": ["signed"], "labels": {}}
        ],
    }


class SyntheticTokenizer:
    """Deterministic two-byte tokenizer, never production-model attestation."""

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(message["content"] for message in messages)

    def encode(self, text, **kwargs):
        raw = text.encode("utf-8")
        return [int.from_bytes(raw[i : i + 2], "big") for i in range(0, len(raw), 2)]


def fake_pair_transport(monkeypatch, *, disagreement=False, score_value=None):
    tokenizer, calls = SyntheticTokenizer(), []
    serving = [False]

    def send(session, request, **kwargs):
        payload = json.loads(request.body)
        response = requests.Response()
        response.status_code = 200
        response.url = request.url
        if request.url.endswith("/tokenize"):
            ids = tokenizer.encode(payload["prompt"])
            body = {"tokens": ids, "count": len(ids), "max_model_len": 1024}
        else:
            assert request.url.endswith("/score"), request.url
            assert payload.get("truncate_prompt_tokens") is None
            assert payload["max_tokens_per_query"] == 0
            assert payload["max_tokens_per_doc"] == 0
            calls.append(payload)
            pair = payload["text_1"] + "\n" + payload["text_2"]
            count = len(tokenizer.encode(pair))
            if count > 1024:
                response.status_code = 400
                body = {
                    "error": {
                        "type": "BadRequestError",
                        "code": 400,
                        "param": "input_tokens",
                        "message": "1024 context token limit",
                    }
                }
            else:
                body = {
                    "data": [
                        {
                            "index": 0,
                            "score": (
                                score_value(payload["text_2"]) if score_value else 0.9
                            ),
                        }
                    ],
                    "usage": {
                        "prompt_tokens": count + int(disagreement and serving[0])
                    },
                }
        response._content = json.dumps(body).encode()
        response.request = request
        return response

    monkeypatch.setattr(requests.sessions.Session, "send", send)
    provider = canonical_provider(deadline=monotonic() + 60)
    identity = {
        "runtime_digest": "sha256:" + "a" * 64,
        "model_revision": "b" * 40,
        "tokenizer_revision": "c" * 40,
        "code_revision": "d" * 40,
        "template_sha256": sha256(b"synthetic-template").hexdigest(),
    }
    counter = capability.verify_pair_capability(
        provider,
        tokenizer=tokenizer,
        template="synthetic-template",
        identity=identity,
        budget=TurnBudget(TurnLimits()),
    )
    assert counter is not None
    calls.clear()
    serving[0] = True
    return calls, counter


def observe_originals(monkeypatch):
    from apps.chat.services import rag_selection_coordinator
    from apps.documents.services import (
        chunk_search,
        chunk_search_scope,
        embedding_index_query,
    )

    captured = SimpleNamespace(
        scores=[],
        dispatches=[],
        sealed=[],
        spaces=[],
        dense=[],
        rankings=[],
        phases=[],
        prepared=[],
    )
    original_score, original_sink = WindowSelectionScorer.score_windows, Trace.sink
    original_space = chunk_search_scope.resolve_query_space
    original_dense = embedding_index_query.rank_compatible_chunks
    original_rank = chunk_search.materialize_and_rerank_candidates
    original_prepare = rag_selection_coordinator.prepare_selection_candidates

    def prepare(**kwargs):
        result = original_prepare(**kwargs)
        captured.prepared.append((kwargs, result))
        return result

    def rank(*args, **kwargs):
        result = original_rank(*args, **kwargs)
        captured.rankings.append((args, kwargs, result))
        return result

    def rank_dense(**kwargs):
        result = original_dense(**kwargs)
        captured.dense.append((kwargs, result))
        return result

    def resolve_space(using):
        try:
            result = original_space(using)
        except Exception as exc:
            captured.spaces.append(type(exc).__name__)
            raise
        captured.spaces.append(result)
        return result

    def score(self, query, chunks, **kwargs):
        chunks = tuple(chunks)
        result = original_score(self, query, chunks, **kwargs)
        captured.scores.append((self, query, chunks, result))
        captured.phases.append(kwargs.get("phase", "final"))
        return result

    def sink(self, event, data):
        original_sink(self, event, data)
        if event in ("sdk_start", "synthesis_sealed"):
            from apps.documents.services.source_loading import current_source_runtime
            from lib.llm.turn_context import current_turn

            runtime, turn = current_source_runtime(), current_turn()
            assert runtime is not None and turn is not None
            assert runtime.budget is turn.budget
            record = (runtime, turn, runtime.budget._synthesis)
            if event == "synthesis_sealed":
                captured.sealed.append(record)
            else:
                packet = runtime.observation.get("delivered_packet")
                captured.dispatches.append((*record, packet, data["payload"]))

    monkeypatch.setattr(WindowSelectionScorer, "score_windows", score)
    monkeypatch.setattr(Trace, "sink", sink)
    monkeypatch.setattr(chunk_search_scope, "resolve_query_space", resolve_space)
    monkeypatch.setattr(embedding_index_query, "rank_compatible_chunks", rank_dense)
    monkeypatch.setattr(chunk_search, "materialize_and_rerank_candidates", rank)
    monkeypatch.setattr(
        rag_selection_coordinator, "prepare_selection_candidates", prepare
    )
    return captured


def configure(monkeypatch, settings):
    from aquillm import utils

    settings.DEBUG = True
    settings.CHANNEL_LAYERS = {
        "default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}
    }
    monkeypatch.setattr(utils, "get_embedding", lambda *a, **k: [0.1] * 1024)
    for name, value in {
        "APP_RERANK_PROVIDER": "local",
        "OPENAI_CONTEXT_LIMIT": "16000",
        "RAG_DIRECT_MAX_QUERIES": "1",
        "RAG_DIRECT_TOP_K": "1",
        "MEM0_ENABLED": "0",
        "OPENAI_STREAM_RESPONSES": "0",
    }.items():
        monkeypatch.setenv(name, value)


def fake_answer(monkeypatch, binding):
    from apps.chat.consumers.chat import ChatConsumer
    from lib.llm.providers.openai import OpenAIInterface

    calls = []
    citation = f"[doc:{binding['document_id']} chunk:{binding['chunk_id']}]"
    answer = f"{MARKER[:-1]} {citation}."

    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=answer, tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )

    provider = OpenAIInterface(
        SimpleNamespace(
            base_url="http://local.test",
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        ),
        "test",
    )
    monkeypatch.setattr(ChatConsumer, "llm_if", provider)
    return calls, answer, citation


def activate_synthetic_index(manifest, case, transport):
    """Actual producer receipt and generation, on the real legacy-null source."""
    from apps.documents.models import RawTextDocument, TextChunk
    from apps.documents.services import embedding_index_store as store
    from apps.documents.services.representation_contracts import DocumentRefV1
    from apps.documents.tests.test_embedding_generation import stage

    rows = manifest["cases"][case["case_id"]]["sources"]
    for document_id in dict.fromkeys(row["document_id"] for row in rows):
        doc = RawTextDocument.objects.get(id=document_id)
        assert (
            not TextChunk.objects.filter(doc_id=doc.id)
            .exclude(representation_generation=None)
            .exists()
        )
        ref = DocumentRefV1(
            doc._meta.label_lower,
            doc.pkid,
            doc.id,
            doc.collection_id,
            doc.full_text_hash,
        )
        _, lease, _, _ = stage((doc, ref), transport)
        assert (
            store.activate_embedding_generation(lease, using="default") == "activated"
        )
