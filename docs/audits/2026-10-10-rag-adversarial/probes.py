"""Offline review probes against current production functions.

Run from repository root: rtk proxy python docs/audits/2026-10-10-rag-adversarial/probes.py
No database, provider, live server, or source-document writes are made.
The ORM double establishes ordering/limit behavior, not PostgreSQL query plans.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "aquillm"))

from django.conf import settings

settings.configure(
    SECRET_KEY="offline-review-only",
    INSTALLED_APPS=[],
    DATABASES={},
    RAG_CACHE_ENABLED=False,
)
import django

django.setup()

from apps.documents.services import chunk_rerank as rerank
from apps.documents.services import chunk_rerank_local_scored as local
from apps.documents.services import chunk_search_candidates as candidates
from apps.documents.services.chunk_rerank_parse import fallback_rerank


def row(pk, doc=1, text="Generic unrelated background."):
    return SimpleNamespace(
        pk=pk, id=pk, doc_id=UUID(int=doc), chunk_number=pk,
        content=text, modality="text", embedding=[1.0], distance=pk,
        Modality=SimpleNamespace(IMAGE="image"),
    )


class Rows:
    """Minimal in-memory ORM double: evaluates Q terms, ordering, then slicing."""

    def __init__(self, rows):
        self.rows = list(rows)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, key):
        return Rows(self.rows[key])

    def none(self):
        return Rows([])

    def filter_by_documents(self, docs):
        allowed = {d.id for d in docs}
        return Rows(r for r in self.rows if r.doc_id in allowed)

    def filter(self, *expressions, **kwargs):
        def evaluate(q, r):
            children = [
                evaluate(child, r) if not isinstance(child, tuple)
                else child[1].lower() in r.content.lower()
                for child in q.children
            ]
            return any(children) if q.connector == "OR" else all(children)

        result = self.rows
        for q in expressions:
            result = [r for r in result if evaluate(q, r)]
        for name, value in kwargs.items():
            if name == "similarity__gt":
                result = [r for r in result if 0.0 > value]
            elif name == "pk__in":
                result = [r for r in result if r.pk in value]
            else:
                result = [r for r in result if getattr(r, name) == value]
        return Rows(result)

    def exclude(self, **kwargs):
        return Rows(r for r in self.rows if r.embedding is not None)

    def defer(self, *fields):
        return self

    def annotate(self, **kwargs):
        return self

    def order_by(self, *fields):
        if fields and not isinstance(fields[0], str):
            expression = fields[0]
            if hasattr(expression, "cases"):
                order = {
                    case.condition.children[0][1]: case.result.value
                    for case in expression.cases
                }
                return Rows(sorted(self.rows, key=lambda r: order[r.pk]))
            return Rows(sorted(self.rows, key=lambda r: r.distance))
        if fields == ("-similarity",):
            return self
        return Rows(sorted(self.rows, key=lambda r: tuple(getattr(r, f) for f in fields)))


def model(rows):
    return SimpleNamespace(objects=Rows(rows), Modality=SimpleNamespace(TEXT="text"))


def snapshot(rows, query, embedding=None):
    return candidates.collect_hybrid_candidate_snapshot(
        model(rows), query, 10,
        [SimpleNamespace(id=UUID(int=n)) for n in sorted({r.doc_id.int for r in rows})],
        query_embedding=embedding,
        app_config_getter=lambda _: SimpleNamespace(vector_top_k=12, trigram_top_k=12),
    )


def exact_starvation():
    rows = [row(n, text="ATP unrelated background.") for n in range(1, 13)]
    rows.append(row(99, doc=2, text="BRCA1 ATP: the requested measured result is 42."))
    found = snapshot(rows, "What is the ATP result for BRCA1?")
    assert found.exact_terms == ("ATP", "BRCA1")
    assert found.exact_chunk_ids == tuple(range(1, 13))
    assert 99 not in found.exact_chunk_ids
    return {"exact_terms": found.exact_terms, "returned": found.exact_chunk_ids,
            "answer_chunk_99_excluded": True}


def fallback_drops_exact():
    rows = [row(n) for n in range(1, 13)]
    rows.append(row(99, doc=2, text="BRCA1 measured result is 42."))
    found = snapshot(rows, "What is the BRCA1 result?", embedding=[1.0])
    assert 99 in found.exact_chunk_ids
    ranked = list(fallback_rerank(model(rows), found.baseline_candidates, 10))
    assert [r.pk for r in ranked] == list(range(1, 11))
    return {"exact_answer_in_pool": True, "fallback_ids": [r.pk for r in ranked],
            "answer_chunk_99_excluded": True}


def reranker_failure_and_prefix():
    rows = [row(n, text="Background. " * 100 + "TAIL_ANSWER_42") for n in range(1, 25)]
    calls = []

    def unavailable(endpoint, *, json, **kwargs):
        calls.append((endpoint, json))
        return SimpleNamespace(status_code=503)

    from os import environ
    overrides = {
        "APP_RERANK_PROVIDER": "local", "APP_RERANK_MODEL": "Qwen/Qwen3-Reranker-4B",
        "APP_RERANK_DOC_CHAR_LIMIT": "900", "RAG_RERANK_TEXT_MODE": "legacy",
        "RAG_RERANK_SHADOW_SCORING_ENABLED": "0",
        "APP_RERANK_BASE_URL": "http://offline.invalid/v1", "APP_RERANK_MODEL_REVISION": "",
    }
    with patch.dict(environ, overrides), patch.object(local.requests, "post", unavailable), \
         patch.object(local.rag_cache, "get_cached_rerank_capability", return_value=None):
        result = rerank.rerank_chunks_scored(object, "What is the answer?", rows, 10)
    assert len(calls) == 63, len(calls)
    assert result.ranked_ids == tuple(range(1, 11))
    assert all("TAIL_ANSWER_42" not in json.dumps(payload) for _, payload in calls)
    scored_calls = len(calls)
    calls.clear()
    from apps.documents.services.chunk_rerank_local_vllm import rerank_via_local_vllm
    with patch.dict(environ, overrides), patch.object(local.requests, "post", unavailable), \
         patch.object(local.rag_cache, "get_cached_rerank_capability", return_value=None):
        legacy = rerank_via_local_vllm(model(rows), "What is the answer?", rows, 10)
    assert not list(legacy)
    assert len(calls) == 63, len(calls)
    assert all("TAIL_ANSWER_42" not in json.dumps(payload) for _, payload in calls)
    return {"candidate_count": 24, "http_attempts_on_503": len(calls),
            "scored_http_attempts_on_503": scored_calls,
            "legacy_rank_only_http_attempts_on_503": len(calls),
            "ranked_ids": result.ranked_ids, "score_status": result.score_set.status,
            "answer_after_character_900_sent": False}


def short_scientific_identifiers():
    query = "Compare CO and NO at pH 7"
    terms = candidates._salient_exact_terms(query)
    assert terms == []
    return {"query": query, "exact_terms": terms}


if __name__ == "__main__":
    output = {
        "exact_starvation": exact_starvation(),
        "fallback_drops_exact": fallback_drops_exact(),
        "reranker_failure_and_prefix": reranker_failure_and_prefix(),
        "short_scientific_identifiers": short_scientific_identifiers(),
    }
    print(json.dumps(output, indent=2))
