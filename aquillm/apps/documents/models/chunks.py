"""
TextChunk model for storing document chunks with embeddings.

Includes vector search, trigram similarity, and reranking capabilities.
"""
from __future__ import annotations

import logging
import requests
from os import getenv
from time import perf_counter
from typing import TYPE_CHECKING, Any, Callable, List, Optional

from django.apps import apps
from django.contrib.postgres.indexes import GinIndex
from django.contrib.postgres.search import TrigramSimilarity
from django.core.exceptions import ValidationError
from django.db import DatabaseError, models
from django.db.models import Case, When
from pgvector.django import HnswIndex, L2Distance, VectorField
from tenacity import retry, wait_exponential

if TYPE_CHECKING:
    from django.db.models.query import QuerySet

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        value = int((getenv(name) or str(default)).strip())
    except Exception:
        value = default
    return value if value > 0 else default


class TextChunkQuerySet(models.QuerySet):
    """Custom QuerySet for TextChunk with document filtering."""

    def filter_by_documents(self, docs_or_ids):
        ids = [
            getattr(doc_or_id, "id", doc_or_id)
            for doc_or_id in docs_or_ids
        ]
        return self.filter(doc_id__in=ids)


def _get_descended_from_document():
    """Lazy import to avoid circular dependency."""
    from .document_types import DESCENDED_FROM_DOCUMENT
    return DESCENDED_FROM_DOCUMENT


def doc_id_validator(id):
    descended_from_document = _get_descended_from_document()
    if sum([t.objects.filter(id=id).exists() for t in descended_from_document]) != 1:
        raise ValidationError("Invalid Document UUID -- either no such document or multiple")


class TextChunk(models.Model):
    """
    A chunk of text (or image caption) from a document with embedding for vector search.

    Supports both text and image modalities for multimodal RAG.
    """

    class Modality(models.TextChoices):
        TEXT = "text", "Text"
        IMAGE = "image", "Image"

    content = models.TextField()
    start_position = models.PositiveIntegerField()
    end_position = models.PositiveIntegerField()

    start_time = models.FloatField(null=True)
    chunk_number = models.PositiveIntegerField()
    modality = models.CharField(
        max_length=16, choices=Modality.choices, default=Modality.TEXT, db_index=True
    )
    metadata = models.JSONField(default=dict, blank=True)
    embedding = VectorField(dimensions=1024, blank=True, null=True)

    doc_id = models.UUIDField(editable=False, validators=[doc_id_validator])

    objects = TextChunkQuerySet.as_manager()

    class Meta:
        app_label = "apps_documents"
        db_table = "aquillm_textchunk"
        constraints = [
            models.UniqueConstraint(
                fields=["doc_id", "start_position", "end_position"],
                name="unique_chunk_position_per_document",
            ),
            models.UniqueConstraint(
                fields=["doc_id", "chunk_number"],
                name="uniqe_chunk_per_document",
            ),
        ]
        indexes = [
            models.Index(fields=["doc_id", "start_position", "end_position"]),
            models.Index(fields=["doc_id", "modality"], name="textchunk_doc_modality_idx"),
            HnswIndex(
                name="chunk_embedding_index",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_l2_ops"],
            ),
            GinIndex(
                name="textchunk_content_trgm_idx",
                fields=["content"],
                opclasses=["gin_trgm_ops"],
            ),
        ]
        ordering = ["doc_id", "chunk_number"]

    @property
    def document(self):
        """Get the parent document for this chunk."""
        descended_from_document = _get_descended_from_document()
        ret = None
        for t in descended_from_document:
            doc = t.objects.filter(id=self.doc_id).first()
            if doc:
                ret = doc
        if not ret:
            raise ValidationError(f"TextChunk {self.pk} is not associated with a document!")
        return ret

    @document.setter
    def document(self, doc):
        self.doc_id = doc.id

    def save(self, *args, **kwargs):
        if self.start_position >= self.end_position:
            raise ValueError("end_position must be greater than start_position")
        if not self.embedding:
            try:
                self.get_chunk_embedding()
            except Exception as exc:
                logger.warning(
                    "Chunk embedding failed (doc_id=%s chunk=%s); saving without embedding. Error: %s",
                    self.doc_id,
                    self.chunk_number,
                    exc,
                )

        super().save(*args, **kwargs)

    def _multimodal_caption(self) -> str:
        char_limit = _env_int("APP_RAG_IMAGE_CAPTION_CHAR_LIMIT", 800)
        text = (self.content or "").strip()
        if not text:
            text = "Image chunk"
        return text[:char_limit]

    def _image_data_url(self) -> str | None:
        if self.modality != self.Modality.IMAGE:
            return None
        try:
            doc = self.document
        except Exception:
            return None
        from apps.documents.services.image_payloads import doc_image_data_url

        return doc_image_data_url(doc)

    def _image_embedding_payloads(self) -> list[Any]:
        data_url = self._image_data_url()
        if not data_url:
            return []
        caption = self._multimodal_caption()
        return [
            [
                {"type": "input_text", "text": caption},
                {"type": "input_image", "image_url": data_url},
            ],
            [
                {"type": "input_text", "text": caption},
                {"type": "input_image", "image_url": {"url": data_url}},
            ],
            [
                {"type": "text", "text": caption},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
            [{"type": "input_image", "image_url": data_url}],
        ]

    def _rerank_document_payload(self) -> Any:
        if self.modality != self.Modality.IMAGE:
            return self.content
        data_url = self._image_data_url()
        if not data_url:
            return self.content
        caption = self._multimodal_caption()
        return [
            {"type": "text", "text": caption},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]

    @retry(wait=wait_exponential())
    def get_chunk_embedding(self, callback: Optional[Callable[[], None]] = None):
        from aquillm.utils import get_embedding, get_multimodal_embedding

        if self.modality == self.Modality.IMAGE:
            image_data_url = self._image_data_url()
            caption = self._multimodal_caption()
            if image_data_url:
                self.embedding = get_multimodal_embedding(
                    prompt=caption,
                    image_data_url=image_data_url,
                    input_type="search_document",
                )
            else:
                self.embedding = get_embedding(caption, input_type="search_document")
        else:
            self.embedding = get_embedding(self.content, input_type="search_document")
        if callback:
            callback()

    @classmethod
    def _fallback_rerank(cls, chunks, top_k: int):
        chunk_ids = [chunk.pk for chunk in list(chunks)[:top_k]]
        if not chunk_ids:
            return cls.objects.none()
        preserved = Case(*[When(pk=pk, then=pos) for pos, pk in enumerate(chunk_ids)])
        return cls.objects.filter(pk__in=chunk_ids).order_by(preserved)

    @classmethod
    def _rerank_base_url(cls) -> str:
        base_url = (
            getenv("APP_RERANK_BASE_URL")
            or getenv("VLLM_RERANK_BASE_URL")
            or getenv("VLLM_BASE_URL")
            or "http://vllm_rerank:8000/v1"
        ).rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        return base_url

    @classmethod
    def _rerank_api_key(cls) -> str:
        return (
            getenv("APP_RERANK_API_KEY")
            or getenv("VLLM_RERANK_API_KEY")
            or getenv("VLLM_API_KEY")
            or "EMPTY"
        )

    @classmethod
    def _rerank_model(cls) -> str:
        return (
            getenv("APP_RERANK_MODEL")
            or getenv("VLLM_RERANK_MODEL")
            or "Qwen/Qwen3-Reranker-4B"
        )

    @classmethod
    def _rerank_timeout_seconds(cls) -> int:
        try:
            timeout = int((getenv("APP_RERANK_TIMEOUT_SECONDS") or "3").strip())
        except Exception:
            timeout = 3
        return timeout if timeout > 0 else 3

    @classmethod
    def _ordered_queryset_from_ids(cls, ranked_ids: list[int]):
        if not ranked_ids:
            return cls.objects.none()
        preserved = Case(*[When(pk=pk, then=pos) for pos, pk in enumerate(ranked_ids)])
        return cls.objects.filter(pk__in=ranked_ids).order_by(preserved)

    @classmethod
    def _parse_rerank_results(cls, body, chunks_list) -> list[int]:
        results = []
        if isinstance(body, dict):
            if isinstance(body.get("results"), list):
                results = body.get("results", [])
            elif isinstance(body.get("data"), list):
                results = body.get("data", [])
        ranked_ids: list[int] = []
        seen: set[int] = set()
        for result in results:
            idx = result.get("index") if isinstance(result, dict) else None
            if not isinstance(idx, int) or idx < 0 or idx >= len(chunks_list):
                continue
            chunk_pk = chunks_list[idx].pk
            if chunk_pk in seen:
                continue
            seen.add(chunk_pk)
            ranked_ids.append(chunk_pk)
        return ranked_ids

    @classmethod
    def _parse_score_results(cls, body) -> list[tuple[int, float]]:
        pairs: list[tuple[int, float]] = []
        if not isinstance(body, dict):
            return pairs
        raw_items = None
        if isinstance(body.get("data"), list):
            raw_items = body.get("data")
        elif isinstance(body.get("results"), list):
            raw_items = body.get("results")
        if not raw_items:
            return pairs
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            idx = item.get("index")
            if not isinstance(idx, int):
                continue
            score = item.get("score")
            if not isinstance(score, (int, float)):
                score = item.get("relevance_score")
            if not isinstance(score, (int, float)):
                continue
            pairs.append((idx, float(score)))
        return pairs

    @classmethod
    def _parse_single_score(cls, body) -> float:
        if isinstance(body, (int, float)):
            return float(body)
        if isinstance(body, dict):
            if isinstance(body.get("score"), (int, float)):
                return float(body["score"])
            data = body.get("data")
            if isinstance(data, list) and data:
                first = data[0]
                if isinstance(first, dict):
                    for key in ("score", "relevance_score"):
                        if isinstance(first.get(key), (int, float)):
                            return float(first[key])
            results = body.get("results")
            if isinstance(results, list) and results:
                first = results[0]
                if isinstance(first, dict):
                    for key in ("score", "relevance_score"):
                        if isinstance(first.get(key), (int, float)):
                            return float(first[key])
        raise ValueError(f"Unable to parse score response: {body!r}")

    @classmethod
    def _rerank_model_is_qwen3_vl(cls) -> bool:
        model_name = (cls._rerank_model() or "").lower()
        return "qwen3-vl-reranker" in model_name

    @classmethod
    def _rerank_doc_char_limit(cls) -> int:
        raw = (getenv("APP_RERANK_DOC_CHAR_LIMIT") or "").strip()
        if not raw:
            return 2000
        try:
            value = int(raw)
            return value if value > 0 else 2000
        except Exception:
            return 2000

    @classmethod
    def _rerank_via_local_vllm(cls, query: str, chunks_list, top_k: int):
        raw_documents = [chunk.content for chunk in chunks_list]
        multimodal_documents = [chunk._rerank_document_payload() for chunk in chunks_list]
        char_limit = cls._rerank_doc_char_limit()
        documents = [doc[:char_limit] if len(doc) > char_limit else doc for doc in raw_documents]
        multimodal_documents_trimmed: list[Any] = []
        for mm_doc, text_doc in zip(multimodal_documents, documents):
            if isinstance(mm_doc, list):
                normalized: list[dict[str, Any]] = []
                for part in mm_doc:
                    if isinstance(part, dict) and part.get("type") == "text":
                        normalized.append({"type": "text", "text": text_doc})
                    else:
                        normalized.append(part)
                multimodal_documents_trimmed.append(normalized)
            else:
                multimodal_documents_trimmed.append(text_doc)
        has_multimodal_docs = any(isinstance(doc, list) for doc in multimodal_documents_trimmed)
        if not documents:
            return cls.objects.none()

        headers = {"Content-Type": "application/json"}
        api_key = cls._rerank_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        base_v1 = cls._rerank_base_url()
        base_root = base_v1[:-3] if base_v1.endswith("/v1") else base_v1
        timeout = cls._rerank_timeout_seconds()
        first_http_error: str | None = None

        effective_query = query
        effective_documents = documents
        effective_multimodal_documents = multimodal_documents_trimmed

        def _track_http_error(endpoint: str, response: requests.Response):
            nonlocal first_http_error
            if first_http_error is not None:
                return
            response_body = ""
            try:
                response_body = response.text[:600]
            except Exception:
                response_body = "<unavailable>"
            first_http_error = f"{endpoint} -> {response.status_code}: {response_body}"

        rerank_endpoints = (
            f"{base_root}/rerank",
            f"{base_root}/v2/rerank",
            f"{base_v1}/rerank",
        )
        if cls._rerank_model_is_qwen3_vl():
            rerank_endpoints = ()
        rerank_payloads = (
            {
                "model": cls._rerank_model(),
                "query": effective_query,
                "documents": effective_documents,
                "top_n": top_k,
            },
            {
                "model": cls._rerank_model(),
                "query": effective_query,
                "documents": [{"text": doc} for doc in effective_documents],
                "top_n": top_k,
            },
            {
                "query": effective_query,
                "documents": effective_documents,
                "top_n": top_k,
            },
        )
        if has_multimodal_docs:
            rerank_payloads = rerank_payloads + (
                {
                    "model": cls._rerank_model(),
                    "query": effective_query,
                    "documents": effective_multimodal_documents,
                    "top_n": top_k,
                },
            )

        for endpoint in rerank_endpoints:
            try:
                for rerank_payload in rerank_payloads:
                    response = requests.post(
                        endpoint, headers=headers, json=rerank_payload, timeout=timeout
                    )
                    if response.status_code in (404, 405):
                        continue
                    if response.status_code >= 400:
                        _track_http_error(endpoint, response)
                        continue
                    ranked_ids = cls._parse_rerank_results(response.json(), chunks_list)
                    if ranked_ids:
                        return cls._ordered_queryset_from_ids(ranked_ids)
            except Exception:
                continue

        for endpoint in (f"{base_root}/score", f"{base_v1}/score"):
            try:
                batch_payloads = (
                    {
                        "model": cls._rerank_model(),
                        "text_1": effective_query,
                        "text_2": effective_documents,
                    },
                    {
                        "text_1": effective_query,
                        "text_2": effective_documents,
                    },
                    {
                        "model": cls._rerank_model(),
                        "query": effective_query,
                        "documents": effective_documents,
                    },
                )
                if has_multimodal_docs:
                    batch_payloads = batch_payloads + (
                        {
                            "model": cls._rerank_model(),
                            "query": [{"type": "text", "text": effective_query}],
                            "documents": effective_multimodal_documents,
                        },
                        {
                            "model": cls._rerank_model(),
                            "text_1": [{"type": "text", "text": effective_query}],
                            "text_2": effective_multimodal_documents,
                        },
                    )
                for batch_payload in batch_payloads:
                    response = requests.post(
                        endpoint, headers=headers, json=batch_payload, timeout=timeout
                    )
                    if response.status_code in (404, 405):
                        continue
                    if response.status_code >= 400:
                        _track_http_error(endpoint, response)
                        continue
                    pairs = cls._parse_score_results(response.json())
                    if not pairs:
                        continue
                    ranked_ids = [
                        chunks_list[idx].pk
                        for idx, _ in sorted(pairs, key=lambda item: item[1], reverse=True)[:top_k]
                    ]
                    if ranked_ids:
                        return cls._ordered_queryset_from_ids(ranked_ids)

                scored: list[tuple[int, float]] = []
                for idx, doc in enumerate(effective_documents):
                    mm_doc = effective_multimodal_documents[idx]
                    score_payloads = (
                        {
                            "model": cls._rerank_model(),
                            "text_1": effective_query,
                            "text_2": doc,
                        },
                        {
                            "model": cls._rerank_model(),
                            "query": effective_query,
                            "document": doc,
                        },
                        {
                            "text_1": effective_query,
                            "text_2": doc,
                        },
                    )
                    if has_multimodal_docs and isinstance(mm_doc, list):
                        score_payloads = score_payloads + (
                            {
                                "model": cls._rerank_model(),
                                "query": [{"type": "text", "text": effective_query}],
                                "document": mm_doc,
                            },
                            {
                                "model": cls._rerank_model(),
                                "text_1": [{"type": "text", "text": effective_query}],
                                "text_2": mm_doc,
                            },
                            {
                                "model": cls._rerank_model(),
                                "messages": [
                                    {
                                        "role": "system",
                                        "content": "Given a search query, retrieve relevant candidates that answer the query.",
                                    },
                                    {"role": "query", "content": effective_query},
                                    {"role": "document", "content": mm_doc},
                                ],
                            },
                        )
                    score_found = False
                    for score_payload in score_payloads:
                        response = requests.post(
                            endpoint, headers=headers, json=score_payload, timeout=timeout
                        )
                        if response.status_code in (404, 405):
                            continue
                        if response.status_code >= 400:
                            _track_http_error(endpoint, response)
                            continue
                        score = cls._parse_single_score(response.json())
                        scored.append((idx, score))
                        score_found = True
                        break
                    if not score_found:
                        continue

                if not scored:
                    continue

                scored.sort(key=lambda item: item[1], reverse=True)
                ranked_ids = [chunks_list[idx].pk for idx, _ in scored[:top_k]]
                if ranked_ids:
                    return cls._ordered_queryset_from_ids(ranked_ids)
            except Exception:
                continue

        if first_http_error:
            logger.warning("All local rerank requests failed. First error: %s", first_http_error)

        return cls.objects.none()

    @classmethod
    def rerank(cls, query: str, chunks, top_k: int):
        chunks_list = list(chunks)
        provider = (getenv("APP_RERANK_PROVIDER") or "auto").strip().lower()
        if provider in ("auto", "local", "vllm"):
            try:
                local_results = cls._rerank_via_local_vllm(query, chunks_list, top_k)
                if local_results.exists():
                    return local_results
            except Exception as exc:
                logger.warning("Local rerank failed; trying Cohere fallback. Error: %s", exc)
            if provider in ("local", "vllm"):
                return cls._fallback_rerank(chunks_list, top_k)

        cohere = apps.get_app_config("aquillm").cohere_client  # type: ignore
        if cohere is None:
            return cls._fallback_rerank(chunks_list, top_k)
        try:
            response = cohere.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=[{"content": chunk.content, "id": chunk.pk} for chunk in chunks_list],
                rank_fields=["content"],
                top_n=top_k,
                return_documents=True,
            )
            ranked_list = [result.document.id for result in response.results]
            if not ranked_list:
                return cls._fallback_rerank(chunks_list, top_k)
            return cls._ordered_queryset_from_ids(ranked_list)
        except Exception as exc:
            logger.warning("Cohere rerank failed; using fallback order. Error: %s", exc)
            return cls._fallback_rerank(chunks_list, top_k)

    @classmethod
    def text_chunk_search(cls, query: str, top_k: int, docs: List):
        from aquillm.utils import get_embedding

        vector_top_k = apps.get_app_config("aquillm").vector_top_k  # type: ignore
        trigram_top_k = apps.get_app_config("aquillm").trigram_top_k  # type: ignore
        candidate_multiplier = 3
        vector_limit = max(top_k + 2, min(vector_top_k, top_k * candidate_multiplier))
        trigram_limit = max(top_k + 2, min(trigram_top_k, top_k * candidate_multiplier))
        total_start = perf_counter()

        try:
            try:
                vector_start = perf_counter()
                query_embedding = get_embedding(query)
                vector_results = cls.objects.filter_by_documents(docs).exclude(
                    embedding__isnull=True
                ).order_by(L2Distance("embedding", query_embedding))[
                    :vector_limit
                ]  # type: ignore
                vector_ms = (perf_counter() - vector_start) * 1000
            except Exception as exc:
                logger.warning(
                    "Vector embed/search failed; continuing with trigram-only retrieval. Error: %s",
                    exc,
                )
                vector_results = cls.objects.none()
                vector_ms = (perf_counter() - total_start) * 1000
            trigram_start = perf_counter()
            trigram_results = (
                cls.objects.filter_by_documents(docs)
                .filter(modality=cls.Modality.TEXT)
                .annotate(similarity=TrigramSimilarity("content", query))  # type: ignore
                .filter(similarity__gt=0.000001)
                .order_by("-similarity")[:trigram_limit]
            )
            trigram_ms = (perf_counter() - trigram_start) * 1000
            combined_candidates = list(vector_results) + list(trigram_results)
            deduped_candidates = []
            seen_pks = set()
            for candidate in combined_candidates:
                if candidate.pk in seen_pks:
                    continue
                seen_pks.add(candidate.pk)
                deduped_candidates.append(candidate)
            combined_candidates = deduped_candidates
            if len(combined_candidates) <= top_k:
                reranked_results = cls._fallback_rerank(combined_candidates, top_k)
                rerank_ms = 0.0
            else:
                rerank_start = perf_counter()
                reranked_results = cls.rerank(query, combined_candidates, top_k)
                rerank_ms = (perf_counter() - rerank_start) * 1000
            total_ms = (perf_counter() - total_start) * 1000
            logger.info(
                "text_chunk_search latency %.1fms (vector=%.1fms trigram=%.1fms rerank=%.1fms docs=%d top_k=%d candidates=%d)",
                total_ms,
                vector_ms,
                trigram_ms,
                rerank_ms,
                len(docs),
                top_k,
                len(combined_candidates),
            )
            return vector_results, trigram_results, reranked_results
        except DatabaseError as e:
            logger.error(f"Database error during search: {str(e)}")
            raise e
        except ValidationError as e:
            logger.error(f"Validation error during search: {str(e)}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error during search: {str(e)}")
            raise e
