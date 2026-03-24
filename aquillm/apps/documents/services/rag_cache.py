"""RAG-oriented Django cache helpers: normalized keys, fail-open get/set, metric logging."""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from typing import Any, Mapping, Sequence

from django.conf import settings
from django.core.cache import cache

logger = logging.getLogger(__name__)

_MET_QUERY_EMBED = "rag_cache.query_embed"
_MET_DOC_ACCESS = "rag_cache.doc_access"
_MET_DOC_LOOKUP = "rag_cache.document_lookup"
_MET_IMAGE_URL = "rag_cache.image_data_url"
_MET_RERANK_RES = "rag_cache.rerank_result"
_MET_RERANK_CAP = "rag_cache.rerank_capability"


def _rag_enabled() -> bool:
    return bool(getattr(settings, "RAG_CACHE_ENABLED", False))


def stable_cache_key(prefix: str, *parts: Any) -> str:
    """Build a deterministic key; hash when the canonical payload is long."""
    canonical = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    if len(canonical) > 180:
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"{prefix}:h:{digest}"
    return f"{prefix}:{canonical}"


def cache_get(key: str) -> Any | None:
    if not _rag_enabled():
        return None
    try:
        return cache.get(key)
    except Exception as exc:
        logger.warning("cache_get failed (fail-open) key=%s err=%s", key[:120], exc)
        return None


def cache_set(key: str, value: Any, timeout: int) -> None:
    if not _rag_enabled():
        return
    try:
        cache.set(key, value, timeout=timeout)
    except Exception as exc:
        logger.warning("cache_set failed (fail-open) key=%s err=%s", key[:120], exc)


def _log_hit_miss(metric: str, hit: bool) -> None:
    if not _rag_enabled():
        return
    if hit:
        logger.info("%s hit", metric)
    else:
        logger.debug("%s miss", metric)


def query_embedding_cache_key(query: str, input_type: str, model_signature: str) -> str:
    # Hash query text so keys stay short and safe for memcached-style backends.
    qhash = hashlib.sha256(query.encode("utf-8")).hexdigest()
    return stable_cache_key("qe", input_type, model_signature, qhash)


def query_embed_ttl() -> int:
    return int(getattr(settings, "RAG_QUERY_EMBED_TTL_SECONDS", 300))


def get_cached_query_embedding(
    query: str, input_type: str, model_signature: str
) -> list[float] | None:
    if not _rag_enabled():
        return None
    key = query_embedding_cache_key(query, input_type, model_signature)
    val = cache_get(key)
    if isinstance(val, list) and val and isinstance(val[0], (int, float)):
        _log_hit_miss(_MET_QUERY_EMBED, True)
        return [float(x) for x in val]
    _log_hit_miss(_MET_QUERY_EMBED, False)
    return None


def set_cached_query_embedding(
    query: str, input_type: str, model_signature: str, embedding: list[float]
) -> None:
    if not _rag_enabled():
        return
    key = query_embedding_cache_key(query, input_type, model_signature)
    cache_set(key, embedding, query_embed_ttl())


def doc_access_cache_key(user_id: int, collection_ids: tuple[int, ...], perm: str) -> str:
    return stable_cache_key("da", user_id, collection_ids, perm)


def doc_access_ttl() -> int:
    return int(getattr(settings, "RAG_DOC_ACCESS_TTL_SECONDS", 60))


def get_cached_doc_access_refs(
    user_id: int, collection_ids: tuple[int, ...], perm: str
) -> list[Mapping[str, Any]] | None:
    if not _rag_enabled():
        return None
    key = doc_access_cache_key(user_id, collection_ids, perm)
    val = cache_get(key)
    if val is None:
        _log_hit_miss(_MET_DOC_ACCESS, False)
        return None
    if isinstance(val, list):
        _log_hit_miss(_MET_DOC_ACCESS, True)
        return val  # type: ignore[return-value]
    _log_hit_miss(_MET_DOC_ACCESS, False)
    return None


def set_cached_doc_access_refs(
    user_id: int,
    collection_ids: tuple[int, ...],
    perm: str,
    refs: list[Mapping[str, Any]],
) -> None:
    if not _rag_enabled():
        return
    key = doc_access_cache_key(user_id, collection_ids, perm)
    cache_set(key, refs, doc_access_ttl())


def document_lookup_cache_key(doc_id: uuid.UUID) -> str:
    return f"dl:{doc_id}"


def document_lookup_ttl() -> int:
    return int(getattr(settings, "RAG_DOC_ACCESS_TTL_SECONDS", 60))


def get_cached_document_ref(doc_id: uuid.UUID) -> Mapping[str, Any] | None:
    if not _rag_enabled():
        return None
    key = document_lookup_cache_key(doc_id)
    val = cache_get(key)
    if isinstance(val, dict) and "model" in val and "pkid" in val:
        _log_hit_miss(_MET_DOC_LOOKUP, True)
        return val
    _log_hit_miss(_MET_DOC_LOOKUP, False)
    return None


def set_cached_document_ref(doc_id: uuid.UUID, ref: Mapping[str, Any]) -> None:
    if not _rag_enabled():
        return
    key = document_lookup_cache_key(doc_id)
    cache_set(key, dict(ref), document_lookup_ttl())


def image_data_url_cache_key(doc_id: uuid.UUID, image_file_name: str) -> str:
    return stable_cache_key("img", str(doc_id), image_file_name)


def image_data_url_ttl() -> int:
    return int(getattr(settings, "RAG_IMAGE_DATA_URL_TTL_SECONDS", 120))


def get_cached_image_data_url(doc_id: uuid.UUID, image_file_name: str) -> str | None:
    if not _rag_enabled():
        return None
    key = image_data_url_cache_key(doc_id, image_file_name)
    val = cache_get(key)
    if isinstance(val, str) and val.startswith("data:"):
        _log_hit_miss(_MET_IMAGE_URL, True)
        return val
    _log_hit_miss(_MET_IMAGE_URL, False)
    return None


def set_cached_image_data_url(doc_id: uuid.UUID, image_file_name: str, data_url: str) -> None:
    if not _rag_enabled():
        return
    key = image_data_url_cache_key(doc_id, image_file_name)
    cache_set(key, data_url, image_data_url_ttl())


def rerank_capability_cache_key(base_url: str, model: str) -> str:
    return stable_cache_key("rrcap", base_url.rstrip("/"), model)


def rerank_capability_ttl() -> int:
    return int(getattr(settings, "RAG_RERANK_CAPABILITY_TTL_SECONDS", 900))


def get_cached_rerank_capability(base_url: str, model: str) -> str | None:
    if not _rag_enabled():
        return None
    key = rerank_capability_cache_key(base_url, model)
    val = cache_get(key)
    if isinstance(val, str) and val:
        _log_hit_miss(_MET_RERANK_CAP, True)
        return val
    _log_hit_miss(_MET_RERANK_CAP, False)
    return None


def set_cached_rerank_capability(base_url: str, model: str, endpoint: str) -> None:
    if not _rag_enabled():
        return
    key = rerank_capability_cache_key(base_url, model)
    cache_set(key, endpoint, rerank_capability_ttl())


def rerank_result_cache_key(
    query_signature: str, candidate_ids: Sequence[int], top_k: int, model: str
) -> str:
    ids_tuple = tuple(int(x) for x in candidate_ids)
    return stable_cache_key("rrres", query_signature, ids_tuple, top_k, model)


def rerank_result_ttl() -> int:
    return int(getattr(settings, "RAG_RERANK_RESULT_TTL_SECONDS", 45))


def query_signature_for_rerank(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def get_cached_rerank_result(
    query_signature: str, candidate_ids: Sequence[int], top_k: int, model: str
) -> list[int] | None:
    if not _rag_enabled():
        return None
    key = rerank_result_cache_key(query_signature, candidate_ids, top_k, model)
    val = cache_get(key)
    if isinstance(val, list) and all(isinstance(x, int) for x in val):
        _log_hit_miss(_MET_RERANK_RES, True)
        return val
    _log_hit_miss(_MET_RERANK_RES, False)
    return None


def set_cached_rerank_result(
    query_signature: str,
    candidate_ids: Sequence[int],
    top_k: int,
    model: str,
    ranked_ids: list[int],
) -> None:
    if not _rag_enabled():
        return
    key = rerank_result_cache_key(query_signature, candidate_ids, top_k, model)
    cache_set(key, list(ranked_ids), rerank_result_ttl())


def document_refs_from_documents(documents: Sequence[Any]) -> list[dict[str, Any]]:
    return [{"model": d.__class__.__name__, "pkid": int(d.pkid)} for d in documents]


def rehydrate_documents_from_refs(refs: Sequence[Mapping[str, Any]]) -> list[Any]:
    from django.apps import apps

    out: list[Any] = []
    for ref in refs:
        try:
            model = apps.get_model("apps_documents", str(ref["model"]))
            doc = model.objects.filter(pkid=int(ref["pkid"])).first()
            if doc is not None:
                out.append(doc)
        except Exception:
            continue
    return out


__all__ = [
    "cache_get",
    "cache_set",
    "doc_access_cache_key",
    "document_lookup_cache_key",
    "get_cached_doc_access_refs",
    "get_cached_document_ref",
    "get_cached_image_data_url",
    "get_cached_query_embedding",
    "get_cached_rerank_capability",
    "get_cached_rerank_result",
    "image_data_url_cache_key",
    "query_signature_for_rerank",
    "set_cached_doc_access_refs",
    "set_cached_document_ref",
    "set_cached_image_data_url",
    "set_cached_query_embedding",
    "set_cached_rerank_capability",
    "set_cached_rerank_result",
    "stable_cache_key",
    "document_refs_from_documents",
    "rehydrate_documents_from_refs",
]
