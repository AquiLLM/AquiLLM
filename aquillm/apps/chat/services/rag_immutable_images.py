"""Private captured-media authority; URLs are display references, never receipts."""
from hashlib import sha256
from os import getenv

from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.representation_inputs import manifest_hash
from apps.documents.services.representation_media_acquisition import (
    media_runtime, validate_captured_media, validate_private_capture,
)
from apps.documents.services.representation_store import representation_snapshot_digest


def capture_url(capture):
    return f"/aquillm/document_image/{capture.binding.document.id}/?chunk_id={capture.binding.chunk_id}"


def attach_captured_image_to_rows(rows, url):
    """Whole-tool reconstruction may retain only an already acquired exact chunk."""
    runtime = media_runtime()
    if runtime is None or not url:
        return
    for capture in getattr(runtime, "_media_captures", {}).values():
        if capture_url(capture) != url:
            continue
        validate_private_capture(runtime, capture)
        for row in rows:
            if (row.get("doc_id"), row.get("chunk_id")) == (
                str(capture.binding.document.id), capture.binding.chunk_id):
                row["image_url"] = url
                return


def _urls(packet):
    return set(packet.image_urls) | {figure.get("image_url") for figure in packet.auxiliary_figures}


def retain_packet_media(packet):
    runtime = media_runtime()
    urls = _urls(packet)
    if not urls:
        return
    if runtime is None or None in urls:
        raise SourceScopeUnavailable("media_capture_unavailable")
    with runtime.lock:
        available = {capture_url(c): c for c in getattr(runtime, "_media_captures", {}).values()}
        selected = {(str(row.get("doc_id", row.get("d"))), row.get("chunk_id", row.get("i")))
                    for row in packet.chunks}
        captures = []
        for url in sorted(urls):
            capture = available.get(url)
            if (capture is None or (str(capture.binding.document.id), capture.binding.chunk_id) not in selected
                or capture.snapshot is not getattr(packet, "_publication_representation", None)):
                raise SourceScopeUnavailable("media_capture_unavailable")
            validate_captured_media(runtime, capture)
            captures.append(capture)
        existing = getattr(packet, "_media_bindings", None)
        if existing is not None:
            if len(existing) != len(captures) or any(a is not b for a, b in zip(existing, captures)):
                raise SourceScopeUnavailable("media_capture_changed")
        else:
            packet._media_bindings = tuple(captures)
            packet._publication_media_runtime = runtime


def media_summary(packet, *, current=False):
    captures = getattr(packet, "_media_bindings", None)
    if captures is None:
        return None
    runtime = getattr(packet, "_publication_media_runtime", None)
    from apps.documents.services.representation_manual_scope import packet_media_runtime
    expected_runtime = packet_media_runtime(packet) or media_runtime()
    if runtime is not expected_runtime or {capture_url(c) for c in captures} != _urls(packet):
        raise SourceScopeUnavailable("media_capture_changed")
    result = []
    for capture in captures:
        (validate_captured_media if current else validate_private_capture)(runtime, capture)
        binding = capture.binding
        if capture.snapshot is not packet._publication_representation:
            raise SourceScopeUnavailable("media_capture_changed")
        result.append({"url": capture_url(capture), "artifact": binding.artifact.sha256,
            "byte_digest": sha256(capture.data).hexdigest(), "content_type": binding.content_type,
            "source_revision": binding.source_revision_sha256, "source_fingerprint": capture.source_fingerprint,
            "representation": representation_snapshot_digest(capture.snapshot)})
    return tuple(result)


def media_digest(packet):
    summary = media_summary(packet)
    return manifest_hash(summary) if summary is not None else None


def request_images(packet):
    if (getenv("LLM_TOOL_INLINE_IMAGES", "0") or "").strip().lower() not in {"1", "true", "yes", "on"}:
        return []
    if not getattr(packet, "_media_bindings", None):
        return []
    media_summary(packet)
    from lib.llm.providers.rag_citations import _chunk_citation_from_row
    captures = {}
    for capture in packet._media_bindings:
        key = (str(capture.binding.document.id), capture.binding.chunk_id)
        if key in captures:
            raise SourceScopeUnavailable("media_capture_changed")
        captures[key] = capture
    images, seen = [], set()
    for index, row in enumerate(packet.chunks, 1):
        key = (str(row.get("doc_id", row.get("d"))), row.get("chunk_id", row.get("i")))
        capture = captures.get(key)
        if capture is None:
            if row.get("image_url", row.get("u")):
                raise SourceScopeUnavailable("media_capture_changed")
            continue
        citation = _chunk_citation_from_row(row)
        if (key in seen or citation != f"[doc:{key[0]} chunk:{key[1]}]"
            or citation not in packet.citation_tokens
            or row.get("image_url", row.get("u")) != capture_url(capture)):
            raise SourceScopeUnavailable("media_capture_changed")
        seen.add(key)
        # Exact captured bytes, with their selected evidence position and citation.
        images.append({"image_data_url": capture.data_url, "result_index": index,
            "title": row.get("title", row.get("n", "Untitled")), "caption": citation})
    if seen != captures.keys():
        raise SourceScopeUnavailable("media_capture_changed")
    return images


def selected_image_lines(packet, requested):
    if not requested:
        return ()
    summary = media_summary(packet)
    if summary is None or not requested <= {row["url"] for row in summary}:
        raise SourceScopeUnavailable("media_capture_unavailable")
    return tuple(f"![Selected source image]({url})" for url in sorted(requested))


async def authorize_media_dispatch(packet, *, pre_seal=False):
    """Immediate pre-dispatch check; never hold locks during the provider call."""
    from apps.documents.services.representation_principal import validate_packet_principal
    validate_packet_principal(packet)
    from lib.llm.turn_context import current_turn
    from lib.llm.answer_publication import AnswerPublicationRejected
    turn = current_turn()
    context = getattr(turn, "answer_publication", None)
    if packet.source_authorization is not None or getattr(packet, "_media_bindings", None):
        if context is None:
            if not pre_seal:
                raise AnswerPublicationRejected("unbound_receipt")
        elif context.packet is not packet:
            raise AnswerPublicationRejected("unbound_receipt")
        else:
            context._active()
            context._unchanged()
    if not getattr(packet, "_media_bindings", None):
        if getattr(packet, "_publication_query_binding", None) is not None and not pre_seal:
            await context._authorized()
        return
    from .rag_answer_publication import bind_packet_publication
    if turn is None:
        raise AnswerPublicationRejected("unbound_receipt")
    context = turn.answer_publication
    if context is None:
        context = bind_packet_publication(packet, lease=turn.budget._synthesis)
    elif context.packet is not packet:
        raise AnswerPublicationRejected("evidence_changed")
    await context._authorized()
