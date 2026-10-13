"""One original turn ledger and scope own every verified image-byte acquisition."""
import base64
from dataclasses import dataclass
from hashlib import sha256
from threading import RLock

from django.apps import apps
from django.core.exceptions import ObjectDoesNotExist, ValidationError
from django.db.models.functions import MD5

from apps.documents.models import TextChunk
from lib.llm.turn_context import current_turn
from .representation_contracts import SourceScopeUnavailable
from .representation_media import load_chunk_media_binding, verified_artifact_stream
from .representation_tool_scope import tool_scope
from .source_deadline import check_source_deadline
from .source_loading import current_source_runtime, bounded_source_database


@dataclass(frozen=True, eq=False)
class CapturedChunkMedia:
    binding: object
    snapshot: object
    data: bytes
    data_url: str
    source_fingerprint: str


def initialize_media_runtime(consumer, turn):
    """Called once at the original authorized turn boundary, never by a finalizer."""
    from apps.chat.services.rag_turn import create_runtime
    from .representation_unavailable_parent import (
        UnavailableAcquisitionParent, unavailable_parent, validate_unavailable_parent,
    )
    from .representation_principal import capture_principal, runtime_principal, validate_principal
    lock = turn.__dict__.setdefault("_media_initialization_lock", RLock())
    with lock:
        turn.budget.check_active()
        existing = getattr(turn, "_media_source_runtime", None)
        if existing is not None:
            binding = runtime_principal(existing, turn=turn, principal=consumer.user, required=True)
            if binding.consumer is not consumer:
                raise SourceScopeUnavailable("source_scope_unavailable")
            if type(existing) is UnavailableAcquisitionParent:
                return validate_unavailable_parent(existing, turn=turn, principal=consumer.user)
            if existing.budget is not turn.budget or existing.authorization is not existing._media_authorization:
                raise SourceScopeUnavailable("source_scope_unavailable")
            tool_scope(existing, using=existing.authorization.database_alias, require_existing=True)
            return existing
        binding = capture_principal(consumer, turn)
        if hasattr(turn, "_media_principal_binding"):
            raise SourceScopeUnavailable("source_scope_unavailable")
        turn._media_principal_binding = binding
        runtime = create_runtime(consumer, turn.budget)
        validate_principal(binding, turn=turn)
        if runtime.authorization is None:
            runtime = unavailable_parent(consumer, turn, binding)
            turn._media_source_runtime = runtime
            return runtime
        runtime._media_principal_binding = binding
        runtime._media_col_ref = consumer.col_ref
        runtime._media_consumer = consumer
        runtime._media_collections = tuple(consumer.col_ref.collections)
        runtime._media_authorization = runtime.authorization
        # Only capture temporarily binds SourceRuntime; routing keeps its default.
        tool_scope(runtime, using=runtime.authorization.database_alias)
        runtime_principal(runtime, turn=turn, required=True)
        turn._media_source_runtime = runtime
        return runtime


def media_runtime():
    from .representation_manual_scope import private_acquisition_runtime
    return current_source_runtime() or private_acquisition_runtime()


def _check_runtime(runtime):
    from .source_loading import SourceRuntime
    turn = current_turn()
    if not isinstance(runtime, SourceRuntime) or runtime.authorization is None or turn is None or turn.budget is not runtime.budget:
        raise SourceScopeUnavailable("media_acquisition_unavailable")
    from .representation_principal import runtime_principal
    runtime_principal(runtime)
    runtime.budget.check_active()
    if runtime.budget._synthesis is None:
        check_source_deadline()
    ref = getattr(runtime, "_media_col_ref", None)
    if ref is not None and (tuple(ref.collections) != runtime._media_collections
        or runtime._media_consumer.col_ref is not ref
        or runtime.authorization is not runtime._media_authorization):
        raise SourceScopeUnavailable("source_scope_unavailable")


def _metadata(runtime, chunk_id, *, require_existing):
    scope = tool_scope(runtime, using=runtime.authorization.database_alias, require_existing=require_existing)
    binding, revision = _binding_metadata(scope.queryset(TextChunk.objects.using(scope.using).all()),
                                         chunk_id, using=scope.using)
    return scope.representation, binding, revision


def _binding_metadata(query, chunk_id, *, using):
    meta = query.filter(pk=chunk_id).annotate(
        revision=MD5("content")).values("pk", "doc_id", "revision").first()
    if meta is None:
        raise SourceScopeUnavailable("media_acquisition_unavailable")
    binding = load_chunk_media_binding(chunk_id, kind="image", using=using)
    ref = binding.document
    model = apps.get_model(ref.model_label)
    if "image_file" not in {field.name for field in model._meta.fields}:
        raise SourceScopeUnavailable("media_acquisition_unavailable")
    path = model.objects.using(using).filter(pk=ref.pkid, id=ref.id).values_list("image_file", flat=True).get()
    if path != binding.artifact.storage_key:
        raise SourceScopeUnavailable("media_acquisition_unavailable")
    return binding, meta["revision"]


def validate_private_capture(runtime, capture):
    _check_runtime(runtime)
    stored = getattr(runtime, "_media_originals", {}).get(id(capture))
    if (type(capture) is not CapturedChunkMedia or stored is None or stored[0] is not capture
        or capture.binding is not stored[1] or capture.snapshot is not stored[2]
        or capture.data is not stored[3] or capture.data_url is not stored[4]
        or capture.source_fingerprint != stored[5]
        or len(capture.data) != capture.binding.artifact.size_bytes
        or sha256(capture.data).hexdigest() != capture.binding.artifact.sha256):
        raise SourceScopeUnavailable("media_capture_changed")
    if runtime.budget._synthesis is not None:
        from .representation_media_publication import media_publication_context
        media_publication_context(runtime, capture)


def validate_captured_media(runtime, capture):
    """Revalidate metadata and the retained bytes, without reopening storage."""
    if runtime.budget._synthesis is not None:
        from .representation_media_publication import validate_published_capture
        return validate_published_capture(runtime, capture)
    with runtime.lock, bounded_source_database(runtime, runtime.authorization.database_alias):
        validate_private_capture(runtime, capture)
        snapshot, binding, revision = _metadata(runtime, capture.binding.chunk_id, require_existing=True)
        if snapshot is not capture.snapshot or binding != capture.binding or revision != capture.source_fingerprint:
            raise SourceScopeUnavailable("media_capture_changed")
    return capture


def acquire_chunk_media(chunk, *, runtime=None):
    runtime = runtime or media_runtime()
    _check_runtime(runtime)
    if not runtime.budget.can_publish():
        raise SourceScopeUnavailable("media_acquisition_closed")
    with runtime.lock:
        captured = getattr(runtime, "_media_captures", {}).get(chunk.pk)
        if captured is not None:
            return validate_captured_media(runtime, captured)
        with bounded_source_database(runtime, runtime.authorization.database_alias):
            snapshot, binding, revision = _metadata(runtime, chunk.pk, require_existing=False)
            size = binding.artifact.size_bytes
            encoded_size = 4 * ((size + 2) // 3) + len(binding.content_type) + 13
            if (not runtime.budget.admit_source((chunk.pk, revision))
                or not runtime.budget.reserve_text(size + encoded_size, kind="materialized")
                or not runtime.budget.reserve_text(encoded_size, kind="tokenized")):
                raise SourceScopeUnavailable("media_budget_unavailable")
        def check():
            _check_runtime(runtime)
        with verified_artifact_stream(binding.artifact, check_active=check) as stream:
            data = stream.read(size + 1)
        encoded = f"data:{binding.content_type};base64," + base64.b64encode(data).decode("ascii")
        capture = CapturedChunkMedia(binding, snapshot, data, encoded, revision)
        if not hasattr(runtime, "_media_originals"):
            runtime._media_originals, runtime._media_captures = {}, {}
        runtime._media_originals[id(capture)] = (capture, binding, snapshot, data, encoded, revision)
        try:
            validate_captured_media(runtime, capture)
        except Exception:
            runtime._media_originals.pop(id(capture), None)
            raise
        runtime._media_captures[chunk.pk] = capture
        return capture


def chunk_media_url(chunk):
    try:
        capture = acquire_chunk_media(chunk)
        return f"/aquillm/document_image/{capture.binding.document.id}/?chunk_id={chunk.pk}"
    except (ValueError, ObjectDoesNotExist, ValidationError, OSError):
        return None


def captured_chunk_data_url(chunk):
    try:
        return acquire_chunk_media(chunk).data_url
    except (ValueError, ObjectDoesNotExist, ValidationError, OSError):
        return None
