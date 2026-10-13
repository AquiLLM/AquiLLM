"""Bind answer publication to the representation acquired in this same turn."""
from types import SimpleNamespace

from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.representation_store import freeze_active_representations, validate_representation_snapshot
from apps.documents.services.representation_history import reference
from apps.documents.services.source_loading import current_source_runtime
from lib.llm.turn_context import current_turn


def publication_runtime():
    from apps.documents.services.representation_media_acquisition import media_runtime
    return media_runtime()


def authorized_snapshot(authority, *, using):
    from apps.documents.services.representation_tool_scope import original_document_metadata
    documents = original_document_metadata(SimpleNamespace(authorization=authority), using=using)
    return freeze_active_representations(tuple(reference(d) for d in documents), using=using)


def capture_legacy_snapshot(packet, authority, documents):
    runtime = publication_runtime()
    if runtime is not None:
        from apps.documents.services.representation_principal import retain_packet_principal
        retain_packet_principal(packet, runtime)
        from apps.documents.services.chunk_search_scope import validate_requested_authorization
        from apps.documents.services.representation_tool_scope import tool_scope
        with runtime.lock:
            validate_requested_authorization(runtime, authority, documents)
            snapshot = tool_scope(runtime, using=authority.database_alias, require_existing=True).representation
        authority = runtime.authorization
    else:
        if current_turn() is not None:
            raise SourceScopeUnavailable("source_scope_unavailable")
        snapshot = authorized_snapshot(authority, using=authority.database_alias)
    old = getattr(packet, "_publication_representation", None)
    if old is not None and old is not snapshot:
        raise SourceScopeUnavailable("source_scope_unavailable")
    packet._publication_representation = snapshot
    return authority


def retain_acquired_snapshot(packet):
    """No database work or fresh scope capture at the publication boundary."""
    runtime = publication_runtime()
    if runtime is None:
        if current_turn() is not None:
            raise SourceScopeUnavailable("source_scope_unavailable")
        return
    from apps.documents.services.representation_principal import retain_packet_principal
    retain_packet_principal(packet, runtime)
    from .retrieval_query_publication import retain_query_binding
    retain_query_binding(packet, runtime)
    with runtime.lock:
        scope = getattr(runtime, "_frozen_retrieval_scope", None)
        if scope is None or scope.runtime is not runtime:
            raise SourceScopeUnavailable("source_scope_unavailable")
        old = getattr(packet, "_publication_representation", None)
        if old is not None and old is not scope.representation:
            raise SourceScopeUnavailable("source_scope_unavailable")
        authority = packet.source_authorization
        if authority is not None and (
            authority.principal_reference != runtime.authorization.principal_reference
            or not authority.selected_document_ids <= runtime.authorization.selected_document_ids
            or authority.database_alias != runtime.authorization.database_alias
        ):
            raise SourceScopeUnavailable("source_scope_unavailable")
        packet._publication_representation = scope.representation


def current_packet_snapshot(packet):
    from apps.documents.services.representation_principal import validate_packet_principal
    validate_packet_principal(packet)
    snapshot = getattr(packet, "_publication_representation", None)
    if snapshot is None:
        raise SourceScopeUnavailable("source_scope_unavailable")
    return validate_representation_snapshot(snapshot, using=packet.source_authorization.database_alias)
