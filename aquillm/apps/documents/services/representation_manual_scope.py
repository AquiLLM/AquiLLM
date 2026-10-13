"""Explicit human document operations own a child, never a wider original scope."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from uuid import UUID

from lib.llm.turn_context import current_turn
from .representation_contracts import SourceScopeUnavailable
from .source_loading import SourceRuntime, bounded_source_database, current_source_runtime

_PRIVATE_CHILD = ContextVar("manual_document_private_runtime", default=None)


@dataclass(frozen=True)
class _ManualChild:
    turn: object
    parent: object
    runtime: object
    authority: object
    snapshot: object
    document: object
    principal_binding: object


def private_acquisition_runtime():
    child = _PRIVATE_CHILD.get()
    if child is not None:
        _record(child)
        return child
    return getattr(current_turn(), "_media_source_runtime", None)


def manual_acquisition_collections(documents):
    runtime = current_source_runtime() or private_acquisition_runtime()
    record = registered_child_record(runtime)
    if record is None:
        return None
    if any((d._meta.label_lower, d.pk, d.id, d.collection_id) != (
        record.document.model_label, record.document.pkid, record.document.id,
        record.document.collection_id) for d in documents):
        raise SourceScopeUnavailable("manual_source_scope_unavailable")
    return record.authority.selected_collection_ids


def _record(runtime):
    from .representation_principal import runtime_principal
    turn = current_turn()
    record = getattr(turn, "_manual_representation_children", {}).get(id(runtime))
    if (record is None or record.runtime is not runtime or record.turn is not turn
        or getattr(runtime, "_manual_factory_record", None) is not record
        or record.parent.budget is not turn.budget or runtime.budget is not turn.budget
        or record.authority is not runtime.authorization
        or runtime._frozen_retrieval_scope.runtime is not runtime
        or runtime._frozen_retrieval_scope.authorization is not record.authority
        or runtime._frozen_retrieval_scope.representation is not record.snapshot):
        raise SourceScopeUnavailable("manual_source_scope_unavailable")
    turn.budget.check_active()
    if (runtime_principal(record.parent, required=True) is not record.principal_binding
        or runtime_principal(runtime, required=True) is not record.principal_binding):
        raise SourceScopeUnavailable("manual_source_scope_unavailable")
    return record


def registered_child_record(runtime):
    """An initialized nonparent remains a child even after registry corruption."""
    turn = current_turn()
    if (getattr(runtime, "_manual_factory_record", None) is not None
        or id(runtime) in getattr(turn, "_manual_representation_children", {})
        or (getattr(runtime, "_media_principal_binding", None) is not None
            and runtime is not getattr(turn, "_media_source_runtime", None))):
        return _record(runtime)
    return None


def create_manual_document_runtime(user, document_id, *, parent):
    """Private caller contract: only the parsed/resolved human /document branch."""
    from apps.collections.services.django_retrieval_authorization import build_selected_scope_authorization_context
    from apps.documents.models import DESCENDED_FROM_DOCUMENT
    from .representation_tool_scope import tool_scope
    from .representation_history import reference
    turn = current_turn()
    if (turn is None or parent is None or parent.budget is not turn.budget
        or parent is not (current_source_runtime() or private_acquisition_runtime())):
        raise SourceScopeUnavailable("manual_source_scope_unavailable")
    from .representation_principal import runtime_principal
    binding = runtime_principal(parent, principal=user, required=True)
    registered_child_record(parent)
    document_id = UUID(str(document_id))
    from .representation_unavailable_parent import UnavailableAcquisitionParent, validate_unavailable_parent
    unavailable = type(parent) is UnavailableAcquisitionParent
    if unavailable:
        validate_unavailable_parent(parent, principal=user)
    alias = "default" if unavailable else parent.authorization.database_alias
    with parent.lock, bounded_source_database(parent, alias):
        # Preserve the parent's original selection even when manual is first.
        if not unavailable:
            tool_scope(parent, using=alias)
        matches = []
        for model in DESCENDED_FROM_DOCUMENT:
            matches.extend(model._base_manager.using(alias).filter(id=document_id).only(
                "pkid", "id", "collection_id", "full_text_hash")[:2])
            if len(matches) > 1:
                raise SourceScopeUnavailable("manual_source_scope_unavailable")
        if len(matches) != 1:
            raise SourceScopeUnavailable("manual_source_scope_unavailable")
        document = matches[0]
        authority = build_selected_scope_authorization_context(principal=user,
            selected_collection_ids=(document.collection_id,), selected_documents=(document,), database_alias=alias)
        if authority is None:
            raise SourceScopeUnavailable("manual_source_scope_unavailable")
        child = SourceRuntime(parent.budget, authority, parent.cache, parent.windows, parent.lock, parent.observation)
        child._media_authorization = authority
        child._media_principal_binding = binding
        for name in ("_media_col_ref", "_media_consumer", "_media_collections"):
            if hasattr(parent, name):
                setattr(child, name, getattr(parent, name))
        captured = tool_scope(child, using=alias)
        registry = turn.__dict__.setdefault("_manual_representation_children", {})
        runtime_principal(child, principal=user, required=True)
        record = _ManualChild(turn, parent, child, authority, captured.representation, reference(document), binding)
        child._manual_factory_record = record
        registry[id(child)] = record
        return child


@contextmanager
def private_manual_acquisition(runtime):
    _record(runtime)
    if current_source_runtime() is not None:
        raise SourceScopeUnavailable("manual_private_scope_requires_default_route")
    token = _PRIVATE_CHILD.set(runtime)
    try:
        yield runtime
    finally:
        _PRIVATE_CHILD.reset(token)


def bind_manual_publication(packet, context, runtime):
    record = registered_child_record(runtime)
    if record is None:
        return
    if (packet.source_authorization is not record.authority
        or getattr(packet, "_publication_representation", None) is not record.snapshot
        or getattr(packet, "_publication_manual_record", None) is not record
        or context._manual_record is not record):
        raise SourceScopeUnavailable("manual_source_scope_unavailable")
    bindings = record.turn.__dict__.setdefault("_manual_publication_bindings", {})
    previous = bindings.get(id(packet))
    if previous is not None and any(a is not b for a, b in zip(previous, (packet, context, record))):
        raise SourceScopeUnavailable("manual_publication_changed")
    bindings[id(packet)] = (packet, context, record)


def packet_media_runtime(packet):
    """Final sinks may use only this exact privately registered child/context."""
    turn = current_turn()
    bound = getattr(turn, "_manual_publication_bindings", {}).get(id(packet))
    if bound is None:
        record = getattr(packet, "_publication_manual_record", None)
        if record is not None:
            # Constructor digesting is still inside the exact child's acquisition.
            # Once its context exists, every use requires the retained registry.
            context = getattr(turn, "answer_publication", None)
            active = current_source_runtime() or _PRIVATE_CHILD.get()
            if (getattr(context, "packet", None) is not packet and active is record.runtime
                and record is _record(active)):
                return active
            raise SourceScopeUnavailable("manual_publication_changed")
        return None
    original, context, record = bound
    if (original is not packet or context is not turn.answer_publication
        or context.packet is not packet or record is not _record(record.runtime)
        or context._manual_record is not record
        or getattr(packet, "_publication_manual_record", None) is not record):
        raise SourceScopeUnavailable("manual_publication_changed")
    context._active()
    return record.runtime
