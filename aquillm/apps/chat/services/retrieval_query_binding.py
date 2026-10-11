"""One original query admission; metadata checks never refresh its witnesses."""
from dataclasses import dataclass
from uuid import UUID

from apps.documents.services.representation_contracts import SourceScopeUnavailable
from lib.llm.turn_context import current_turn


@dataclass(frozen=True, eq=False)
class QueryAdmission:
    runtime: object
    budget: object
    authorization: object
    scope: object
    representation: object
    turn: object
    attempt: object


def admission(runtime):
    from apps.documents.services.representation_principal import runtime_principal
    runtime_principal(runtime)
    value = getattr(runtime, "_query_admission", None)
    turn = current_turn()
    if (type(value) is not QueryAdmission or value.runtime is not runtime
        or value.budget is not runtime.budget or value.authorization is not runtime.authorization
        or value.scope is not getattr(runtime, "_frozen_retrieval_scope", None)
        or value.scope.runtime is not runtime or value.scope.authorization is not value.authorization
        or value.representation is not value.scope.representation
        or value.attempt is not getattr(runtime, "_query_admission_attempt", None)
        or value.turn is not turn
        or (turn is not None and (turn.budget is not value.budget
            or getattr(turn, "_query_admissions", {}).get(id(runtime)) is not value))):
        raise SourceScopeUnavailable("source_scope_unavailable")
    value.budget.check_active()
    return value


def ensure_admission(runtime):
    from apps.documents.services.source_loading import SourceRuntime, current_source_runtime
    from apps.documents.services.representation_tool_scope import tool_scope
    if not isinstance(runtime, SourceRuntime) or current_source_runtime() is not runtime:
        raise SourceScopeUnavailable("source_scope_unavailable")
    with runtime.lock:
        if hasattr(runtime, "_query_admission_attempt"):
            return admission(runtime)
        turn, attempt = current_turn(), object()
        runtime._query_admission_attempt = attempt
        if turn is not None:
            if turn.budget is not runtime.budget:
                raise SourceScopeUnavailable("source_scope_unavailable")
            registry = turn.__dict__.setdefault("_query_admissions", {})
            if id(runtime) in registry:
                raise SourceScopeUnavailable("source_scope_unavailable")
            registry[id(runtime)] = attempt
        from apps.documents.services.representation_principal import runtime_principal
        initialized = runtime_principal(runtime) is not None
        tool_scope(runtime, using=runtime.authorization.database_alias, require_existing=initialized)
        scope = runtime._frozen_retrieval_scope
        value = QueryAdmission(runtime, runtime.budget, runtime.authorization, scope,
                               scope.representation, turn, attempt)
        runtime._query_admission = value
        if turn is not None:
            registry[id(runtime)] = value
        return admission(runtime)


def retain_original(runtime, bound):
    witness = admission(runtime)
    if hasattr(runtime, "_query_original_binding"):
        raise SourceScopeUnavailable("source_scope_unavailable")
    runtime._query_original_binding = bound
    runtime._query_binding_record = (witness, bound)
    if witness.turn is not None:
        witness.turn.__dict__.setdefault("_query_binding_records", {})[id(runtime)] = runtime._query_binding_record


def original_binding(runtime):
    witness = admission(runtime)
    record = getattr(runtime, "_query_binding_record", None)
    bound = getattr(runtime, "_query_original_binding", None)
    if (type(record) is not tuple or len(record) != 2 or record[0] is not witness
        or record[1] is not bound or bound is None or bound.runtime is not runtime
        or (witness.turn is not None and
            getattr(witness.turn, "_query_binding_records", {}).get(id(runtime)) is not record)):
        raise SourceScopeUnavailable("source_scope_unavailable")
    return bound


def anchor_rows(witness, anchors):
    """Exact coordinate metadata only; caller owns first/latest current checks."""
    from django.db.models import Q
    from apps.documents.models import TextChunk
    from apps.documents.services.representation_store import active_chunk_scope
    wanted = tuple(anchors.chunk_identities)
    if len(wanted) > 45 or len(set(wanted)) != len(wanted):
        raise SourceScopeUnavailable("source_scope_unavailable")
    allowed = witness.authorization.selected_document_ids
    condition = Q(pk__in=[])
    for pk, doc, number in wanted:
        if type(pk) is not int or pk <= 0 or type(number) is not int or number < 0:
            raise SourceScopeUnavailable("source_scope_unavailable")
        try:
            identity = UUID(doc)
        except (TypeError, ValueError, AttributeError):
            raise SourceScopeUnavailable("source_scope_unavailable") from None
        if identity not in allowed:
            raise SourceScopeUnavailable("source_scope_unavailable")
        condition |= Q(pk=pk, doc_id=identity, chunk_number=number)
    if not wanted:
        return ()
    rows = tuple(active_chunk_scope(TextChunk.objects.using(witness.scope.using).filter(condition),
        snapshot=witness.representation).values_list("pk", "doc_id", "chunk_number")[:len(wanted) + 1])
    actual = {(pk, str(doc), number) for pk, doc, number in rows}
    if len(rows) != len(wanted) or actual != set(wanted):
        raise SourceScopeUnavailable("source_scope_unavailable")
    return rows


def validate_anchors(runtime, anchors):
    witness = admission(runtime)
    if anchors is None:
        return
    if anchors.unresolved_references:
        raise SourceScopeUnavailable("source_scope_unavailable")
    if not anchors.chunk_identities:
        return
    with runtime.lock, witness.scope.database():
        admission(runtime)
        witness.scope.validate()
        anchor_rows(witness, anchors)
        witness.scope.validate()
        admission(runtime)
