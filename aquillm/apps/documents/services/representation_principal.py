"""Original turn identity is retained, never recreated by acquisition or sinks."""
from dataclasses import dataclass

from lib.llm.turn_context import current_turn
from .representation_contracts import SourceScopeUnavailable


@dataclass(frozen=True, eq=False)
class OriginalPrincipal:
    consumer: object
    principal: object
    principal_pk: object
    turn: object
    budget: object
    collection_ref: object
    requested_collection_ids: tuple | None


def capture_principal(consumer, turn):
    from .representation_unavailable_parent import normalized_selection
    return OriginalPrincipal(consumer, consumer.user, consumer.user.pk, turn, turn.budget,
                             consumer.col_ref, normalized_selection(consumer.col_ref.collections))


def validate_principal(binding, *, turn=None, principal=None):
    turn = turn or current_turn()
    if (type(binding) is not OriginalPrincipal or turn is None
        or binding.turn is not turn or binding.budget is not turn.budget
        or getattr(turn, "_media_principal_binding", None) is not binding
        or binding.consumer.user is not binding.principal
        or binding.principal.pk != binding.principal_pk
        or (principal is not None and principal is not binding.principal)):
        raise SourceScopeUnavailable("source_scope_unavailable")
    turn.budget.check_active()
    return binding


def runtime_principal(runtime, *, turn=None, principal=None, required=False):
    turn = turn or current_turn()
    binding = getattr(runtime, "_media_principal_binding", None)
    if binding is None:
        # Genuine standalone public text runtimes predate turn initialization.
        if (required or hasattr(runtime, "_media_consumer")
            or getattr(turn, "_media_principal_binding", None) is not None):
            raise SourceScopeUnavailable("source_scope_unavailable")
        return None
    validate_principal(binding, turn=turn, principal=principal)
    authority = runtime.authorization
    if (runtime.budget is not binding.budget
        or getattr(runtime, "_media_consumer", None) is not binding.consumer
        or (authority is not None and
            authority.reauthorization_capability._principal is not binding.principal)):
        raise SourceScopeUnavailable("source_scope_unavailable")
    return binding


def retain_packet_principal(packet, runtime):
    binding = runtime_principal(runtime)
    old = getattr(packet, "_publication_principal_binding", None)
    if old is not None and old is not binding:
        raise SourceScopeUnavailable("source_scope_unavailable")
    packet._publication_principal_binding = binding
    packet._publication_principal_runtime = runtime if binding is not None else None
    from .representation_manual_scope import registered_child_record
    packet._publication_manual_record = registered_child_record(runtime)


def validate_packet_principal(packet):
    binding = getattr(packet, "_publication_principal_binding", None)
    runtime = getattr(packet, "_publication_principal_runtime", None)
    if binding is not None:
        if runtime_principal(runtime, required=True) is not binding:
            raise SourceScopeUnavailable("source_scope_unavailable")
    elif packet.source_authorization is not None and (
        runtime is not None or getattr(current_turn(), "_media_principal_binding", None) is not None
    ):
        raise SourceScopeUnavailable("source_scope_unavailable")
    from .representation_manual_scope import packet_media_runtime
    packet_media_runtime(packet)
