"""An unavailable original selection owns a lifetime, never retrieval authority."""
from dataclasses import dataclass, field
from threading import RLock

from lib.llm.turn_context import current_turn
from .representation_contracts import SourceScopeUnavailable


def normalized_selection(values):
    if type(values) not in (list, tuple, frozenset):
        return None
    result = []
    for value in values:
        if type(value) is str and value.isascii() and value.isdecimal():
            value = int(value)
        if type(value) is not int or value < 1:
            return None
        result.append(value)
    return tuple(sorted(set(result))) if len(result) <= 128 else None


@dataclass(frozen=True, eq=False)
class UnavailableAcquisitionParent:
    turn: object
    budget: object
    principal: object
    principal_pk: int | None
    _media_consumer: object
    _media_col_ref: object
    _media_collections: tuple
    normalized_collection_ids: tuple | None
    _media_principal_binding: object
    authorization: None = field(default=None, init=False)
    _media_authorization: None = field(default=None, init=False)
    cache: dict = field(default_factory=dict, repr=False)
    windows: dict = field(default_factory=dict, repr=False)
    lock: object = field(default_factory=RLock, repr=False)
    observation: dict = field(default_factory=dict, repr=False)


def unavailable_parent(consumer, turn, binding):
    return UnavailableAcquisitionParent(turn, turn.budget, consumer.user, consumer.user.pk,
        consumer, consumer.col_ref, tuple(consumer.col_ref.collections),
        normalized_selection(consumer.col_ref.collections), binding)


def validate_unavailable_parent(parent, *, turn=None, principal=None):
    turn = turn or current_turn()
    if (type(parent) is not UnavailableAcquisitionParent or turn is None
        or parent.turn is not turn or parent.budget is not turn.budget
        or getattr(turn, "_media_source_runtime", None) is not parent
        or parent._media_consumer.col_ref is not parent._media_col_ref
        or parent.normalized_collection_ids is None
        or normalized_selection(parent._media_col_ref.collections) != parent.normalized_collection_ids
        or parent._media_consumer.user is not parent.principal
        or parent.principal.pk != parent.principal_pk
        or (principal is not None and principal is not parent.principal)):
        raise SourceScopeUnavailable("manual_source_scope_unavailable")
    turn.budget.check_active()
    return parent
