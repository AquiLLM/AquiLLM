"""Private accepted-action origins for optional native baseline recovery.

Origin affects work admission only. Current source authorization and exact
scoring remain the ordinary final preparation path's responsibility.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from hashlib import sha256

from apps.documents.services.native_search_schedule import scoped_baseline_recovery_enabled
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.source_loading import current_source_runtime
from lib.llm.turn_context import current_turn
from lib.retrieval.evidence import SourceEvidence
from lib.retrieval.operation import Operation, _OPERATION

_ACTION = ContextVar("native_recovery_origin_action", default=None)


def _identity(source):
    return source.chunk_id, source.document_id, source.chunk_number, source.source_fingerprint


def _payload_rows(result):
    from .rag_retrieval import _PUBLIC_ROW_KEYS, _verified_row_coordinates

    rows = tuple(dict((k, v) for k, v in row.items() if k in _PUBLIC_ROW_KEYS)
                 for row in result.get("result", ()) if isinstance(row, dict))
    if any(_verified_row_coordinates(row) is None for row in rows):
        raise SourceScopeUnavailable("recovery_result_changed")
    return rows


@dataclass(frozen=True, slots=True)
class _Occurrence:
    result: object
    rows: tuple
    sources: tuple
    ordinary: tuple


@dataclass(frozen=True, slots=True)
class _Origins:
    runtime: object
    budget: object
    authorization: object
    turn: object
    query: str
    frozen: object = None
    representation: object = None
    dense: object = None
    occurrences: list = field(default_factory=list)

    def check(self):
        # TurnBudget.can_publish delegates to recursive Operation.active.
        # Inspect the same ledger under its lock, with bounded callback fences.
        with self.budget._lock:
            identity, check = capture_operation_check(_OPERATION.get(), self.budget)
            self.check_owner()
            check()
            now = self.budget._clock()
            identity()
            self.check_owner()
            if now >= self.budget._deadline:
                raise SourceScopeUnavailable("recovery_original_scope_unavailable")

    def check_owner(self):
        """Identity-only fence after every callback-capable clock sample."""
        runtime, turn = self.runtime, current_turn()
        frozen = getattr(runtime, "_frozen_retrieval_scope", None)
        if (current_source_runtime() is not runtime or runtime.budget is not self.budget
                or runtime.authorization is not self.authorization or turn is not self.turn
                or (turn is not None and turn.budget is not self.budget)
                or self.budget.stop_reason is not None
                or (self.frozen is not None and (frozen is not self.frozen
                    or frozen.representation is not self.representation or frozen.dense is not self.dense))):
            raise SourceScopeUnavailable("recovery_original_scope_unavailable")


@dataclass(slots=True)
class _Action:
    origins: _Origins
    operation: object
    check_operation: object
    native: list = field(default_factory=list)
    accepted: bool = False


def capture_operation_check(operation, budget):
    """Bound ancestry before callbacks; never recurse through mutable parents."""
    chain, seen, owner = [], set(), operation
    while owner is not None:
        if type(owner) is not Operation or id(owner) in seen or len(chain) >= 64:
            raise SourceScopeUnavailable("recovery_original_scope_unavailable")
        seen.add(id(owner))
        chain.append((owner, owner.parent, owner.deadline, owner.clock, owner.budget))
        owner = owner.parent

    def identity():
        if (_OPERATION.get() is not operation or any(
                not op.live or op.parent is not parent or op.deadline != deadline
                or op.clock is not clock or op.budget is not original or original is not budget
                for op,parent,deadline,clock,original in chain)):
            raise SourceScopeUnavailable("recovery_original_scope_unavailable")

    def check():
        identity()
        for _,_,deadline,clock,_ in chain:
            if deadline is not None:
                now = clock()
                identity()
                if now >= deadline:
                    raise SourceScopeUnavailable("recovery_original_scope_unavailable")

    return identity, check


def _check_action(action):
    if action.accepted or _ACTION.get() is not action:
        raise SourceScopeUnavailable("recovery_action_unavailable")
    action.check_operation()
    action.origins.check_owner()
    if _ACTION.get() is not action:
        raise SourceScopeUnavailable("recovery_action_unavailable")


def start_recovery_origins(question, budget, initial_results=(), evidence_views=None):
    runtime = current_source_runtime()
    if runtime is None or not scoped_baseline_recovery_enabled():
        return None
    with runtime.lock:
        prior = getattr(runtime, "_recovery_origins", None)
        runtime._recovery_origins = None
        if prior is not None:
            return None
        record = _Origins(runtime, budget, runtime.authorization, current_turn(),
                          sha256(question.encode()).hexdigest())
        record.check()
        runtime._recovery_origins = record
        # These inputs are already accepted ordinary obligations. They cannot
        # create a recovery-only origin or a second vote for the same payload.
        for result in initial_results:
            if not any(old.result is result for old in record.occurrences):
                _accept_occurrence(record, result, evidence_views([result]), None)
        return record


@contextmanager
def recovery_action(origins):
    if origins is None:
        yield None
        return
    operation = _OPERATION.get()
    _, check = capture_operation_check(operation, origins.budget)
    origins.check()
    check()
    if operation is None:
        raise SourceScopeUnavailable("recovery_action_unavailable")
    action = _Action(origins, operation, check)
    token = _ACTION.set(action)
    try:
        yield action
    finally:
        _ACTION.reset(token)


def record_native_recovery(captured, rows, ordinary_ids):
    """Pending only: a native result may finish after its caller is cancelled."""
    action = _ACTION.get()
    if action is None:
        return
    origins = action.origins
    origins.check()
    _check_action(action)
    if (captured.scope.runtime is not origins.runtime
            or captured.operation is not action.operation):
        raise SourceScopeUnavailable("recovery_action_unavailable")
    from lib.retrieval.evidence import fingerprint_source

    values = tuple((row.pk, str(row.doc_id), row.chunk_number, fingerprint_source(row.content))
                   for row in rows)
    action.native.append((values, tuple(value for value in values if value[0] in ordinary_ids)))


def accept_recovery_action(action, result, sources):
    if action is None:
        return
    origins = action.origins
    origins.check()
    _check_action(action)
    _accept_occurrence(origins, result, sources, action.native, action)
    action.accepted = True


def _accept_occurrence(origins, result, sources, native, action=None):
    from .rag_retrieval import _verified_row_coordinates

    rows, sources = _payload_rows(result), tuple(sources)
    if any(type(source) is not SourceEvidence for source in sources):
        raise SourceScopeUnavailable("recovery_result_changed")
    identities = tuple(_identity(source) for source in sources)
    if (len({item[0] for item in identities}) != len(identities)
            or {_verified_row_coordinates(row)[:3] for row in rows} != {v[:3] for v in identities}):
        raise SourceScopeUnavailable("recovery_result_changed")
    ordinary = identities
    if native:
        if len(native) != 1 or native[0][0] != identities:
            raise SourceScopeUnavailable("recovery_result_changed")
        ordinary = native[0][1]
    runtime = origins.runtime
    with runtime.lock:
        origins.check()
        if action is not None:
            _check_action(action)
        frozen = getattr(runtime, "_frozen_retrieval_scope", None)
        if (getattr(runtime, "_recovery_origins", None) is not origins
                or frozen is None or len(origins.occurrences) >= origins.budget.limits.actions):
            raise SourceScopeUnavailable("recovery_original_scope_unavailable")
        if origins.frozen is None:
            for name, value in (("frozen", frozen), ("representation", frozen.representation), ("dense", frozen.dense)):
                object.__setattr__(origins, name, value)
        if not any(old.result is result for old in origins.occurrences):
            origins.occurrences.append(_Occurrence(result, rows, sources, ordinary))


@dataclass(frozen=True, slots=True)
class RecoveryOrder:
    origins: _Origins
    ordinary_ids: tuple
    optional_ids: tuple
    identities: tuple
    ordinary_scores: tuple

    def check(self):
        self.origins.check()


def take_recovery_order(pool, question, authorization):
    """Use accepted occurrences once; duplicated public payloads add no weight."""
    runtime = current_source_runtime()
    if runtime is None:
        return None
    with runtime.lock:
        origins = getattr(runtime, "_recovery_origins", None)
        runtime._recovery_origins = None
        if not scoped_baseline_recovery_enabled() or type(origins) is not _Origins:
            return None
        origins.check()
        if (authorization is not origins.authorization
                or origins.query != sha256(question.encode()).hexdigest()):
            raise SourceScopeUnavailable("recovery_original_scope_unavailable")
        sources, scores, first = {}, {}, {}
        for occurrence in origins.occurrences:
            if _payload_rows(occurrence.result) != occurrence.rows:
                raise SourceScopeUnavailable("recovery_result_changed")
            for source in occurrence.sources:
                identity = _identity(source)
                if source.chunk_id in sources and sources[source.chunk_id] != identity:
                    raise SourceScopeUnavailable("recovery_source_changed")
                sources[source.chunk_id] = identity
            for rank, identity in enumerate(occurrence.ordinary, 1):
                pk = identity[0]
                first.setdefault(pk, len(first))
                scores[pk] = scores.get(pk, 0) + 1 / (60 + rank)
        ids = tuple(row.get("chunk_id", row.get("i")) for row in pool.rows)
        for row in pool.rows:
            pk = row.get("chunk_id", row.get("i"))
            if pk not in sources or not any(row == original for occurrence in origins.occurrences
                                            for original in occurrence.rows):
                raise SourceScopeUnavailable("recovery_result_changed")
        optional = tuple(pk for pk in ids if pk not in scores)
        if not optional or not scores:
            return None
        ordinary = tuple(pk for pk in sorted(scores, key=lambda pk: (-scores[pk], first[pk])) if pk in ids)
        citations = {row.get("chunk_id", row.get("i")): row.get("citation", row.get("ref")) for row in pool.rows}
        return RecoveryOrder(origins, ordinary, optional, tuple(sources[pk] for pk in ids),
                             tuple((citations[pk], scores[pk]) for pk in ordinary))
