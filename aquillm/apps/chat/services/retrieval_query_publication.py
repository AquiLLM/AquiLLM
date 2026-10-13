"""Retain original query constraints through the existing publication lease."""
from dataclasses import asdict
from time import monotonic

from apps.collections.services.vocabulary_contracts import digest
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from lib.llm.turn_context import current_turn

from .retrieval_query_binding import admission, anchor_rows, original_binding


def retain_query_binding(packet, runtime):
    turn = current_turn()
    expected = hasattr(runtime, "_query_admission_attempt") or (
        turn is not None and id(runtime) in getattr(turn, "_query_admissions", {}))
    if not expected:
        return
    witness, bound = admission(runtime), original_binding(runtime)
    old = getattr(packet, "_publication_query_binding", None)
    if old is not None and (old is not bound or
        getattr(packet, "_publication_query_admission", None) is not witness):
        raise SourceScopeUnavailable("source_scope_unavailable")
    packet._publication_query_binding = bound
    packet._publication_query_admission = witness


def _packet_binding(packet):
    if packet.source_authorization is None:
        return None
    witness = getattr(packet, "_publication_query_admission", None)
    bound = getattr(packet, "_publication_query_binding", None)
    runtime = getattr(packet, "_publication_principal_runtime", None)
    turn = current_turn()
    expected = (runtime is not None and (hasattr(runtime, "_query_admission_attempt")
        or (turn is not None and id(runtime) in getattr(turn, "_query_admissions", {}))))
    if not expected and witness is None and bound is None:
        return None
    if (runtime is None or witness is not admission(runtime)
        or bound is not original_binding(runtime)
        or packet._publication_representation is not witness.representation):
        raise SourceScopeUnavailable("source_scope_unavailable")
    return witness, bound


def query_binding_digest(packet):
    # Pure retained-object identity for receipts and post-turn private traces.
    # Live authorization remains mandatory in _packet_binding / _unchanged.
    from .retrieval_query_binding import QueryAdmission
    from .retrieval_query_runtime import BoundProposal
    witness = getattr(packet, "_publication_query_admission", None)
    bound = getattr(packet, "_publication_query_binding", None)
    if witness is None and bound is None:
        return None
    if (type(witness) is not QueryAdmission or type(bound) is not BoundProposal
        or bound.runtime is not witness.runtime):
        raise SourceScopeUnavailable("source_scope_unavailable")
    from apps.documents.services.representation_store import representation_snapshot_digest
    return digest({"query": bound.query, "policy": bound.policy_key,
        "vocabulary": asdict(bound.vocabulary) if bound.vocabulary is not None else None,
        "anchors": asdict(bound.anchors) if bound.anchors is not None else None,
        "authorization": witness.authorization.authorization_context_signature,
        "representation": representation_snapshot_digest(witness.representation)})


def _publication_context(packet):
    value = _packet_binding(packet)
    if value is None:
        return None
    witness, bound = value
    turn = current_turn()
    context = getattr(turn, "answer_publication", None)
    lease = getattr(context, "lease", None)
    if (context is None or context.packet is not packet or context.turn is not turn
        or context.budget is not witness.budget or turn.budget is not witness.budget
        or context._query_binding is not bound or context._query_admission is not witness
        or context._authority is not packet.source_authorization):
        raise SourceScopeUnavailable("source_scope_unavailable")
    context._active()
    context._unchanged()  # Pure identity/digest checks; never calls authorize.
    if witness.budget._synthesis is None:
        if lease is not None or not witness.budget.can_publish():
            raise SourceScopeUnavailable("source_scope_unavailable")
    else:
        if (lease is None or lease is not witness.budget._synthesis
            or lease.budget is not witness.budget or lease.authority is not context._authority):
            raise SourceScopeUnavailable("source_scope_unavailable")
        lease.check_active()
    return context, witness, bound


def validate_packet_query_binding(packet):
    value = _publication_context(packet)
    if value is None:
        return
    context, witness, bound = value
    if witness.budget._synthesis is None:
        from .retrieval_query_runtime import validate_bound
        with witness.scope.database():
            validate_bound(bound)
        _publication_context(packet)
        return
    from apps.collections.services.vocabulary import _validate_vocabulary_metadata
    from apps.collections.services.vocabulary_database import _bounded_vocabulary_database
    from apps.collections.services.retrieval_authorization import revalidate_retrieval_authorization_context
    from apps.documents.services.representation_store import validate_representation_snapshot
    from .rag_publication_database import _bounded_database
    # No retrieval wrapper here: a sealed retrieval budget correctly has zero
    # remaining_ms. This exact existing publication lease owns only metadata work.
    seconds = min(context.remaining_seconds(), context.lease.remaining_seconds())
    if seconds <= 0:
        raise SourceScopeUnavailable("source_scope_unavailable")
    with _bounded_database(witness.scope.using, seconds):
        _publication_context(packet)
        with _bounded_vocabulary_database(witness.scope.using, monotonic() + seconds):
            if bound.vocabulary is not None:
                _validate_vocabulary_metadata(bound.vocabulary, authorization=witness.authorization)
            current = revalidate_retrieval_authorization_context(context=witness.authorization)
            if (frozenset(current.document_ids) != witness.authorization.selected_document_ids
                or frozenset(current.collection_ids) != witness.authorization.selected_collection_ids):
                raise SourceScopeUnavailable("source_scope_unavailable")
            if bound.anchors is not None:
                if bound.anchors.unresolved_references:
                    raise SourceScopeUnavailable("source_scope_unavailable")
                anchor_rows(witness, bound.anchors)
            validate_representation_snapshot(witness.representation, using=witness.scope.using)
        _publication_context(packet)
