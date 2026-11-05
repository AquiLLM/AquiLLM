"""Known late graph authority and source guards retain only static reason labels."""

import pytest

from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.knowledge_graph.retrieval.graph_failure_observations_v2 import failure_diagnostics
from apps.knowledge_graph.retrieval.ready_scope import ReadyScopeError, ReadyScopeFailureReason


@pytest.mark.parametrize("error,kind,reason", [
    (ReadyScopeError(ReadyScopeFailureReason.READINESS_MISMATCH), "ready_scope_error", "readiness_mismatch"),
    (ReadyScopeError(ReadyScopeFailureReason.AUTHORIZATION_CONTEXT_INVALID), "ready_scope_error", "authorization_context_invalid"),
    (ValueError("captured authority changed"), "value_error", "captured_authority_changed"),
    (ValueError("authorization signature changed"), "value_error", "authorization_signature_changed"),
    (ValueError("materialization source identity changed"), "value_error", "materialization_source_changed"),
    (SourceScopeUnavailable("source_admission_seed_unavailable"), "source_scope_unavailable", "seed_unavailable"),
])
def test_known_native_guard_is_diagnosable_without_serializing_exception(error, kind, reason):
    observed = failure_diagnostics(error, "authority_revalidation", owner_valid=True)
    assert observed == {"graph_failure": {
        "version": "native-graph-failure-v1", "type": kind,
        "reason": reason, "phase": "authority_revalidation", "owner_valid": True,
    }}
