"""Branch-local setup failures preserve independently completed siblings."""

import pytest

from apps.knowledge_graph.retrieval.branch_contracts import (
    BranchStatusV1,
    DirectBranchFailureReason,
    ExtendedBranchFailureReason,
)
from apps.knowledge_graph.retrieval.scheduler import (
    HybridGraphBranchScheduler,
    LocalBranchSchedulerFailure,
)
from apps.knowledge_graph.retrieval.topology.contracts import HybridBranchKind
from apps.knowledge_graph.tests.test_retrieval_branch_scheduler import (
    _Runtime,
    _settings,
)


@pytest.mark.parametrize("failed_kind", tuple(HybridBranchKind))
def test_typed_branch_setup_exception_preserves_completed_sibling(failed_kind) -> None:
    runtime = _Runtime()
    reason = (
        DirectBranchFailureReason.DIRECT_SEED_INVALID
        if failed_kind is HybridBranchKind.DIRECT
        else ExtendedBranchFailureReason.EXTENDED_SEED_INVALID
    )

    def local_failure(**_kwargs):
        raise LocalBranchSchedulerFailure(failed_kind, reason)

    if failed_kind is HybridBranchKind.DIRECT:
        runtime.run_direct = local_failure
    else:
        runtime.prepare_extended = local_failure
    outcome = HybridGraphBranchScheduler(runtime, clock=lambda: 100.0).run(
        query="q",
        baseline=object(),
        authorization=object(),
        settings=_settings(),
        deadline=101.0,
    )

    failed = (
        outcome.direct if failed_kind is HybridBranchKind.DIRECT else outcome.extended
    )
    sibling = (
        outcome.extended if failed_kind is HybridBranchKind.DIRECT else outcome.direct
    )
    assert failed.failure_reason is reason
    assert sibling.status is BranchStatusV1.SUCCEEDED
    assert outcome.shared_failure_reason is None
