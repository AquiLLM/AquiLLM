"""Payload-free topology rejection boundaries; public failure contracts stay fixed."""

from enum import StrEnum
from time import monotonic

import structlog

from lib.retrieval_redaction import (
    MAX_RETRIEVAL_LOG_ELAPSED_MS,
    RetrievalLogReason,
    retrieval_log_fields,
)

logger = structlog.stdlib.get_logger(__name__)


class TopologyDiagnosticPhase(StrEnum):
    SOURCE_FAMILY_CAP = "source_family_cap"
    FAMILY_SCHEMA = "family_schema"
    EMPTY_FRONTIER = "empty_frontier"
    SNAPSHOT_BUILD = "snapshot_build"
    SNAPSHOT_COMPOSE = "snapshot_compose"


_FAMILIES = frozenset(
    {
        "none",
        "ProjectedEntity",
        "AutomaticMembership",
        "ProjectedDocument",
        "ProjectedChunk",
        "ProjectedRelationSemantics",
        "ProjectedRelation",
        "ProjectedEvidence",
        "ProjectedEntityMention",
        "ArtifactProvenance",
    }
)


def record_topology_failure(
    *, phase, started, family="none", branch="unscoped", count=0, maximum=0
):
    # All fields are closed labels or bounded numeric observations. No exception
    # object/message, query, backend row, identity, or request can enter this API.
    if (
        type(phase) is not TopologyDiagnosticPhase
        or family not in _FAMILIES
        or branch not in {"direct", "extended", "unscoped"}
        or type(maximum) is not int
        or not 0 <= maximum <= 50_000
    ):
        raise ValueError("invalid topology diagnostic labels")
    fields = retrieval_log_fields(
        reason=(
            RetrievalLogReason.PAYLOAD_TOO_LARGE
            if phase is TopologyDiagnosticPhase.SOURCE_FAMILY_CAP
            else RetrievalLogReason.INVALID_REQUEST
        ),
        count=count,
        elapsed_ms=min(
            MAX_RETRIEVAL_LOG_ELAPSED_MS, max(0.0, (monotonic() - started) * 1000)
        ),
    )
    try:
        logger.info(
            "obs.rag.topology_rejected",
            phase=phase.value,
            family=family,
            branch=branch,
            maximum=maximum,
            **fields,
        )
    except Exception:
        # Logging cannot replace the rejection or turn an invalid result valid.
        pass
