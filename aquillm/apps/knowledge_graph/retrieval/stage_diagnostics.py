"""Fixed-label, bounded timings without request or graph data."""

from contextlib import contextmanager

import structlog

logger = structlog.stdlib.get_logger(__name__)
_BRANCHES = frozenset(("direct", "extended", "shared"))
_STAGES = frozenset(
    ("ontology", "extraction", "entity_resolution", "topology", "materialization")
)


@contextmanager
def graph_stage(*, branch, stage, clock):
    if type(branch) is not str or branch not in _BRANCHES:
        raise ValueError("invalid graph timing branch")
    if type(stage) is not str or stage not in _STAGES:
        raise ValueError("invalid graph timing stage")
    started = clock()
    try:
        yield
    finally:
        logger.info(
            "obs.rag.graph_stage",
            branch=branch,
            stage=stage,
            elapsed_ms=max(0, min(5000, int((clock() - started) * 1000))),
        )
