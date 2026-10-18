"""Small real scope states and exact request/leaf inputs for v3 tests."""

from apps.knowledge_graph.projection.frontier_generation_batch_v2 import attest_all
from apps.knowledge_graph.projection.frontier_scope_complete_v3 import (
    ScopeCompletePreparation,
)
from apps.knowledge_graph.retrieval.topology.frontier_binary_codec_v2 import (
    encode_binary_response,
)
from apps.knowledge_graph.retrieval.topology.frontier_scope_window_contracts_v3 import (
    encode_request,
)
from apps.knowledge_graph.retrieval.topology.frontier_window_budget_v2 import (
    preview_window,
)
from apps.knowledge_graph.retrieval.topology.frontier_window_checkpoint_v2 import (
    snapshot,
)
from apps.knowledge_graph.retrieval.topology.gateway_contracts import _canonical
from apps.knowledge_graph.tests.frontier_generation_batch_support import (
    GenerationFixture,
)


def scope_state(*, stage="scope_directory", generations=1, documents=10):
    fixture = GenerationFixture(generations, documents)
    reads = fixture.session.reads
    reads.reserve_final(multi=generations > 1)
    if generations > 1:
        attest_all(reads, fixture.session.selected)
    else:
        reads.attest(next(iter(fixture.session.selected.values())))
    sources = {
        g: [
            s.collection_key,
            *(
                d.document_key
                for d in fixture.ready.authorized_documents
                if d.generation_key == g
            ),
        ]
        for g, s in fixture.session.selected.items()
    }
    state = ScopeCompletePreparation(reads, sources)
    while (effect := state.next_read()) is not None and state.stage != stage:
        state.apply_result(reads.read_batch(effect.kind, effect.pairs))
    assert state.stage == stage and effect is not None
    return fixture, state


def request_body(fixture, state, *, single=False):
    operation, checkpoint = snapshot(state)
    return encode_request(
        fixture.ready,
        operation,
        checkpoint,
        preview_window(fixture.budget),
        1,
        fixture.budget.check_active,
        pairs=state.effect.pairs if single else None,
    )


def leaf(rows=1, *, padding=0):
    row = {
        "generation_key": "a" * 64,
        "lookup_key": "b" * 64,
        "record": {"padding": "x" * padding},
    }
    return encode_binary_response(
        _canonical({"rows": [row] * rows, "scanned_rows": rows, "queries": 1})
    )
