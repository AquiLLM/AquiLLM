"""One occupied worker runs a bounded physical prefix of a finite preparation."""

from apps.knowledge_graph.projection.frontier_preparation_bytes_v2 import (
    binary_bound,
    expanded_bound,
)
from apps.knowledge_graph.projection.frontier_preparation_reads_v2 import OwnerBoundary
from apps.knowledge_graph.projection.frontier_reads_v2 import FrontierReadsV2

from . import gateway_service as v1
from .frontier_binary_codec_v2 import decode_binary_response
from .frontier_contracts_v2 import FrontierBudgetExceeded
from .frontier_gateway_contracts_v2 import (
    FrontierCompletedBudgetFailureV2,
    FrontierGatewayBatchRequestV2,
)
from .frontier_gateway_service_v2 import read_payload
from .frontier_scope_window_contracts_v3 import ScopeRawRequest, single_pairs
from .frontier_window_checkpoint_v2 import restore
from .frontier_window_codec_v2 import KINDS, MAGIC
from .frontier_window_contracts_v2 import CompletedWindowBudgetFailure


class PlanningCursor:
    """Numeric pre-hold allowance only; cannot reserve or authorize any read."""

    def __init__(self, rows, deadline):
        self.rows, self.deadline = rows, deadline

    def check_active(self):
        if self.clock() >= self.deadline:
            raise TimeoutError("frontier original request expiry")

    def available_read_rows(self):
        self.check_active()
        return self.rows

    @staticmethod
    def clock():
        return v1.monotonic()


def read_window(driver, request, deadline, maximum, *, complete_scope=False):
    kinds, magic = KINDS, MAGIC
    single = complete_scope and request.mode == "single"
    if complete_scope:
        from .frontier_scope_window_codec_v3 import KINDS as kinds
        from .frontier_scope_window_codec_v3 import MAGIC as magic

    grant = request.grant
    cursor = PlanningCursor(grant.planning_rows, deadline)
    reads = FrontierReadsV2(None, ready=request.ready, budget=cursor, deadline=deadline)
    state = (
        None
        if single
        else restore(
            reads, request.operation, request.checkpoint, complete_scope=complete_scope
        )
    )
    frames, queries, scanned, raw_count, expanded, wire = [], 0, 0, 0, 0, 41
    while queries < (1 if single else grant.queries):
        cursor.check_active()
        try:
            if single:
                from apps.knowledge_graph.projection.frontier_scope_complete_v3 import (
                    ScopeCompleteRead,
                )

                effect = ScopeCompleteRead(
                    single_pairs(request.ready, request.operation)
                )
                if cursor.rows < 320:
                    raise FrontierBudgetExceeded("frontier physical work exhausted")
            else:
                effect = state.next_read(cursor.rows)
        except (OwnerBoundary, FrontierBudgetExceeded):
            break
        if effect is None:
            break
        raw = effect.kind in {"raw_batch_multi", "scope_raw_complete_batch"}
        ordinary_rows = (
            len(effect.pairs)
            if not raw or single
            else sum((g, key) in effect.pairs for g, _, key in state.expected)
        )
        if (
            (raw and raw_count >= 4)
            or scanned + effect.maximum > grant.rows
            or scanned + ordinary_rows > 320
        ):
            break
        remaining_expanded = 1048576 - expanded
        remaining_wire = min(maximum, grant.wire) - wire - 5
        bound = None if state is None else expanded_bound(state, effect)
        # A valid completed prefix is returned before a subsequent known-size
        # index read can exhaust this container. Replay chooses the same pairs;
        # the next original-owner window continues the unmodified effect.
        if frames and (
            remaining_wire <= 0
            or (
                bound is not None
                and (bound > remaining_expanded or binary_bound(bound) > remaining_wire)
            )
        ):
            break
        physical_request = (
            ScopeRawRequest(effect.pairs, request.ready)
            if effect.kind == "scope_raw_complete_batch"
            else FrontierGatewayBatchRequestV2(
                effect.kind, effect.pairs, request.ready, max(1, remaining_wire)
            )
        )
        try:
            frame = read_payload(
                driver,
                physical_request,
                deadline,
                max(0, remaining_wire),
                binary=True,
                expanded_maximum=remaining_expanded,
            )
        except FrontierCompletedBudgetFailureV2 as exc:
            raise CompletedWindowBudgetFailure(
                request.request_digest, queries + 1, scanned + exc.scanned_rows
            ) from exc
        payload = decode_binary_response(frame)
        rows = tuple(payload["rows"])
        queries += 1
        scanned += len(rows)
        raw_count += raw
        cursor.rows -= len(rows)
        expanded += int.from_bytes(frame[8:12], "big")
        wire += 5 + len(frame)
        if scanned > 320 or expanded > 1048576 or wire > min(maximum, grant.wire):
            raise CompletedWindowBudgetFailure(request.request_digest, queries, scanned)
        if state is not None:
            state.apply_result(rows)
        frames.append(
            bytes([kinds[effect.kind]]) + len(frame).to_bytes(4, "big") + frame
        )
    cursor.check_active()
    if not frames:
        raise ValueError("frontier window did not complete a physical read")
    return (
        magic
        + bytes.fromhex(request.request_digest)
        + bytes([len(frames)])
        + b"".join(frames)
    )
