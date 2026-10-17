"""One shared occupied worker per fixed frontier read, including wire encoding."""

import asyncio
from functools import partial
from hmac import compare_digest

from apps.knowledge_graph.projection.frontier_reads_v2 import QUERIES
from apps.knowledge_graph.projection.memgraph_driver import MemgraphDriverError
from apps.knowledge_graph.projection.topology_adapter import _failure

from . import gateway_service as v1
from .failures import TopologyLoadError
from .frontier_gateway_contracts_v2 import (
    BATCH_CHECKSUM,
    BATCH_VERSION,
    BINARY_CHECKSUM,
    BINARY_VERSION,
    CHECKSUM,
    VERSION,
    FrontierCompletedBudgetFailureV2,
    FrontierGatewayBatchRequestV2,
    decode_batch_request,
    decode_request,
)
from .frontier_scope_window_contracts_v3 import ScopeRawRequest, protocol_headers_valid
from .frontier_window_contracts_v2 import CompletedWindowBudgetFailure
from .gateway_contracts import FAILURE_HTTP_STATUS, _canonical


async def respond(send, status, payload, *, version=VERSION, checksum=CHECKSUM):
    await v1._respond(
        send, status, _canonical(payload), version=version, checksum=checksum
    )


def read_payload(
    driver, request, deadline, maximum, *, binary=False, expanded_maximum=None
):
    remaining = deadline - v1.monotonic()
    if remaining <= 0:
        raise TimeoutError
    parameters = {}
    batch = type(request) in {FrontierGatewayBatchRequestV2, ScopeRawRequest}
    if batch:
        parameters = {"lookups_csv": ",".join(g + ":" + k for g, k in request.lookups)}
    elif request.kind == "index":
        parameters = {"index_key": request.keys[0]}
    elif request.kind in {"raw", "index_batch"}:
        parameters["generation_key"] = request.generation
        parameters["keys_csv"] = ",".join(request.keys)
    else:
        parameters = {"generation_key": request.generation}
    try:
        rows = driver.execute_read(
            QUERIES[request.kind],
            parameters,
            timeout_seconds=float(remaining),
            max_records=request.maximum,
        )
    except MemgraphDriverError as exc:
        if exc.code == "memgraph_timeout":
            raise TimeoutError from exc
        raise TopologyLoadError(_failure(exc.code)) from exc
    if type(rows) is not tuple or len(rows) > request.maximum:
        raise ValueError("frontier backend shape invalid")
    if type(request) is ScopeRawRequest and len(rows) == 320:
        raise FrontierCompletedBudgetFailureV2(320)
    if batch:
        rows = tuple(sorted(rows, key=_canonical))
    payload = _canonical({"rows": rows, "scanned_rows": len(rows), "queries": 1})
    if batch:
        from .frontier_gateway_codec_v2 import MAX_EXPANDED_BYTES, encode_batch_response

        expanded_limit = maximum if expanded_maximum is None else expanded_maximum
        if len(payload) > min(expanded_limit, MAX_EXPANDED_BYTES):
            raise FrontierCompletedBudgetFailureV2(len(rows))
        if v1.monotonic() >= deadline:
            raise TimeoutError
        if binary:
            from .frontier_binary_codec_v2 import encode_binary_response

            payload = encode_binary_response(payload)
        else:
            payload = encode_batch_response(payload)
    if len(payload) > min(maximum, request.response_bytes):
        raise FrontierCompletedBudgetFailureV2(len(rows))
    if v1.monotonic() >= deadline:
        raise TimeoutError
    return payload


async def topology_frontier(
    scope,
    receive,
    send,
    *,
    batch=False,
    binary=False,
    window=False,
    complete_scope=False,
):
    batch = batch or binary
    version, checksum = (
        (BATCH_VERSION, BATCH_CHECKSUM) if batch else (VERSION, CHECKSUM)
    )
    if binary:
        version, checksum = BINARY_VERSION, BINARY_CHECKSUM
    if window:
        from .frontier_window_contracts_v2 import CHECKSUM as WINDOW_CHECKSUM
        from .frontier_window_contracts_v2 import VERSION as WINDOW_VERSION

        version, checksum = WINDOW_VERSION, WINDOW_CHECKSUM
    if complete_scope:
        from .frontier_scope_window_contracts_v3 import CHECKSUM as checksum
        from .frontier_scope_window_contracts_v3 import VERSION as version

        window = True
    reply = partial(respond, version=version, checksum=checksum)
    started = v1.monotonic()
    try:
        settings = v1.load_topology_gateway_settings(v1.environ)
    except Exception:
        await reply(send, 503, {"reason": "unavailable"})
        return
    expected = b"Bearer " + settings.bearer_token.get_secret_value().encode("ascii")
    if not compare_digest(v1._single_header(scope, b"authorization") or b"", expected):
        await reply(send, 401, {"reason": "authentication"})
        return
    length = v1._length(scope, settings.max_request_bytes)
    if length is not None and length > settings.max_request_bytes:
        await reply(send, 413, {"reason": "request_too_large"})
        return
    raw = v1._single_header(scope, b"x-frontier-timeout-ms")
    try:
        if (
            raw is None
            or not raw.isdigit()
            or len(raw) > 5
            or str(int(raw)).encode() != raw
        ):
            raise ValueError
        timeout_ms = int(raw)
        if (
            not 1 <= timeout_ms <= 60000
            or length is None
            or not v1._wire_valid(scope, version=version, checksum=checksum)
        ):
            raise ValueError
        if v1._header_values(scope, b"content-encoding"):
            raise ValueError
        if complete_scope and not protocol_headers_valid(
            (k.decode("latin1") for k, _ in scope.get("headers", ())), request=True
        ):
            raise ValueError
        deadline = started + min(timeout_ms, settings.timeout_ms) / 1000
        body = await asyncio.wait_for(
            v1._read_body(receive, length), max(0, deadline - v1.monotonic())
        )
        if window:
            from .frontier_window_contracts_v2 import (
                decode_request as decode_window_request,
            )

            if complete_scope:
                from .frontier_scope_window_contracts_v3 import (
                    decode_request as decode_scope,
                )

                def check():
                    if v1.monotonic() >= deadline:
                        raise TimeoutError

                request = decode_scope(body, check)
            else:
                request = decode_window_request(body)
        else:
            request = (decode_batch_request if batch else decode_request)(body)
    except TimeoutError:
        await reply(send, 504, {"reason": "deadline"})
        return
    except (ValueError, TypeError, KeyError, IndexError):
        await reply(send, 400, {"reason": "malformed_request"})
        return
    try:
        runtime = v1._get_runtime(settings)
        runner = partial(read_payload, binary=binary)
        if window:
            from .frontier_window_worker_v2 import read_window

            runner = partial(read_window, complete_scope=complete_scope)

        payload = await v1.GATEWAY_WORKERS.run(
            runner,
            runtime.driver,
            request,
            deadline,
            settings.max_response_bytes,
            expires=deadline,
            clock=v1.monotonic,
        )
    except CompletedWindowBudgetFailure as exc:
        receipt = _canonical(
            {
                "reason": "frontier_budget",
                "request_digest": exc.request_digest,
                "queries": exc.queries,
                "scanned_rows": exc.scanned_rows,
            }
        )
        if len(receipt) > min(request.grant.wire, settings.max_response_bytes):
            await v1._respond(send, 503, b"", version=version, checksum=checksum)
        else:
            await v1._respond(send, 422, receipt, version=version, checksum=checksum)
        return
    except FrontierCompletedBudgetFailureV2 as exc:
        await reply(
            send,
            422,
            {
                "reason": "frontier_budget",
                "scanned_rows": exc.scanned_rows,
                "queries": 1,
            },
        )
        return
    except TimeoutError:
        await reply(send, 504, {"reason": "deadline"})
        return
    except TopologyLoadError as exc:
        reason = v1._mapped_reason(exc)
        await reply(send, FAILURE_HTTP_STATUS[reason], {"reason": reason.value})
        return
    except Exception:
        await reply(send, 503, {"reason": "unavailable"})
        return
    await v1._respond(
        send,
        200,
        payload,
        version=version,
        checksum=checksum,
        content_type=b"application/octet-stream"
        if binary or window
        else b"application/json",
    )
