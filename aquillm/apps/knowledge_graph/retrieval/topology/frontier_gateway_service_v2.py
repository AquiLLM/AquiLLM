"""One shared occupied worker per fixed frontier read, including wire encoding."""

import asyncio
from hmac import compare_digest

from apps.knowledge_graph.projection.frontier_reads_v2 import QUERIES
from apps.knowledge_graph.projection.memgraph_driver import MemgraphDriverError
from apps.knowledge_graph.projection.topology_adapter import _failure

from . import gateway_service as v1
from .failures import TopologyLoadError
from .frontier_gateway_contracts_v2 import (
    CHECKSUM,
    VERSION,
    FrontierCompletedBudgetFailureV2,
    decode_request,
)
from .gateway_contracts import FAILURE_HTTP_STATUS, _canonical


async def respond(send, status, payload):
    await v1._respond(
        send, status, _canonical(payload), version=VERSION, checksum=CHECKSUM
    )


def read_payload(driver, request, deadline, maximum):
    remaining = deadline - v1.monotonic()
    if remaining <= 0:
        raise TimeoutError
    parameters = {"generation_key": request.generation}
    if request.kind == "index":
        parameters = {"index_key": request.keys[0]}
    elif request.kind in {"raw", "index_batch"}:
        parameters["keys_csv"] = ",".join(request.keys)
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
    payload = _canonical({"rows": rows, "scanned_rows": len(rows), "queries": 1})
    if len(payload) > min(maximum, request.response_bytes):
        raise FrontierCompletedBudgetFailureV2(len(rows))
    if v1.monotonic() >= deadline:
        raise TimeoutError
    return payload


async def topology_frontier(scope, receive, send):
    started = v1.monotonic()
    try:
        settings = v1.load_topology_gateway_settings(v1.environ)
    except Exception:
        await respond(send, 503, {"reason": "unavailable"})
        return
    expected = b"Bearer " + settings.bearer_token.get_secret_value().encode("ascii")
    if not compare_digest(v1._single_header(scope, b"authorization") or b"", expected):
        await respond(send, 401, {"reason": "authentication"})
        return
    length = v1._length(scope, settings.max_request_bytes)
    if length is not None and length > settings.max_request_bytes:
        await respond(send, 413, {"reason": "request_too_large"})
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
            or not v1._wire_valid(scope, version=VERSION, checksum=CHECKSUM)
        ):
            raise ValueError
        if v1._header_values(scope, b"content-encoding"):
            raise ValueError
        deadline = started + min(timeout_ms, settings.timeout_ms) / 1000
        body = await asyncio.wait_for(
            v1._read_body(receive, length), max(0, deadline - v1.monotonic())
        )
        request = decode_request(body)
    except TimeoutError:
        await respond(send, 504, {"reason": "deadline"})
        return
    except (ValueError, TypeError, KeyError):
        await respond(send, 400, {"reason": "malformed_request"})
        return
    try:
        runtime = v1._get_runtime(settings)
        payload = await v1.GATEWAY_WORKERS.run(
            read_payload,
            runtime.driver,
            request,
            deadline,
            settings.max_response_bytes,
            expires=deadline,
            clock=v1.monotonic,
        )
    except FrontierCompletedBudgetFailureV2 as exc:
        await respond(
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
        await respond(send, 504, {"reason": "deadline"})
        return
    except TopologyLoadError as exc:
        reason = v1._mapped_reason(exc)
        await respond(send, FAILURE_HTTP_STATUS[reason], {"reason": reason.value})
        return
    except Exception:
        await respond(send, 503, {"reason": "unavailable"})
        return
    await v1._respond(send, 200, payload, version=VERSION, checksum=CHECKSUM)
