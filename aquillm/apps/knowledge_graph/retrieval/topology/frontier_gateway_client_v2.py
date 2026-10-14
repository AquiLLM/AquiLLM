"""Zero-retry HTTP transport charging actual received response-body bytes."""

from math import floor
from urllib.error import HTTPError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import ProxyHandler, Request

from .frontier_contracts_v2 import FrontierBudgetExceeded
from .frontier_gateway_contracts_v2 import (
    BATCH_CHECKSUM,
    BATCH_PATH,
    BATCH_VERSION,
    CHECKSUM,
    PATH,
    VERSION,
    FrontierCompletedBudgetFailureV2,
    FrontierGatewayBatchRequestV2,
    FrontierGatewayRequestV2,
    encode_request,
)
from .gateway_client import (
    TopologyGatewayClient,
    _failure,
    _RejectRedirect,
    _schema_failure,
)
from .gateway_contracts import FAILURE_HTTP_STATUS, GatewayFailureReason, _load
from .gateway_opener import build_opener


class FrontierGatewayClientV2:
    def __init__(self, client):
        if type(client) is not TopologyGatewayClient:
            raise TypeError(
                "frontier transport requires validated internal gateway config"
            )
        self.client = client
        parts = urlsplit(client._endpoint)
        self.endpoint = urlunsplit((parts.scheme, parts.netloc, PATH, "", ""))

    def execute_frontier_read(self, *, kind, generation, keys, ready, deadline, budget):
        return self._execute(
            kind=kind,
            generation=generation,
            keys=keys,
            ready=ready,
            deadline=deadline,
            budget=budget,
        )

    def execute_frontier_batch(self, *, kind, lookups, ready, deadline, budget):
        return self._execute(
            kind=kind, lookups=lookups, ready=ready, deadline=deadline, budget=budget
        )

    def _execute(
        self, *, kind, ready, deadline, budget, generation=None, keys=(), lookups=None
    ):
        budget.check_active()
        allowance = budget.available_wire_bytes()
        if allowance <= 0:
            raise FrontierBudgetExceeded("frontier wire allowance exhausted")
        if lookups is None:
            body = encode_request(
                FrontierGatewayRequestV2(kind, generation, keys, ready, allowance)
            )
            endpoint, version, checksum = self.endpoint, VERSION, CHECKSUM
        else:
            body = encode_request(
                FrontierGatewayBatchRequestV2(kind, lookups, ready, allowance)
            )
            parts = urlsplit(self.endpoint)
            endpoint = urlunsplit((parts.scheme, parts.netloc, BATCH_PATH, "", ""))
            version, checksum = BATCH_VERSION, BATCH_CHECKSUM
        remaining = min(
            deadline - budget.clock(),
            budget.turn_budget.remaining_ms() / 1000,
            self.client.timeout_ceiling,
        )
        timeout_ms = floor(remaining * 1000)
        if timeout_ms <= 0:
            raise TimeoutError("frontier original deadline expired")
        request = Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": "Bearer " + self.client.bearer_token,
                "Content-Type": "application/json",
                "Accept-Encoding": "identity",
                "X-Topology-Schema-Version": version,
                "X-Topology-Schema-Checksum": checksum,
                "X-Frontier-Timeout-Ms": str(timeout_ms),
            },
        )
        opener = build_opener(endpoint, ProxyHandler({}), _RejectRedirect())
        try:
            try:
                response = opener.open(request, timeout=remaining)
            except HTTPError as error:
                response = error  # Invalid/error bodies consume the same byte ledger.
            try:
                return self._decode(
                    response,
                    allowance=allowance,
                    budget=budget,
                    deadline=deadline,
                    version=version,
                    checksum=checksum,
                )
            finally:
                response.close()
        except FrontierCompletedBudgetFailureV2:
            raise
        except BaseException:
            budget.stop_uncertain()
            raise

    def _decode(
        self,
        response,
        *,
        allowance,
        budget,
        deadline,
        version=VERSION,
        checksum=CHECKSUM,
    ):
        chunks, received = [], 0
        # urllib does not decompress bodies; request identity and reject encoded
        # responses after charging exactly the raw bytes received.
        while received < allowance:
            remaining = min(
                deadline - budget.clock(), budget.turn_budget.remaining_ms() / 1000
            )
            if remaining <= 0:
                raise TimeoutError("frontier response exceeded original deadline")
            stream, socket = response, None
            for _ in range(3):
                stream = getattr(stream, "fp", None)
                socket = getattr(getattr(stream, "raw", None), "_sock", None)
                if socket is not None:
                    break
            if socket is not None:
                socket.settimeout(remaining)
            chunk = response.read1(min(65536, allowance - received))
            if type(chunk) is not bytes:
                raise _schema_failure()
            budget.charge_page(scanned_rows=0, wire_bytes=len(chunk))
            received += len(chunk)
            chunks.append(chunk)
            if not chunk:
                break
        body = b"".join(chunks)
        budget.check_active()
        if budget.clock() >= deadline:
            raise TimeoutError("frontier response arrived late")
        headers = response.headers
        names = (
            "Content-Type",
            "Content-Length",
            "X-Topology-Schema-Version",
            "X-Topology-Schema-Checksum",
        )
        if any(len(headers.get_all(name, [])) != 1 for name in names):
            raise _schema_failure()
        if (
            headers.get("Content-Encoding") is not None
            or headers.get("Transfer-Encoding") is not None
            or headers.get("Content-Type") != "application/json"
            or headers.get("X-Topology-Schema-Version") != version
            or headers.get("X-Topology-Schema-Checksum") != checksum
        ):
            raise _schema_failure()
        if headers.get("Content-Length") != str(received):
            if received == allowance:
                raise FrontierBudgetExceeded("frontier wire allowance exhausted")
            raise _schema_failure()
        value = _load(body, maximum=allowance)
        if response.status != 200:
            if type(value) is dict and value.get("reason") == "frontier_budget":
                if (
                    set(value) != {"reason", "scanned_rows", "queries"}
                    or value["queries"] != 1
                    or type(value["queries"]) is not int
                ):
                    raise _schema_failure()
                raise FrontierCompletedBudgetFailureV2(value["scanned_rows"])
            if type(value) is dict and value.get("reason") == "deadline":
                raise TimeoutError("frontier gateway deadline")
            if type(value) is dict and set(value) == {"reason"}:
                try:
                    reason = GatewayFailureReason(value["reason"])
                except ValueError:
                    raise _schema_failure() from None
                if response.status != FAILURE_HTTP_STATUS[reason]:
                    raise _schema_failure()
                _failure(reason)
            raise _schema_failure()
        if version == BATCH_VERSION:
            from .frontier_gateway_codec_v2 import decode_batch_response

            budget.check_active()
            if budget.clock() >= deadline:
                raise TimeoutError("frontier decode exceeded original deadline")
            value = decode_batch_response(value)
            budget.check_active()
            if budget.clock() >= deadline:
                raise TimeoutError("frontier decode exceeded original deadline")
        if (
            type(value) is not dict
            or set(value) != {"rows", "scanned_rows", "queries"}
            or type(value["rows"]) is not list
            or type(value["scanned_rows"]) is not int
            or value["scanned_rows"] != len(value["rows"])
            or type(value["queries"]) is not int
            or value["queries"] != 1
        ):
            raise _schema_failure()
        return tuple(value["rows"])
