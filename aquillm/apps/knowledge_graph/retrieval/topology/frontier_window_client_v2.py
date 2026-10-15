"""Single zero-retry HTTP exchange under an already held original grant."""

from hashlib import sha256
from math import floor
from urllib.error import HTTPError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import ProxyHandler, Request

from . import frontier_gateway_client_v2 as client_module
from .frontier_window_contracts_v2 import CHECKSUM, PATH, VERSION
from .gateway_client import _RejectRedirect


def execute_window(transport, *, body, grant, deadline, budget, complete_scope=False):
    path, version, checksum = PATH, VERSION, CHECKSUM
    single = False
    if complete_scope:
        from .frontier_scope_window_contracts_v3 import CHECKSUM as checksum
        from .frontier_scope_window_contracts_v3 import PATH as path
        from .frontier_scope_window_contracts_v3 import VERSION as version
        from .frontier_scope_window_contracts_v3 import decode_request

        single = decode_request(body, budget.check_active).mode == "single"
    budget.check_active()
    remaining = min(
        deadline - budget.clock(),
        budget.turn_budget.remaining_ms() / 1000,
        transport.client.timeout_ceiling,
    )
    timeout_ms = floor(remaining * 1000)
    if timeout_ms <= 0:
        raise TimeoutError("frontier original deadline expired")
    parts = urlsplit(transport.endpoint)
    endpoint = urlunsplit((parts.scheme, parts.netloc, path, "", ""))
    request = Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Authorization": "Bearer " + transport.client.bearer_token,
            "Content-Type": "application/json",
            "Accept-Encoding": "identity",
            "X-Topology-Schema-Version": version,
            "X-Topology-Schema-Checksum": checksum,
            "X-Frontier-Timeout-Ms": str(timeout_ms),
        },
    )
    opener = client_module.build_opener(endpoint, ProxyHandler({}), _RejectRedirect())
    try:
        response = opener.open(request, timeout=remaining)
    except HTTPError as exc:
        response = exc
    try:
        return transport._decode(
            response,
            allowance=grant.wire,
            budget=budget,
            deadline=deadline,
            version=version,
            checksum=checksum,
            window_digest=sha256(body).hexdigest(),
            scope_single=single,
        )
    finally:
        response.close()
