"""Actual owned Memgraph behind the unmodified gateway app."""
from contextlib import contextmanager
from ipaddress import IPv4Address
import os
import socket
import subprocess
import sys
from time import monotonic, sleep
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from apps.knowledge_graph.retrieval.topology.gateway_config import load_topology_gateway_settings
from apps.knowledge_graph.retrieval.topology.gateway_contracts import MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES
from apps.knowledge_graph.retrieval.topology.gateway_client import TopologyGatewayClient


@contextmanager
def owned_gateway(tmp_path):
    assert os.environ.get("RAG_REMEDIATION_TEST") == "1"
    assert os.environ["KG_MEMGRAPH_URI"] == "bolt://memgraph:7687"
    addresses = {row[4][0] for row in socket.getaddrinfo(
        "memgraph", 7687, family=socket.AF_INET, type=socket.SOCK_STREAM,
    )}
    assert len(addresses) == 1, "owned Memgraph must resolve to one IPv4 address"
    address = IPv4Address(addresses.pop())
    assert address.is_private and not address.is_loopback and not address.is_unspecified
    environment = dict(os.environ)
    environment.update(
        KG_MEMGRAPH_URI=f"bolt://{address}:7687",
        KG_MEMGRAPH_DATABASE="memgraph",
        KG_MEMGRAPH_QUERY_USERNAME="synthetic_gateway",
        KG_MEMGRAPH_QUERY_PASSWORD="synthetic_gateway",
        KG_TOPOLOGY_GATEWAY_MAX_REQUEST_BYTES=str(MAX_REQUEST_BYTES),
        KG_TOPOLOGY_GATEWAY_MAX_RESPONSE_BYTES=str(MAX_RESPONSE_BYTES),
        KG_TOPOLOGY_GATEWAY_TIMEOUT_MS="5000",
    )
    config = load_topology_gateway_settings(environment)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    process = None
    log_path = tmp_path / "persisted-graph-gateway.log"
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn",
                 "apps.knowledge_graph.retrieval.topology.gateway_service:app",
                 "--fd", str(sock.fileno()), "--log-level", "warning"],
                env=environment, pass_fds=(sock.fileno(),),
                stdout=log, stderr=subprocess.STDOUT,
            )
            sock.close()
            end = monotonic() + 10
            ready = False
            while monotonic() < end and process.poll() is None:
                try:
                    with urlopen(url + "/readyz", timeout=1) as response:
                        ready = response.status == 200
                except HTTPError as error:
                    error.close()
                except (URLError, TimeoutError, ConnectionError):
                    pass
                if ready:
                    break
                sleep(0.02)
            assert ready, "actual owned gateway did not become ready"
            yield TopologyGatewayClient(
                url, config.bearer_token.get_secret_value(), 5.0,
            )
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
                raise AssertionError("owned gateway required forced termination")
        sock.close()
