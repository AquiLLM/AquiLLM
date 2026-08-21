# ruff: noqa: E501,E701,E702
"""Pure, provider-neutral configuration for hybrid graph retrieval."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from ipaddress import ip_address
from urllib.parse import unquote_to_bytes, urlsplit

__all__ = ("HybridRetrievalConfigError", "HybridRetrievalSettings", "SecretSetting", "QUERY_EXTRACTOR_MODEL", "QUERY_EXTRACTOR_MODEL_REVISION", "QUERY_SCHEMA_CHECKSUM", "QUERY_SCHEMA_VERSION", "load_django_hybrid_retrieval_settings", "load_hybrid_retrieval_settings", "select_evaluation_topology_backend")  # fmt: skip

QUERY_EXTRACTOR_MODEL = "fastino/gliner2-base-v1"
QUERY_EXTRACTOR_MODEL_REVISION = "8437ba583a733d87f56ae902f3b197934eedd58e"
QUERY_SCHEMA_VERSION = "query-entities-v1"
QUERY_SCHEMA_CHECKSUM = "45bc8f86637a73324d2edae3096aac61d242fb0bcbab3c481cfa7599456cd271"  # fmt: skip

_EVALUATION_BACKEND_CAPABILITY = object()
_DECIMAL = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")
_LOWER_REVISION = re.compile(r"^[0-9a-f]{40}$")
_LOWER_CHECKSUM = re.compile(r"^[0-9a-f]{64}$")
_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_LOWER_TOKEN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,255}$")
_DNS_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_ROLE = re.compile(r"^[a-z][a-z0-9_]{0,62}$")
_DB_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_BAD_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")


class HybridRetrievalConfigError(ValueError):
    """Raised when hybrid retrieval configuration is invalid or unsafe."""


# fmt: off
class SecretSetting:
    __value: str
    __slots__ = ("__value",)
    def __init__(self, value: str) -> None:
        if type(value) is not str:
            raise TypeError("secret setting must be an exact string")
        object.__setattr__(self, "_SecretSetting__value", value)
    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("secret setting is immutable")
    def __repr__(self) -> str:
        return "<redacted>"
    __str__ = __repr__
    def __bool__(self) -> bool:
        return bool(self.__value)
    def __eq__(self, other: object) -> bool:
        return isinstance(other, SecretSetting) and type(other) is SecretSetting and self.__value == other.__value
    def __hash__(self) -> int:
        return hash(self.__value)
    def __deepcopy__(self, memo: object) -> SecretSetting:
        return self
    def get_secret_value(self) -> str:
        return self.__value
@dataclass(frozen=True, slots=True)
class HybridRetrievalSettings:
    memgraph_projection_enabled: bool; memgraph_traversal_enabled: bool; graph_direct_enabled: bool; graph_extended_enabled: bool
    graph_topology_backend: str; graph_algorithm: str; memgraph_image: str; memgraph_uri: str = field(repr=False)
    memgraph_database: str; memgraph_query_username: str; memgraph_query_password: SecretSetting = field(repr=False); memgraph_projection_username: str
    memgraph_projection_password: SecretSetting = field(repr=False); projection_postgres_source_dsn: SecretSetting = field(repr=False); projection_postgres_state_dsn: SecretSetting = field(repr=False)
    projection_queue: str; projection_schema_version: str; projection_format_version: str; projection_identifier_hmac_key: SecretSetting = field(repr=False)
    projection_identifier_key_version: str; projection_batch_size: int; projection_lease_seconds: int; projection_max_attempts: int; projection_retention: int
    projection_max_lag_seconds: int; query_extractor_url: str = field(repr=False); query_extractor_bearer_token: SecretSetting = field(repr=False)
    query_extractor_model: str; query_extractor_model_revision: str; query_extractor_build_hash: str; query_extractor_expected_schema_version: str; query_extractor_expected_schema_checksum: str
    query_extractor_timeout_ms: int; query_max_bytes: int; query_max_codepoints: int; query_max_spans: int
    graph_overall_timeout_ms: int; graph_direct_timeout_ms: int; graph_extended_timeout_ms: int; graph_direct_max_seeds: int
    graph_direct_max_depth: int; graph_direct_max_nodes: int; graph_direct_max_edges: int; graph_direct_max_candidates: int
    graph_extended_max_seeds: int; graph_extended_max_depth: int; graph_extended_max_nodes: int; graph_extended_max_edges: int
    graph_extended_max_candidates: int; graph_fusion_rrf_k: int; direct_embedding_enabled: bool; direct_min_similarity: float
    direct_winner_margin: float; graph_eval_parity_backend: str
_BOOL_DEFAULTS = dict.fromkeys((
    "KG_MEMGRAPH_PROJECTION_ENABLED", "KG_MEMGRAPH_TRAVERSAL_ENABLED",
    "KG_GRAPH_DIRECT_ENABLED", "KG_GRAPH_EXTENDED_ENABLED", "KG_DIRECT_EMBEDDING_ENABLED",
), "0")
_INT_RULES = {
    "KG_PROJECTION_BATCH_SIZE": ("500", 1, 5000), "KG_PROJECTION_LEASE_SECONDS": ("300", 10, 3600), "KG_PROJECTION_MAX_ATTEMPTS": ("5", 1, 20), "KG_PROJECTION_RETENTION": ("2", 1, 50),
    "KG_PROJECTION_MAX_LAG_SECONDS": ("300", 1, 86400), "KG_QUERY_EXTRACTOR_TIMEOUT_MS": ("75", 10, 1000), "KG_QUERY_MAX_BYTES": ("4096", 1, 16384), "KG_QUERY_MAX_CODEPOINTS": ("2048", 1, 8192),
    "KG_QUERY_MAX_SPANS": ("32", 1, 128), "KG_GRAPH_OVERALL_TIMEOUT_MS": ("300", 25, 5000), "KG_GRAPH_DIRECT_TIMEOUT_MS": ("125", 10, 5000), "KG_GRAPH_EXTENDED_TIMEOUT_MS": ("125", 10, 5000),
    "KG_GRAPH_DIRECT_MAX_SEEDS": ("32", 1, 64), "KG_GRAPH_DIRECT_MAX_DEPTH": ("2", 1, 2), "KG_GRAPH_DIRECT_MAX_NODES": ("200", 1, 200), "KG_GRAPH_DIRECT_MAX_EDGES": ("1000", 1, 1000),
    "KG_GRAPH_DIRECT_MAX_CANDIDATES": ("20", 1, 20), "KG_GRAPH_EXTENDED_MAX_SEEDS": ("64", 1, 64), "KG_GRAPH_EXTENDED_MAX_DEPTH": ("2", 1, 2), "KG_GRAPH_EXTENDED_MAX_NODES": ("200", 1, 200),
    "KG_GRAPH_EXTENDED_MAX_EDGES": ("1000", 1, 1000), "KG_GRAPH_EXTENDED_MAX_CANDIDATES": ("20", 1, 20),
    "KG_GRAPH_FUSION_RRF_K": ("60", 60, 60),
}
_FLOAT_RULES = {"KG_DIRECT_MIN_SIMILARITY": ("0.80", 0.0, 1.0), "KG_DIRECT_WINNER_MARGIN": ("0.05", 0.0, 1.0)}
_TEXT_DEFAULTS = {
    "KG_GRAPH_TOPOLOGY_BACKEND": "memgraph", "KG_GRAPH_ALGORITHM": "ppr_projected_v1", "KG_MEMGRAPH_IMAGE": "memgraph/memgraph-mage:3.8.1", "KG_MEMGRAPH_URI": "", "KG_MEMGRAPH_DATABASE": "memgraph",
    "KG_MEMGRAPH_QUERY_USERNAME": "", "KG_MEMGRAPH_QUERY_PASSWORD": "", "KG_MEMGRAPH_PROJECTION_USERNAME": "", "KG_MEMGRAPH_PROJECTION_PASSWORD": "",
    "KG_PROJECTION_POSTGRES_SOURCE_DSN": "", "KG_PROJECTION_POSTGRES_STATE_DSN": "", "KG_PROJECTION_QUEUE": "knowledge_graph_projection", "KG_PROJECTION_SCHEMA_VERSION": "collection-graph-v1",
    "KG_PROJECTION_FORMAT_VERSION": "projection-v1", "KG_PROJECTION_IDENTIFIER_HMAC_KEY": "", "KG_PROJECTION_IDENTIFIER_KEY_VERSION": "", "KG_QUERY_EXTRACTOR_URL": "",
    "KG_QUERY_EXTRACTOR_BEARER_TOKEN": "", "KG_QUERY_EXTRACTOR_MODEL": QUERY_EXTRACTOR_MODEL, "KG_QUERY_EXTRACTOR_MODEL_REVISION": QUERY_EXTRACTOR_MODEL_REVISION, "KG_QUERY_EXTRACTOR_BUILD_HASH": "",
    "KG_QUERY_EXTRACTOR_EXPECTED_SCHEMA_VERSION": QUERY_SCHEMA_VERSION, "KG_QUERY_EXTRACTOR_EXPECTED_SCHEMA_CHECKSUM": "", "KG_GRAPH_EVAL_PARITY_BACKEND": "postgres",
}
# fmt: on
_ALLOWED_KEYS = frozenset(_BOOL_DEFAULTS | _INT_RULES | _FLOAT_RULES | _TEXT_DEFAULTS)
_SECRET_KEYS = frozenset(
    key
    for key in _TEXT_DEFAULTS
    if any(marker in key for marker in ("PASSWORD", "BEARER_TOKEN", "HMAC_KEY", "DSN"))
)


# fmt: off
def _error(key: str, reason: str) -> HybridRetrievalConfigError: return HybridRetrievalConfigError(f"{key} {reason}")
def _raw(source: Mapping[str, str], key: str, default: str) -> str:
    value = source.get(key, default)
    if type(value) is not str:
        raise _error(key, "must be an exact string")
    return value
def _parse_text(source: Mapping[str, str], key: str, default: str) -> str:
    value = _raw(source, key, default)
    if len(value) > 4096 or any(ord(char) < 32 or ord(char) == 127 or 0xD800 <= ord(char) <= 0xDFFF for char in value):
        raise _error(key, "contains invalid text")
    if key not in _SECRET_KEYS and value != value.strip():
        raise _error(key, "must not contain surrounding whitespace")
    return value
def _parse_int(source: Mapping[str, str], key: str, rule: tuple[str, int, int]) -> int:
    raw = _raw(source, key, rule[0])
    if len(raw) > len(str(rule[2])) or not raw.isascii() or not raw.isdecimal() or (len(raw) > 1 and raw[0] == "0"):
        raise _error(key, "must be a canonical decimal integer")
    value = int(raw)
    if not rule[1] <= value <= rule[2]:
        raise _error(key, "is outside the supported range")
    return value
def _parse_float(source: Mapping[str, str], key: str, rule: tuple[str, float, float]) -> float:
    raw = _raw(source, key, rule[0])
    if _DECIMAL.fullmatch(raw) is None:
        raise _error(key, "must be a canonical decimal number")
    value = float(raw)
    if not rule[1] <= value <= rule[2]:
        raise _error(key, "is outside the supported range")
    return value
def _decoded_uri_part(key: str, raw: str, forbidden: str) -> str:
    if _BAD_ESCAPE.search(raw): raise _error(key, "contains an invalid percent escape")
    try:
        decoded = unquote_to_bytes(raw).decode("utf-8", errors="strict")
    except UnicodeError: raise _error(key, "contains invalid UTF-8") from None
    if len(decoded) > 4096 or any(ord(char) < 32 or ord(char) == 127 or char in forbidden for char in decoded): raise _error(key, "contains invalid decoded text")
    return decoded
def _host_port_identity(host_port: str) -> tuple[str, int] | None:
    bracketed = host_port.startswith("[")
    if bracketed:
        closing = host_port.find("]")
        if closing < 0: return None
        host_text, tail = host_port[1:closing], host_port[closing + 1:]
        port_text = tail[1:] if tail.startswith(":") else None if not tail else ""
    else:
        if host_port.count(":") > 1: return None
        host_text, delimiter, port_text = host_port.partition(":")
        port_text = port_text if delimiter else None
    if not host_text or "%" in host_text or "\\" in host_text: return None
    try:
        address = ip_address(host_text)
        normalized_host = address.compressed
        if (address.version == 6) != bracketed: return None
    except ValueError:
        normalized_host = host_text.lower()
        if bracketed or len(normalized_host) > 253 or normalized_host.startswith(".") or normalized_host.endswith(".") or not all(_DNS_LABEL.fullmatch(label) for label in normalized_host.split(".")): return None
    if port_text is None: return normalized_host, 5432
    if len(port_text) > 5 or not port_text.isascii() or not port_text.isdecimal() or (len(port_text) > 1 and port_text[0] == "0"): return None
    port = int(port_text)
    return (normalized_host, port) if 1 <= port <= 65535 else None
def _validate_url(key: str, value: str, schemes: frozenset[str]) -> None:
    if not value:
        return
    try:
        parsed = urlsplit(value)
    except ValueError:
        raise _error(key, "must be a canonical service URL") from None
    invalid = not value.isascii() or len(value) > 2048 or any(char.isspace() for char in value) or parsed.scheme not in schemes or not any(value.startswith(f"{scheme}://") for scheme in schemes) or "@" in parsed.netloc or _host_port_identity(parsed.netloc) is None or parsed.netloc != parsed.netloc.lower() or parsed.query or parsed.fragment or "\\" in parsed.path
    if key == "KG_MEMGRAPH_URI":
        invalid = invalid or parsed.path not in {"", "/"}
    if invalid:
        raise _error(key, "must be a canonical service URL")
def _postgres_identity(key: str, value: str) -> tuple[str, str, int, str]:
    if not value.startswith("postgresql://") or not value.isascii() or any(char.isspace() for char in value) or "\\" in value or value.count("@") != 1:
        raise _error(key, "must be a canonical PostgreSQL URI")
    try:
        parsed = urlsplit(value)
    except ValueError:
        raise _error(key, "must be a canonical PostgreSQL URI") from None
    if parsed.netloc.count("@") != 1 or parsed.query or parsed.fragment or parsed.path.count("/") != 1 or not parsed.path.startswith("/"):
        raise _error(key, "must be a canonical PostgreSQL URI")
    userinfo, host_port = parsed.netloc.split("@")
    raw_username, password_separator, raw_password = userinfo.partition(":")
    username = _decoded_uri_part(key, raw_username, ":/@?#\\")
    if _ROLE.fullmatch(username) is None:
        raise _error(key, "must contain a canonical PostgreSQL role")
    if password_separator:
        _decoded_uri_part(key, raw_password, "")
    database = _decoded_uri_part(key, parsed.path[1:], "/@?#\\")
    host_identity = _host_port_identity(host_port)
    if host_identity is None or _DB_NAME.fullmatch(database) is None:
        raise _error(key, "must contain a canonical host and database")
    return username, host_identity[0], host_identity[1], database
def _require(settings: HybridRetrievalSettings, keys: str) -> None:
    for required_key in keys.split():
        if not getattr(settings, required_key[3:].lower()):
            raise _error(required_key, "is required for the enabled path")
def _validate_settings(settings: HybridRetrievalSettings) -> None:
    if settings.graph_topology_backend != "memgraph":
        raise _error("KG_GRAPH_TOPOLOGY_BACKEND", "must be memgraph in production")
    fixed = {"KG_GRAPH_ALGORITHM": "ppr_projected_v1", "KG_PROJECTION_SCHEMA_VERSION": "collection-graph-v1", "KG_PROJECTION_FORMAT_VERSION": "projection-v1", "KG_GRAPH_EVAL_PARITY_BACKEND": "postgres"}
    for fixed_key, expected in fixed.items():
        if getattr(settings, fixed_key[3:].lower()) != expected:
            raise _error(fixed_key, "selects an unsupported contract")
    if settings.graph_direct_timeout_ms > settings.graph_overall_timeout_ms:
        raise _error("KG_GRAPH_DIRECT_TIMEOUT_MS", "must not exceed overall timeout")
    if settings.graph_extended_timeout_ms > settings.graph_overall_timeout_ms:
        raise _error("KG_GRAPH_EXTENDED_TIMEOUT_MS", "must not exceed overall timeout")
    if settings.direct_winner_margin > settings.direct_min_similarity:
        raise _error("KG_DIRECT_WINNER_MARGIN", "must not exceed minimum similarity")
    if settings.memgraph_projection_enabled:
        _require(settings, "KG_MEMGRAPH_URI KG_MEMGRAPH_DATABASE KG_MEMGRAPH_PROJECTION_USERNAME KG_MEMGRAPH_PROJECTION_PASSWORD KG_PROJECTION_POSTGRES_SOURCE_DSN KG_PROJECTION_POSTGRES_STATE_DSN KG_PROJECTION_IDENTIFIER_HMAC_KEY KG_PROJECTION_IDENTIFIER_KEY_VERSION")
    if settings.memgraph_traversal_enabled:
        _require(settings, "KG_MEMGRAPH_URI KG_MEMGRAPH_DATABASE KG_MEMGRAPH_QUERY_USERNAME KG_MEMGRAPH_QUERY_PASSWORD")
    if settings.graph_direct_enabled:
        if not settings.memgraph_traversal_enabled:
            raise _error("KG_MEMGRAPH_TRAVERSAL_ENABLED", "is required for direct retrieval")
        _require(settings, "KG_QUERY_EXTRACTOR_URL KG_QUERY_EXTRACTOR_BEARER_TOKEN KG_QUERY_EXTRACTOR_BUILD_HASH")
        direct = {"KG_QUERY_EXTRACTOR_MODEL": QUERY_EXTRACTOR_MODEL, "KG_QUERY_EXTRACTOR_MODEL_REVISION": QUERY_EXTRACTOR_MODEL_REVISION, "KG_QUERY_EXTRACTOR_EXPECTED_SCHEMA_VERSION": QUERY_SCHEMA_VERSION, "KG_QUERY_EXTRACTOR_EXPECTED_SCHEMA_CHECKSUM": QUERY_SCHEMA_CHECKSUM}
        for direct_key, expected in direct.items():
            if getattr(settings, direct_key[3:].lower()) != expected:
                raise _error(direct_key, "does not match the direct retrieval contract")
    if settings.graph_extended_enabled and not settings.memgraph_traversal_enabled:
        raise _error("KG_MEMGRAPH_TRAVERSAL_ENABLED", "is required for extended retrieval")
    if settings.direct_embedding_enabled and not settings.graph_direct_enabled:
        raise _error("KG_GRAPH_DIRECT_ENABLED", "must be enabled for direct embedding")
def load_hybrid_retrieval_settings(source: Mapping[str, str]) -> HybridRetrievalSettings:
    if not isinstance(source, Mapping):
        raise HybridRetrievalConfigError("configuration source must be a mapping")
    for source_key, source_value in source.items():
        if type(source_key) is not str:
            raise HybridRetrievalConfigError("configuration keys and values must be exact strings")
        if type(source_value) is not str:
            raise _error(source_key, "must be an exact string")
        if source_key.startswith("KG_") and source_key not in _ALLOWED_KEYS:
            raise _error(source_key, "is not a supported hybrid retrieval setting")
    values: dict[str, object] = {}
    for bool_key, default in _BOOL_DEFAULTS.items():
        raw = _raw(source, bool_key, default)
        if raw not in {"0", "1"}:
            raise _error(bool_key, "must be exactly 0 or 1")
        values[bool_key[3:].lower()] = raw == "1"
    for int_key, int_rule in _INT_RULES.items():
        values[int_key[3:].lower()] = _parse_int(source, int_key, int_rule)
    for float_key, float_rule in _FLOAT_RULES.items():
        values[float_key[3:].lower()] = _parse_float(source, float_key, float_rule)
    for text_key, default in _TEXT_DEFAULTS.items():
        text_value = _parse_text(source, text_key, default)
        values[text_key[3:].lower()] = SecretSetting(text_value) if text_key in _SECRET_KEYS else text_value
    settings = HybridRetrievalSettings(**values)  # type: ignore[arg-type]
    _validate_url("KG_MEMGRAPH_URI", settings.memgraph_uri, frozenset({"bolt", "bolt+s", "neo4j", "neo4j+s"}))
    _validate_url("KG_QUERY_EXTRACTOR_URL", settings.query_extractor_url, frozenset({"http", "https"}))
    source_dsn = settings.projection_postgres_source_dsn.get_secret_value(); state_dsn = settings.projection_postgres_state_dsn.get_secret_value()
    source_identity = _postgres_identity("KG_PROJECTION_POSTGRES_SOURCE_DSN", source_dsn) if source_dsn else None
    state_identity = _postgres_identity("KG_PROJECTION_POSTGRES_STATE_DSN", state_dsn) if state_dsn else None
    if source_identity is not None and state_identity is not None:
        if source_identity == state_identity:
            raise _error("KG_PROJECTION_POSTGRES_STATE_DSN", "must use a distinct canonical database identity")
        if source_identity[0] == state_identity[0]:
            raise _error("KG_PROJECTION_POSTGRES_STATE_DSN", "must use a distinct PostgreSQL role")
    if _TOKEN.fullmatch(settings.projection_queue) is None:
        raise _error("KG_PROJECTION_QUEUE", "must be a bounded token")
    if settings.memgraph_database and _TOKEN.fullmatch(settings.memgraph_database) is None:
        raise _error("KG_MEMGRAPH_DATABASE", "must be a bounded token")
    for username_key in ("KG_MEMGRAPH_QUERY_USERNAME", "KG_MEMGRAPH_PROJECTION_USERNAME"):
        username = getattr(settings, username_key[3:].lower())
        if username and _TOKEN.fullmatch(username) is None:
            raise _error(username_key, "must be a bounded token")
    if any(char.isspace() for char in settings.memgraph_image):
        raise _error("KG_MEMGRAPH_IMAGE", "must be a bounded image reference")
    if _LOWER_TOKEN.fullmatch(settings.query_extractor_expected_schema_version) is None:
        raise _error("KG_QUERY_EXTRACTOR_EXPECTED_SCHEMA_VERSION", "must be a canonical schema version")
    if settings.projection_identifier_key_version and _LOWER_TOKEN.fullmatch(settings.projection_identifier_key_version) is None:
        raise _error("KG_PROJECTION_IDENTIFIER_KEY_VERSION", "must be a canonical key version")
    if _MODEL.fullmatch(settings.query_extractor_model) is None:
        raise _error("KG_QUERY_EXTRACTOR_MODEL", "must be a bounded model identifier")
    if _LOWER_REVISION.fullmatch(settings.query_extractor_model_revision) is None:
        raise _error("KG_QUERY_EXTRACTOR_MODEL_REVISION", "must be a lowercase revision")
    for checksum_key in ("KG_QUERY_EXTRACTOR_BUILD_HASH", "KG_QUERY_EXTRACTOR_EXPECTED_SCHEMA_CHECKSUM"):
        checksum = getattr(settings, checksum_key[3:].lower())
        if checksum and _LOWER_CHECKSUM.fullmatch(checksum) is None: raise _error(checksum_key, "must be a lowercase SHA-256")
    _validate_settings(settings)
    return settings
def load_django_hybrid_retrieval_settings(source: Mapping[str, str]) -> dict[str, object]:
    if not isinstance(source, Mapping): raise HybridRetrievalConfigError("configuration source must be a mapping")
    settings = load_hybrid_retrieval_settings({key: source[key] for key in _ALLOWED_KEYS if key in source})
    return {key: getattr(settings, key[3:].lower()) for key in _ALLOWED_KEYS}
def select_evaluation_topology_backend(settings: HybridRetrievalSettings, *, capability: object) -> str:
    """Return the parity backend only to code holding the private test capability."""
    if capability is not _EVALUATION_BACKEND_CAPABILITY:
        raise HybridRetrievalConfigError("evaluation backend capability is required")
    if type(settings) is not HybridRetrievalSettings or settings.graph_eval_parity_backend != "postgres":
        raise HybridRetrievalConfigError("evaluation backend must be postgres")
    return "postgres"
# fmt: on
