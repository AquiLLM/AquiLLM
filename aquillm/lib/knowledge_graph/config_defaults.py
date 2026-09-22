"""Shared constants and validation error for graph worker configuration."""

from __future__ import annotations

import re
from pathlib import Path

DEFAULT_EXTRACTOR_PROVIDER = "gliner2_local"
DEFAULT_GLINER2_MODEL = "fastino/gliner2-base-v1"
DEFAULT_GLINER2_REVISION = "8437ba583a733d87f56ae902f3b197934eedd58e"
DEFAULT_GLINER2_DEVICE = "cpu"
DEFAULT_GLINER2_BATCH_SIZE = 8
DEFAULT_GLINER2_MAX_BATCH_CHARACTERS = 64_000
DEFAULT_GLINER2_CACHE_DIR = Path("/root/.cache/huggingface")
DEFAULT_ARTIFACT_RETENTION_DAYS = 30
DEFAULT_ARTIFACT_KEEP_SUPERSEDED = 2
DEFAULT_EXTRACTION_QUEUE = "knowledge-graph-extraction"
INVALID_EXTRACTION_QUEUE = "invalid-knowledge-graph-extraction"
MAX_EXTRACTION_QUEUE_LENGTH = 64
RESERVED_NON_GRAPH_QUEUES = frozenset({"celery", "memory-promotion"})
DEFAULT_EXTRACTION_CPU_UTILIZATION_PERCENT = 80
DEFAULT_GLINER2_CPU_THREADS_PER_WORKER = 8
MAX_EXTRACTION_WORKER_CONCURRENCY = 8

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-fA-F]{40}$")
_EXTRACTION_QUEUE = re.compile(
    rf"^[A-Za-z0-9][A-Za-z0-9._-]{{0,{MAX_EXTRACTION_QUEUE_LENGTH - 1}}}$"
)
class KnowledgeGraphConfigError(ValueError):
    """Raised when enabled graph extraction would use unsafe configuration."""
