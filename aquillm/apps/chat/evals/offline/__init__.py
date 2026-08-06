"""Pure, deterministic scorers for the offline evidence evaluation."""

from .metrics import (
    aggregate_evidence,
    binary_metrics,
    categorical_conformance,
    citation_diagnostics,
    compare_policies,
    exact_set_metrics,
    memory_stratum_errors,
    query_conformance,
    score_evidence_case,
)
from .policies import sequential_select
from .schema import (
    SOURCE_TEXT_HASH_ALGORITHM,
    canonical_json_bytes,
    load_dataset,
    sha256_canonical_text,
    sha256_file,
    validate_dataset,
    validate_test_manifest,
)

__all__ = [
    "SOURCE_TEXT_HASH_ALGORITHM",
    "aggregate_evidence",
    "binary_metrics",
    "canonical_json_bytes",
    "categorical_conformance",
    "citation_diagnostics",
    "compare_policies",
    "exact_set_metrics",
    "load_dataset",
    "memory_stratum_errors",
    "query_conformance",
    "score_evidence_case",
    "sequential_select",
    "sha256_canonical_text",
    "sha256_file",
    "validate_dataset",
    "validate_test_manifest",
]
