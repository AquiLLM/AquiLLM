"""Inspection catalog for independent experiments, never an activation authority.

Requested settings, dependency reports and actual evaluation reports are separate
records. Existing A3/D17/G6 gates remain responsible for eligibility. This module
does not certify incoming reports or turn a configured identity into an observation.
"""

import json
from copy import deepcopy
from hashlib import sha256

_CATALOG = {
    "window_source": (
        ("legacy", "windowed_only", "source_only", "windowed_source"),
        ("R1", "R2", "A1", "A2", "A3"),
    ),
    "retrieval_fusion": (("exact", "fts", "rrf", "protected_slot"), ("R3",)),
    "query_proposals": (("original", "vocabulary", "rewrite"), ("R4",)),
    "candidate_depth": (("baseline", "bounded_increase"), ("R3",)),
    "structure": (("legacy", "section_metadata"), ("I5",)),
    "embedding_preparation": (("verified_baseline", "contract_query"), ("I4",)),
    "graph": (("baseline", "direct", "extended", "combined"), ("G3", "G6")),
    "document_capacity": (("legacy", "budgeted"), ("A3",)),
    "followup": (("off", "on"), ("R4", "A3")),
    "iterative": (("off", "on"), ("A3",)),
    "adaptive_selection": (("legacy", "adaptive"), ("A3",)),
    "reranker_model": (("current_baseline", "separate_model_study"), ("R2",)),
}

# Names of required experimental controls, not claims about effective values.
_FIXED_CONTROLS = (
    "answer_model_and_revision",
    "final_evidence_cap",
    "shared_acquisition_actions",
    "shared_materialized_sources_and_codepoints",
    "shared_tokenized_codepoints",
    "shared_acquisition_and_final_pairs",
    "shared_pair_concurrency",
    "planner_calls_time_and_output",
    "retrieval_deadline_and_final_reserve",
    "pair_context_and_current_reranker",
    "gpu_allocation",
    "graph_budgets",
)

_REPORT_FIELDS = (
    "schema_version",
    "kind",
    "backend",
    "quality_claim",
    "mode",
    "split",
    "profile",
    "repetition",
    "revision",
    "dirty",
    "corpus_sha256",
    "manifest_sha256",
    "runtime_verified",
    "activation_eligible",
    "gate",
    "observations",
    "summary",
    "support_supplement",
)


def _invalid_constant(value):
    raise ValueError("nonfinite report JSON")


def _report(raw):
    if not isinstance(raw, bytes):
        raise ValueError("original report bytes required")
    data = json.loads(raw, parse_constant=_invalid_constant)
    if not isinstance(data, dict):
        raise ValueError("evaluation report object required")
    # Digest original bytes, including whitespace. Omitted fields remain omitted.
    return {
        "sha256": sha256(raw).hexdigest(),
        **{key: deepcopy(data[key]) for key in _REPORT_FIELDS if key in data},
    }


def treatment_manifest(
    *, configured=None, reports=None, associations=None, dependencies=None
):
    """Catalog supplied artifacts without running models, gates or enrollment.

    ``associations`` only identifies relevant reports; it does not certify that
    an independent treatment was executed. Reports retain their original gate
    decisions and observed controls inside observations. No missing observation
    is filled from ``configured``, dependency status or an artifact digest.

    A catalog alone cannot establish a treatment outcome, so its result remains
    inconclusive even if an input report claims eligibility. Controlled study
    results and independently trusted activation remain separate artifacts.
    """
    configured = configured or {}
    associations = associations or {}
    dependencies = dependencies or {}
    if set(configured) - _CATALOG.keys() or set(associations) - _CATALOG.keys():
        raise ValueError("unknown treatment")
    captured = {name: _report(raw) for name, raw in (reports or {}).items()}
    treatments = {}
    for name, (variants, requires) in _CATALOG.items():
        report_ids = associations.get(name, [])
        if (
            not isinstance(report_ids, (list, tuple))
            or any(not isinstance(value, str) for value in report_ids)
            or len(report_ids) != len(set(report_ids))
            or set(report_ids) - captured.keys()
        ):
            raise ValueError("unknown or duplicate treatment report association")
        treatments[name] = {
            "variants": list(variants),
            "requires": list(requires),
            "configured": deepcopy(configured.get(name)),
            "report_ids": list(report_ids),
            "dependency_observations": {
                key: deepcopy(dependencies[key])
                for key in requires
                if key in dependencies
            },
            "missing_dependency_observations": [
                key for key in requires if key not in dependencies
            ],
            "result": "inconclusive",
            "result_reason": "catalog does not certify an independent experiment",
        }
    return {
        "kind": "retrieval-treatment-inspection",
        "schema_version": 1,
        "activation_decision": "not_issued_by_catalog",
        "required_fixed_controls": list(_FIXED_CONTROLS),
        "reports": captured,
        "treatments": treatments,
    }
