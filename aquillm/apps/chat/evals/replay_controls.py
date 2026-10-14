"""Explicit process-local retrieval-only treatment controls."""

import os
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

EXPERIMENTS = (
    "baseline",
    "graph-off",
    "direct-only",
    "extended-only",
    "depth30",
    "depth60",
    "full-text",
    "depth60-full",
    "current",
)
MODE_CHOICES = {
    "acquisition_schedule": ("scored", "native_closure"),
    "rerank_text_mode": ("legacy", "windowed"),
    "evidence_text_mode": ("legacy", "source"),
    "document_capacity_mode": ("legacy", "budgeted"),
    "query_mode": ("off", "vocabulary"),
    "graph_mode": ("configured", "off", "direct", "extended", "both"),
}
MODE_DEFAULTS = {key: values[0] for key, values in MODE_CHOICES.items()}


def effective_schedule_config():
    from apps.documents.services.native_search_schedule import (
        NATIVE_CLOSURE_RESERVE_MS,
        NATIVE_CLOSURE_SCHEDULE_VERSION,
        acquisition_schedule,
    )

    schedule = acquisition_schedule()
    native = schedule == "native_closure"
    return {
        "acquisition_schedule": schedule,
        "acquisition_schedule_version": (
            NATIVE_CLOSURE_SCHEDULE_VERSION if native else None
        ),
        "native_closure_reserve_ms": NATIVE_CLOSURE_RESERVE_MS if native else None,
    }


def validate_modes(experiment, modes):
    if experiment not in EXPERIMENTS or set(modes) != set(MODE_CHOICES):
        raise ValueError("experiment_unknown")
    if any(value not in MODE_CHOICES[key] for key, value in modes.items()):
        raise ValueError("mode_unknown")
    if experiment != "current" and modes != MODE_DEFAULTS:
        raise ValueError("current_modes_require_current_experiment")


@contextmanager
def replay_runtime(experiment, settings, config, pipeline, recorder, **overrides):
    modes = MODE_DEFAULTS | overrides
    validate_modes(experiment, modes)
    env = {
        "RAG_ACQUISITION_SCHEDULE": modes["acquisition_schedule"],
        "RAG_DIRECT_ENABLED": "1",
        "RAG_RERANK_TEXT_MODE": modes["rerank_text_mode"],
        "RAG_EVIDENCE_TEXT_MODE": modes["evidence_text_mode"],
        "RAG_DOCUMENT_CAPACITY_MODE": modes["document_capacity_mode"],
        "RAG_QUERY_REWRITE_ENABLED": "0",
        "RAG_VOCABULARY_QUERY_ENABLED": str(int(modes["query_mode"] == "vocabulary")),
        "RAG_FOLLOWUP_EVIDENCE_ENABLED": "0",
        "RAG_ITERATIVE_RETRIEVAL_ENABLED": "0",
        "RAG_RERANK_SHADOW_SCORING_ENABLED": "0",
    }
    attributes = [
        (settings, "RAG_CACHE_ENABLED", False),
        (pipeline, "synthesize_from_evidence", recorder),
    ]
    graph = {
        "graph-off": "off",
        "direct-only": "direct",
        "extended-only": "extended",
    }.get(experiment, modes["graph_mode"])
    if graph != "configured":
        flags = {
            "KG_OVERLAY_ENABLED": graph != "off",
            "KG_GRAPH_DIRECT_ENABLED": graph in ("direct", "both"),
            "KG_GRAPH_EXTENDED_ENABLED": graph in ("extended", "both"),
        }
        if graph == "extended":
            flags["KG_DIRECT_EMBEDDING_ENABLED"] = False
        for name, value in flags.items():
            attributes.append((settings, name, value))
            env[name] = str(int(value))
    if experiment in ("depth30", "depth60", "depth60-full"):
        depth = 30 if experiment == "depth30" else 60
        attributes.extend(
            [
                (config, "vector_top_k", depth),
                (config, "trigram_top_k", depth),
                (settings, "RAG_CANDIDATE_MULTIPLIER", 4.0),
            ]
        )
    if experiment in ("full-text", "depth60-full"):
        env["APP_RERANK_DOC_CHAR_LIMIT"] = "2048"
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, env))
        for obj, name, value in attributes:
            stack.enter_context(patch.object(obj, name, value, create=True))
        yield
