"""Orchestrate one production-backed Task21 live observation publication."""

from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from .fixture_seed_cases import logical_fixture
from .task21_hybrid_eval import (
    TASK21_HYBRID_ARMS,
    build_task21_hybrid_report,
)
from .task21_hybrid_live_authority import load_live_fixture_authority
from .task21_hybrid_live_execution import ProductionArmExecutor
from .task21_hybrid_live_parity import build_live_backend_parity


def _digest(values) -> str:
    encoded = json.dumps(
        list(values), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return sha256(encoded).hexdigest()


def _snapshot(prepared):
    from apps.documents.models.chunks import TextChunk
    from apps.documents.services.chunk_search_candidates import (
        collect_hybrid_candidate_snapshot,
    )
    from apps.knowledge_graph.retrieval import get_graph_expansion_config
    from aquillm.utils import get_embedding

    query = prepared.case["query"]
    embedding = get_embedding(query)
    snapshot = collect_hybrid_candidate_snapshot(
        TextChunk,
        query,
        10,
        prepared.selected_documents,
        query_embedding=embedding,
        graph_config=get_graph_expansion_config(),
        authorization_context=prepared.authorization,
    )
    if snapshot.vector_error is not None or snapshot.graph_seed_error:
        raise RuntimeError("live candidate acquisition failed closed")
    return replace(prepared, snapshot=snapshot)


def _freshness(scopes, settings):
    from django.utils import timezone

    from apps.knowledge_graph.models.projections import CollectionGraphProjection

    by_projection = {
        authority.projection_id: authority
        for scope in scopes
        for authority in scope.projections
    }
    if not by_projection:
        raise RuntimeError("live run observed no ready projection authority")
    rows = tuple(
        CollectionGraphProjection.objects.using("projection_source")
        .filter(pk__in=tuple(by_projection))
        .only("id", "state", "ready_at", "graph_checksum")
    )
    if {row.id for row in rows} != set(by_projection):
        raise RuntimeError("live projection freshness rows are incomplete")
    now = timezone.now()
    ages = []
    for row in rows:
        authority = by_projection[row.id]
        if (
            row.state != CollectionGraphProjection.State.READY
            or row.ready_at is None
            or row.graph_checksum != authority.graph_checksum
        ):
            raise RuntimeError("live projection authority is not current-ready")
        ages.append(max(0.0, (now - row.ready_at).total_seconds()))
    maximum = float(settings.projection_max_lag_seconds)
    age = max(ages)
    if age > maximum:
        raise RuntimeError("live projection freshness gate rejected stale data")
    generations = tuple(
        sorted(
            {
                row.generation_key
                for scope in scopes
                for row in scope.ready.selected_generations
            }
        )
    )
    projections = tuple(
        sorted(
            {
                row.projection_key
                for scope in scopes
                for row in scope.ready.selected_generations
            }
        )
    )
    graph_checksums = tuple(
        sorted({authority.graph_checksum for authority in by_projection.values()})
    )
    ready_checksums = tuple(sorted({scope.ready.bundle_checksum for scope in scopes}))
    ontology = {
        (authority.ontology_version, authority.ontology_checksum)
        for authority in by_projection.values()
    }
    if len(ontology) != 1:
        raise RuntimeError("selected ready projections use mixed ontology")
    ontology_version, ontology_checksum = ontology.pop()
    return {
        "generation_key": _digest(generations),
        "projection_checksum": _digest(graph_checksums),
        "age_seconds": age,
        "max_age_seconds": maximum,
        "projection_keys": list(projections),
        "generation_keys": list(generations),
        "graph_checksums": list(graph_checksums),
        "ready_bundle_checksums": list(ready_checksums),
        "ontology_version": ontology_version,
        "ontology_checksum": ontology_checksum,
    }


def run_production_live_observations(
    *,
    run_id: str,
    source_commit: str,
    fixture_manifest: Path,
    runtime_identity: dict[str, object],
    output_paths,
) -> None:
    """Run all arms, validate the report, and atomically publish exact evidence."""

    from apps.documents.services.hybrid_graph_dependencies import (
        django_hybrid_retrieval_settings,
    )

    from .task21_hybrid_live_evidence import publish_live_artifacts
    from .task21_hybrid_live_observations import generate_case_arms

    authority = load_live_fixture_authority(fixture_manifest)
    settings = django_hybrid_retrieval_settings()
    executor = ProductionArmExecutor(settings)
    observations = {arm: [] for arm in TASK21_HYBRID_ARMS}
    cases = tuple(dict(case) for case in logical_fixture().retrieval_cases)
    for case in cases:
        prepared = authority.prepare_case(case)
        if prepared is None:
            accessible = frozenset(case["accessible_collection_ids"])
            case["adversarial_chunk_ids"] = tuple(
                sorted(
                    chunk["chunk_id"]
                    for document in case["documents"]
                    if document["collection_id"] not in accessible
                    for chunk in document["chunks"]
                )
            )
        else:
            prepared = _snapshot(prepared)
        rows = generate_case_arms(case=case, prepared=prepared, executor=executor)
        for arm in TASK21_HYBRID_ARMS:
            observations[arm].extend(rows[arm])
    freshness = _freshness(executor.ready_scopes, settings)
    parity = build_live_backend_parity(
        call_pairs=executor.parity_calls,
        ready_scopes=executor.ready_scopes,
        settings=settings,
    )
    build_task21_hybrid_report(
        cases=cases,
        observations=observations,
        freshness=freshness,
        backend_parity=parity,
    )
    publish_live_artifacts(
        paths=output_paths,
        run_id=run_id,
        source_commit=source_commit,
        runtime_identity=runtime_identity,
        observations=observations,
        freshness=freshness,
        backend_parity=parity,
    )


__all__ = ["run_production_live_observations"]
