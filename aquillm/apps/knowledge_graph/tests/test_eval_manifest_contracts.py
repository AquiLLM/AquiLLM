"""Fixture identity and operator runbook contracts for graph evaluation."""

from __future__ import annotations

import inspect
from uuid import UUID

import pytest

from apps.knowledge_graph.evals import fixture_manifest, run_kg_eval
from apps.knowledge_graph.tests.test_eval_runner import AQUILLM_DIR, _fake_fixture_manifest


def test_fixture_manifest_binds_exact_symbols_topology_scope_and_hashes():
    manifest = _fake_fixture_manifest()
    requests = (
        (101, UUID("11111111-1111-4111-8111-111111111111")),
        (202, UUID("22222222-2222-4222-8222-222222222222")),
    )

    resolved = run_kg_eval.validate_fixture_manifest(
        manifest,
        extraction_cases=run_kg_eval.load_extraction_cases(),
        retrieval_cases=run_kg_eval.load_retrieval_cases(),
        collection_requests=requests,
    )

    assert resolved.authorized_scope == requests
    assert resolved.collections["collection-security-private"].authorized is False
    assert resolved.manifest_checksum == fixture_manifest.fixture_manifest_checksum(
        manifest
    )
    canonical = resolved.canonical_identity_assertions[0]
    assert resolved.chunk(canonical.source_chunk_symbol).collection_id == 101
    assert resolved.chunk(canonical.target_chunk_symbol).collection_id == 202


def test_fixture_manifest_overlap_assembly_and_embedding_digest_are_exact():
    first = "Atlas supplies the retrieval service. The retrieval service ranks passages"
    second = "The retrieval service ranks passages before synthesis."

    full_text, spans = run_kg_eval.assemble_fixture_document((first, second))

    assert full_text[spans[0][0] : spans[0][1]] == first
    assert full_text[spans[1][0] : spans[1][1]] == second
    assert spans[1][0] < spans[0][1]
    vector = tuple(0.25 for _ in range(1024))
    assert run_kg_eval.canonical_embedding_sha256(vector) == (
        run_kg_eval.canonical_embedding_sha256(vector)
    )
    with pytest.raises(run_kg_eval.FixtureValidationError, match="1024"):
        run_kg_eval.canonical_embedding_sha256((0.25,))


def test_fixture_checksum_is_shared_by_provider_neutral_manifest_module():
    assert fixture_manifest.fixture_checksum() == run_kg_eval.fixture_checksum()
    source = inspect.getsource(fixture_manifest)
    assert "django" not in source.lower()
    assert "torch" not in source.lower()
    assert "transformers" not in source.lower()


def test_eval_runbook_fences_worker_cleanup_by_exact_compose_labels():
    runbook = (
        AQUILLM_DIR.parent
        / "docs"
        / "documents"
        / "operations"
        / "knowledge-graph-overlay-runbook.md"
    ).read_text(encoding="utf-8")
    cleanup = runbook.split("stop_eval_worker() {", 1)[1].split("}\n", 1)[0]

    assert cleanup.index("docker inspect") < cleanup.index("docker rm -fv")
    assert 'com.docker.compose.project"' in cleanup
    assert 'com.docker.compose.service"' in cleanup
    assert 'com.docker.compose.oneoff"' in cleanup
    assert "worker_knowledge_graph" in cleanup
