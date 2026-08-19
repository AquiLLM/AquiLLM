from __future__ import annotations

import json
import os
import stat
from io import StringIO
from pathlib import Path
from uuid import UUID, uuid5

import pytest
from django.contrib.auth.models import User
from django.core.management import CommandError, call_command
from django.test import override_settings

from apps.knowledge_graph.tests.task21_fixture_test_support import (
    FIXTURE_ID,
    FIXTURE_NAMESPACE,
    HIDDEN_USERNAME,
    PHYSICAL_BINDINGS,
    VISIBLE_USERNAME,
    assert_no_fixture_graph_rows,
    cleanup,
    database_counts,
    fixture_module,
    fixture_row_counts,
    manifest_checksum,
    seed,
    strict_eval_environment,
)

_STRICT_EVAL_ENVIRONMENT = strict_eval_environment


@pytest.mark.django_db(transaction=True)
@override_settings(DEBUG=True)
def test_seed_publishes_exact_valid_manifest_and_synthetic_rows(
    tmp_path, monkeypatch
) -> None:
    from apps.collections.models import Collection, CollectionPermission
    from apps.documents.models import RawTextDocument, TextChunk
    from apps.knowledge_graph.evals.fixture_manifest import (
        assemble_fixture_document,
        canonical_embedding_sha256,
        fixture_checksum,
        validate_fixture_manifest,
    )
    from apps.knowledge_graph.evals.run_kg_eval import (
        load_extraction_cases,
        load_retrieval_cases,
    )

    manifest_path = tmp_path / "fixture.json"
    payload, output, observed = seed(manifest_path, monkeypatch)
    extraction_cases = load_extraction_cases()
    retrieval_cases = load_retrieval_cases()
    scope = tuple(
        (row["collection_id"], UUID(row["rebuild_request_id"]))
        for row in payload["authorized_scope"]
    )
    resolved = validate_fixture_manifest(
        payload,
        extraction_cases=extraction_cases,
        retrieval_cases=retrieval_cases,
        collection_requests=scope,
        expected_fixture_checksum=fixture_checksum(),
    )
    assert manifest_path.read_bytes() == (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
        + b"\n"
    )
    if os.name != "nt":
        assert stat.S_IMODE(manifest_path.stat().st_mode) & 0o077 == 0
    assert f"fixture_id={FIXTURE_ID}" in output
    assert f"fixture_checksum={fixture_checksum()}" in output
    assert f"manifest_checksum={resolved.manifest_checksum}" in output
    assert "collections=5" in output and "collections_deleted" not in output
    assert len(output) < 2_048
    physical_ids = {
        symbol: resolved.collections[symbol].collection_id
        for symbol in sorted(resolved.collections)
    }
    assert len(set(physical_ids.values())) == 5
    assert physical_ids["collection-public"] == physical_ids["collection-policy-a"]
    assert Collection.objects.filter(pk__in=set(physical_ids.values())).count() == 5
    for symbol, physical_name in PHYSICAL_BINDINGS.items():
        collection = Collection.objects.get(pk=physical_ids[symbol])
        assert collection.name == f"{FIXTURE_ID}-{physical_name}"
        assert collection.parent_id is None
        binding = resolved.collections[symbol]
        expected_request = (
            None
            if physical_name == "hidden"
            else uuid5(FIXTURE_NAMESPACE, f"rebuild:{physical_name}")
        )
        assert binding.rebuild_request_id == expected_request
    users = {
        user.username: user
        for user in User.objects.filter(
            username__in=(VISIBLE_USERNAME, HIDDEN_USERNAME)
        )
    }
    assert set(users) == {VISIBLE_USERNAME, HIDDEN_USERNAME}
    assert all(
        not user.is_active and not user.has_usable_password() for user in users.values()
    )
    visible_ids = {
        binding.collection_id
        for binding in resolved.collections.values()
        if binding.authorized
    }
    hidden_id = resolved.collections["collection-security-private"].collection_id
    permissions = set(
        CollectionPermission.objects.filter(user__in=users.values()).values_list(
            "user__username", "collection_id", "permission"
        )
    )
    assert permissions == {
        *((VISIBLE_USERNAME, value, "MANAGE") for value in visible_ids),
        (HIDDEN_USERNAME, hidden_id, "MANAGE"),
    }
    logical_documents = {}
    for case in (*extraction_cases, *retrieval_cases):
        for document in case["documents"]:
            rows = tuple(
                (chunk["chunk_id"], chunk["text"]) for chunk in document["chunks"]
            )
            logical = (document["collection_id"], rows)
            assert logical_documents.setdefault(document["doc_id"], logical) == logical
    assert (len(resolved.documents), len(resolved.chunks), len(observed[0])) == (
        28,
        30,
        30,
    )
    for symbol, binding in resolved.documents.items():
        document = RawTextDocument.objects.get(id=binding.document_id)
        expected_collection, logical_chunks = logical_documents[symbol]
        full_text, spans = assemble_fixture_document(
            tuple(row[1] for row in logical_chunks)
        )
        assert document.id == uuid5(FIXTURE_NAMESPACE, f"document:{symbol}")
        assert document.collection_id == physical_ids[expected_collection]
        assert (document.full_text, document.full_text_hash) == (
            full_text,
            binding.full_text_sha256,
        )
        chunks = list(TextChunk.objects.filter(doc_id=document.id))
        for number, ((chunk_symbol, text), span, chunk) in enumerate(
            zip(logical_chunks, spans, chunks, strict=True)
        ):
            chunk_binding = resolved.chunks[chunk_symbol]
            assert (chunk.pk, chunk.start_position, chunk.end_position) == (
                chunk_binding.chunk_id,
                *span,
            )
            assert (chunk.chunk_number, chunk.content) == (number, text)
            assert chunk.metadata == {
                "chunk_symbol": chunk_symbol,
                "fixture_id": FIXTURE_ID,
            }
            assert canonical_embedding_sha256(tuple(map(float, chunk.embedding))) == (
                chunk_binding.embedding_sha256
            )
    assert_no_fixture_graph_rows()


@pytest.mark.django_db(transaction=True)
@override_settings(DEBUG=True)
def test_seed_is_exactly_idempotent_without_reembedding(tmp_path, monkeypatch) -> None:
    manifest_path = tmp_path / "fixture.json"
    payload, first_output, _observed = seed(manifest_path, monkeypatch)
    first_bytes = manifest_path.read_bytes()
    first_counts = database_counts()
    monkeypatch.setattr(
        fixture_module(),
        "strict_index_embedding_signature",
        lambda: (_ for _ in ()).throw(AssertionError("embedding reached")),
    )
    output = StringIO()
    call_command(
        "seed_knowledge_graph_eval_fixture",
        "--fixture-manifest",
        str(manifest_path),
        stdout=output,
    )
    assert manifest_path.read_bytes() == first_bytes
    assert database_counts() == first_counts
    assert output.getvalue() == first_output
    assert manifest_checksum(payload) in output.getvalue()


@pytest.mark.django_db(transaction=True)
@override_settings(DEBUG=True)
def test_seed_rejects_manifest_when_owned_rows_were_cleaned(
    tmp_path, monkeypatch
) -> None:
    manifest_path = tmp_path / "fixture.json"
    payload, _output, _observed = seed(manifest_path, monkeypatch)
    cleanup(manifest_path, payload)
    monkeypatch.setattr(
        fixture_module(),
        "strict_index_embedding_signature",
        lambda: (_ for _ in ()).throw(AssertionError("embedding reached")),
    )
    with pytest.raises(CommandError, match="topology"):
        call_command(
            "seed_knowledge_graph_eval_fixture",
            "--fixture-manifest",
            str(manifest_path),
        )
    assert fixture_row_counts() == (0, 0, 0, 0)


@pytest.mark.django_db(transaction=True)
@override_settings(DEBUG=True)
def test_seed_allows_active_ontology_without_graph_rows(tmp_path, monkeypatch) -> None:
    from apps.knowledge_graph.models import OntologyVersion
    from apps.knowledge_graph.services.ontology import activate_ontology, load_ontology

    ontology = Path(__file__).resolve().parents[1] / "ontologies" / "research-v1.yaml"
    active = activate_ontology(load_ontology(ontology))
    seed(tmp_path / "fixture.json", monkeypatch)
    assert (
        OntologyVersion.objects.get(pk=active.pk).status
        == OntologyVersion.Status.ACTIVE
    )
    assert_no_fixture_graph_rows()
