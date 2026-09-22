"""Offline audit probes; no application writes, database, or network required.

Run from repository root: rtk proxy python artifacts/audits/2026-09-21-knowledge-graph/query_probes.py
These assert the current defects, not the desired corrected behavior.
"""

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "aquillm"))


def direct_identifier_mismatch():
    from apps.knowledge_graph.tests.test_direct_seed_repository import (
        _ready, _scope, _membership_state,
    )
    from apps.knowledge_graph.retrieval.topology.contracts import (
        ReadyGenerationBundleV1, ready_generation_bundle_checksum,
    )
    from apps.knowledge_graph.projection.identifiers import (
        HmacSha256ProjectionIdentifierCodec, ProjectionIdentifierDomain as Domain,
    )
    from apps.knowledge_graph.retrieval.direct_seed_repository import (
        DirectSeedRepository, DirectSeedCandidateRow,
    )
    from apps.knowledge_graph.retrieval.direct_seed_contracts import (
        DirectResolutionSpanInputV1,
    )
    from lib.knowledge_graph.query_extractor.contracts import QueryEntitySpanV1

    codec = HmacSha256ProjectionIdentifierCodec(b"audit-key", key_version="key-v1")
    generation = UUID("11111111-1111-4111-8111-111111111111")
    # ready_scope._encoded produces this opaque generation marker. Production
    # prepare_direct_seeds passes it unchanged into generation_keys_by_artifact.
    opaque_generation = codec.encode(
        Domain.COLLECTION, generation=generation, source=generation,
    ).value
    original = _ready()
    gen = replace(original.selected_generations[0], generation_key=opaque_generation)
    docs = (replace(original.authorized_documents[0], generation_key=opaque_generation),)
    signature = original.authorization_context_signature
    ready = ReadyGenerationBundleV1(
        (gen,), docs, signature,
        ready_generation_bundle_checksum((gen,), docs, signature),
    )
    scope = replace(
        _scope(ready), generation_keys_by_artifact=((11, opaque_generation),),
    )
    span = QueryEntitySpanV1("model", 0, 5, 1.0)
    # SQL's automatic subquery returns canonical_entity__identity_key, a string.
    rows = (
        DirectSeedCandidateRow(7, 11, "model", None, 1.0),
        DirectSeedCandidateRow(9, 11, "model", "a" * 64, 1.0),
    )
    repository = DirectSeedRepository(
        scope=scope, codec=codec,
        span_inputs=(DirectResolutionSpanInputV1(span, "model"),),
        row_loader=lambda **_kwargs: rows,
        membership_state_loader=lambda **_kwargs: _membership_state(ready),
    )
    matches = repository.canonical_name_matches(span=span, ready=ready, limit=4)
    actual_singleton = next(
        row.component_key for row in matches if row.entity_key == row.component_key
    )
    actual_canonical = next(
        row.component_key for row in matches if row.entity_key != row.component_key
    )
    # These are the exact codec inputs used by projection_encoding._entities and
    # _memberships. CanonicalEntity's database PK and identity_key are distinct.
    projected_singleton = codec.encode(Domain.ENTITY, generation=generation, source=7).value
    projected_canonical = codec.encode(Domain.AUTOMATIC_CANONICAL_IDENTITY, source=42).value
    assert actual_singleton != projected_singleton
    assert actual_canonical != projected_canonical
    print("P1 direct singleton seed equals projected identity: False")
    print("P1 direct canonical seed equals projected identity: False")


def revoked_authorization_falls_back_to_old_rows():
    from django.conf import settings

    settings.configure(
        DATABASES={"default": {"HOST": "unused"}},
        KG_OVERLAY_ENABLED=True,
        KG_MEMGRAPH_TRAVERSAL_ENABLED=True,
        KG_GRAPH_DIRECT_ENABLED=True,
        KG_GRAPH_EXTENDED_ENABLED=True,
        RAG_CACHE_ENABLED=False,
    )
    # Existing helper module probes DB reachability at import. Suppress it;
    # this audit must never connect to a service.
    with patch("socket.create_connection", side_effect=OSError("offline audit")):
        from apps.documents.tests.hybrid_graph_test_support import (
            Policy, authorization, chunk, selected_snapshot, hybrid_settings,
        )
        from apps.documents.tests.test_chunk_search_graph_overlay import _model
    from apps.documents.services import chunk_search, hybrid_graph_dependencies

    policy = Policy()
    auth = authorization(policy)
    snapshot = selected_snapshot(baseline=(chunk(1),))
    policy.rows = ()  # Permission revoked after the initial document selection.
    model, _filters = _model([])
    fake_utils = SimpleNamespace(get_embedding=lambda _query: (0.1, 0.2))
    with (
        patch.dict(sys.modules, {"aquillm.utils": fake_utils}),
        patch.object(hybrid_graph_dependencies, "django_hybrid_retrieval_settings", hybrid_settings),
        patch.object(chunk_search, "collect_hybrid_candidate_snapshot", lambda *_args, **_kwargs: snapshot),
        patch.object(chunk_search, "_fallback_rerank", lambda _model, rows, _k: rows),
    ):
        # Real resolve/build_hybrid_graph_dependencies/text_chunk_search execute.
        # Only ordinary candidate acquisition and reranking are replaced.
        result = chunk_search.text_chunk_search(
            model, "query", 3, list(snapshot.documents), authorization_context=auth,
        )
    returned = [row.pk for row in result[2]]
    assert policy.rows == ()
    assert returned == [1]
    assert result[3]["graph_status"] == "error"
    print("P1 authorized documents after revocation: 0")
    print(f"P1 baseline chunks still returned: {returned}")


if __name__ == "__main__":
    direct_identifier_mismatch()
    revoked_authorization_falls_back_to_old_rows()
