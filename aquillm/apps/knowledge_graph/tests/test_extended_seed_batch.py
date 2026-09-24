"""PostgreSQL contracts for batched, per-chunk extended seeds."""

import os
from types import SimpleNamespace

import pytest
from django.db import connections
from django.test.utils import CaptureQueriesContext

from apps.knowledge_graph.retrieval import extended_seed_repository as source
from apps.knowledge_graph.retrieval.production_extended import prepare_extended_branch
from apps.knowledge_graph.tests.seed_lookup_fixtures import seeds  # noqa: F401

pytestmark = [
    pytest.mark.django_db(transaction=True, databases="__all__"),
    pytest.mark.skipif(
        os.environ.get("KG_REQUIRE_POSTGRES_TESTS") != "1",
        reason="requires isolated PostgreSQL",
    ),
]


class ReadableCodec:
    def encode(self, domain, *, source, **kwargs):
        return SimpleNamespace(value=f"{domain.value}:{source}")


def lookup(fixture, max_rows=4):
    return source.ExtendedSeedRepository().load_seed_identities(
        authority=fixture.authority,
        chunks=tuple((chunk.pk, fixture.document.id) for chunk in fixture.chunks),
        authorization=fixture.authorization,
        codec=ReadableCodec(),
        max_rows=max_rows,
    )


def test_batch_matches_prior_rows_and_independent_chunk_masses(seeds):  # noqa: F811
    from apps.knowledge_graph.projection.identifiers import (
        ProjectionIdentifierDomain as Domain,
    )

    canonical = f"{Domain.AUTOMATIC_CANONICAL_IDENTITY.value}:{seeds.canonical.pk}"
    beta = f"{Domain.ENTITY.value}:{seeds.entities[1].pk}"
    gamma = f"{Domain.ENTITY.value}:{seeds.entities[2].pk}"
    expected = {
        seeds.chunks[0].pk: tuple(sorted((canonical, beta))),
        seeds.chunks[1].pk: (canonical,),
        seeds.chunks[2].pk: (gamma,),
    }
    # Independent hand-checked associations: Alpha+Beta / Alpha / Gamma.
    with CaptureQueriesContext(connections["projection_source"]) as queries:
        assert lookup(seeds) == expected
    reads = [q["sql"] for q in queries if "SELECT DISTINCT" in q["sql"]]
    assert len(reads) == 1
    assert "LIMIT 5" in reads[0]
    for chunk, entities in zip(seeds.chunks, ((0, 1), (0,), (2,)), strict=True):
        prior = tuple(
            source._seed_query(
                authority=seeds.authority,
                chunk_id=chunk.pk,
                document_id=seeds.document.id,
                document_artifact_id=seeds.document_artifact.pk,
                using="projection_source",
            )
        )
        assert prior == tuple(
            (seeds.entities[i].pk, seeds.canonical.pk if i == 0 else None)
            for i in entities
        )
    runtime = SimpleNamespace(
        authorization=seeds.authorization,
        codec=seeds.codec,
        clock=lambda: 0.0,
        projection_repository_factory=source.ExtendedSeedRepository,
        _exact_request=lambda *args: None,
        _shared_scope=lambda _: seeds.scope,
    )
    prepared = prepare_extended_branch(
        runtime,
        baseline=SimpleNamespace(
            graph_seeds=tuple(
                SimpleNamespace(chunk_id=c.pk, restart_weight=w)
                for c, w in zip(seeds.chunks, (0.6, 0.3, 0.1))
            ),
            baseline_candidates=seeds.chunks,
        ),
        shared=object(),
        authorization=seeds.authorization,
        settings=SimpleNamespace(
            graph_extended_enabled=True, graph_extended_max_seeds=3
        ),
        deadline=1.0,
    )
    by_chunk = source.ExtendedSeedRepository().load_seed_identities(
        authority=seeds.authority,
        chunks=tuple((c.pk, seeds.document.id) for c in seeds.chunks),
        authorization=seeds.authorization,
        codec=seeds.codec,
        max_rows=4,
    )
    masses = {row.identity_key: row.mass for row in prepared}
    alpha_key = by_chunk[seeds.chunks[1].pk][0]
    beta_key = next(k for k in by_chunk[seeds.chunks[0].pk] if k != alpha_key)
    gamma_key = by_chunk[seeds.chunks[2].pk][0]
    assert masses == pytest.approx({alpha_key: 0.6, beta_key: 0.3, gamma_key: 0.1})
    with pytest.raises(ValueError, match="hard cap"):
        lookup(seeds, max_rows=3)


@pytest.mark.parametrize("invalid", ("revoked", "stale"))
def test_batch_revalidates_after_read(seeds, monkeypatch, invalid):  # noqa: F811
    assert lookup(seeds)
    original = source._seed_rows

    def read_then_invalidate(**kwargs):
        rows = original(**kwargs)
        if invalid == "revoked":
            seeds.permission.delete()
        else:
            seeds.state.registry_epoch += 1
            seeds.state.save(update_fields=["registry_epoch"])
        return rows

    monkeypatch.setattr(source, "_seed_rows", read_then_invalidate)
    with pytest.raises(ValueError):
        lookup(seeds)
