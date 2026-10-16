"""All retained candidates keep source proof beyond the projected evidence prefix."""

from dataclasses import replace

import pytest

from apps.knowledge_graph.tests.consumed_source_test_support import (
    acquire_fixture,
    multi_document_fixture,
)


@pytest.mark.parametrize("mention_only", [False, True])
def test_every_retained_reference_keeps_document_proof_beyond_projection(mention_only):
    f = multi_document_fixture(mention_only=mention_only)
    seeds = acquire_fixture(f)
    snapshot, references = f.assembly.build(f.request)
    assert len(references) == 4
    projected = {e.document_key for g in snapshot.relation_groups for e in g.evidence}
    projected.update(m.evidence.document_key for m in snapshot.mentions)
    # Mentions share the relation's highest-confidence documents.
    assert len(projected) == (2 if mention_only else 3)
    documents = {
        p.scope_key
        for p in snapshot.artifact_provenance
        if p.scope_type.value == "document"
    }
    assert len(documents) == 4 and documents - projected
    from apps.knowledge_graph.projection.frontier_snapshot_v2 import frontier_config
    from apps.knowledge_graph.retrieval.projected_relation_ranking_v2 import (
        rank_relations_v2,
    )

    rank_relations_v2(
        snapshot=snapshot,
        seeds=seeds,
        request=f.request,
        config=frontier_config(
            f.caps, f.ready.selected_generations[0].resolver_version
        ),
        frontier_context=f.assembly.context,
        deadline_check=f.budget.check_active,
    )
    omitted = next(iter(documents - projected))
    del f.assembly.provenance[omitted]
    with pytest.raises(ValueError, match="candidate.*provenance"):
        f.assembly.build(f.request)


@pytest.mark.parametrize("swap", [False, True])
def test_mention_only_snapshot_requires_its_own_document_proof(swap):
    f = multi_document_fixture(mention_only=True)
    acquire_fixture(f)
    snapshot, _ = f.assembly.build(f.request)
    document = snapshot.mentions[0].evidence.document_key
    rows = [p for p in snapshot.artifact_provenance if p.scope_key != document]
    if swap:
        # Another allowed document cannot stand in for the mentioned document.
        replacement = next(p for p in rows if p.scope_type.value == "document")
        rows.append(replace(replacement, artifact_key="a" * 64))
        rows.sort(key=lambda p: (p.scope_type.value, p.scope_key, p.artifact_key))
    with pytest.raises(ValueError):
        replace(snapshot, artifact_provenance=tuple(rows))


def test_repeated_conflicting_provenance_rejects_before_assembly_mutation():
    f = multi_document_fixture()
    f.session.initialize()
    bundle = f.bundles[0]
    origin = bundle.relations[0].source_entity_key
    copy, _ = next(f.session.copies(bundle.generation.generation_key, origin))
    stream = "relations/" + origin + "/knows"
    entry = next(
        f.session.entries(
            bundle.generation.generation_key,
            copy["links"]["relations"]["knows"],
            stream,
        )
    )
    page = next(
        f.session.relation(
            bundle.generation.generation_key, entry, stream=stream, depth=0
        )
    )
    f.assembly.add_relation(page, entry, bundle.generation.generation_key, origin, 0)
    before = (
        dict(f.assembly.hops),
        dict(f.assembly.members),
        dict(f.assembly.provenance),
        f.assembly.evidence_refs.build(),
    )
    changed = replace(
        page,
        provenance=tuple(
            replace(p, source_hash="a" * 64) if p.scope_type == "document" else p
            for p in page.provenance
        ),
    )
    with pytest.raises(ValueError, match="conflicting admitted document provenance"):
        f.assembly.add_relation(
            changed, entry, bundle.generation.generation_key, origin, 0
        )
    assert before == (
        f.assembly.hops,
        f.assembly.members,
        f.assembly.provenance,
        f.assembly.evidence_refs.build(),
    )
