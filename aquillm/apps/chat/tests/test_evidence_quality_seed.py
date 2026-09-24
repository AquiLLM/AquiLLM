"""Live seed preserves authored identities/history while current DB authority wins."""

import json
from pathlib import Path

import pytest

from apps.chat.evals.evidence_quality_eval import load_cases
from apps.chat.evals.evidence_quality_seed import (
    create_conversation,
    seed_cases,
    validate_case,
)


@pytest.mark.django_db
def test_seed_current_revisions_separate_authority_and_ordered_followup(
    tmp_path, monkeypatch
):
    from apps.chat.models import WSConversation
    from aquillm import utils
    from aquillm.message_adapters import load_conversation_from_db

    monkeypatch.setattr(utils, "get_embedding", lambda *a, **k: [0.1] * 1024)
    cases = load_cases(Path(__file__).parents[1] / "evals/evidence_quality_cases.json")
    cases = [cases[i] for i in (2, 3, 4, 6, 7)]
    manifest = seed_cases(cases, tmp_path / "seed.json")
    for case in cases:
        binding, sources = validate_case(case, manifest)
        assert len(sources) == len({s["source_id"] for s in case["sources"]})
        _, pk, _ = create_conversation(case, manifest)
        convo = load_conversation_from_db(WSConversation.objects.get(pk=pk))
        if case["scenario"] in ("plural_followup", "ordinal_citation"):
            tool = next(m for m in convo.messages if m.role == "tool")
            ids = [r["chunk_id"] for r in tool.result_dict["result"]]
            assert ids == [s["chunk_id"] for s in binding["sources"]]
    with pytest.raises(ValueError, match="exists"):
        seed_cases(cases, tmp_path / "seed.json")


@pytest.mark.django_db
@pytest.mark.parametrize("family", ["res", "tec", "rep", "pol"])
def test_frozen_plural_history_roundtrip_resolves_current_authorized_sources(
    tmp_path, monkeypatch, family
):
    from apps.chat.models import WSConversation
    from apps.chat.services.rag_retrieval import _verified_row_coordinates
    from apps.chat.services.rag_source_continuity import (
        rehydrate_prior_evidence,
        resolve_source_anchors,
    )
    from apps.collections.models import CollectionPermission
    from apps.collections.services.django_retrieval_authorization import (
        build_selected_scope_authorization_context,
    )
    from apps.documents.models import RawTextDocument, TextChunk
    from apps.documents.services.source_loading import (
        SourceRuntime,
        source_runtime_scope,
    )
    from aquillm import utils
    from aquillm.message_adapters import load_conversation_from_db
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits

    monkeypatch.setattr(utils, "get_embedding", lambda *a, **k: [0.1] * 1024)
    cases = load_cases(Path(__file__).parents[1] / "evals/evidence_quality_cases.json")
    case = next(c for c in cases if c["case_id"] == f"ep-dev-{family}-04")
    manifest = seed_cases([case], tmp_path / "seed.json")
    user, pk, _ = create_conversation(case, manifest)
    convo = load_conversation_from_db(WSConversation.objects.get(pk=pk))
    tool = next(m for m in convo.messages if m.role == "tool")
    rows = tool.result_dict["result"]
    binding = manifest["cases"][case["case_id"]]
    expected = tuple((s["chunk_id"], s["document_id"], 1) for s in binding["sources"])
    assert len(rows) == 2
    assert json.loads(tool.content)["result"] == rows
    assert tuple(_verified_row_coordinates(row) for row in rows) == tuple(
        (*identity, f"[doc:{identity[1]} chunk:{identity[0]}]") for identity in expected
    )
    anchors = resolve_source_anchors(case["question"], convo)
    assert anchors.basis == "answer_plural"
    assert anchors.unresolved_references == ()
    assert anchors.chunk_identities == expected

    scope = (binding["collection_id"],)
    authorization = build_selected_scope_authorization_context(
        principal=user,
        selected_collection_ids=scope,
        selected_documents=tuple(
            RawTextDocument.objects.filter(collection_id=scope[0])
        ),
    )
    assert authorization is not None
    runtime = SourceRuntime(TurnBudget(TurnLimits()), authorization)
    with source_runtime_scope(runtime):
        # Historical text identifies sources; current DB text still wins.
        TextChunk.objects.filter(pk=expected[0][0]).update(
            content="Current corrected cap"
        )
        sources = rehydrate_prior_evidence(
            anchors, user=user, selected_scope=scope, budget=runtime.budget
        )
        assert (
            tuple((s.chunk_id, s.document_id, s.chunk_number) for s in sources)
            == expected
        )
        assert sources[0].text == "Current corrected cap"
        assert sources[1].text == case["sources"][1]["text"]
        # Historical coordinates never grant access after current permission revocation.
        CollectionPermission.objects.filter(user=user, collection_id=scope[0]).delete()
        assert (
            rehydrate_prior_evidence(
                anchors, user=user, selected_scope=scope, budget=runtime.budget
            )
            == ()
        )
