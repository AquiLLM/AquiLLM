"""Live seed preserves authored identities/history while current DB authority wins."""

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
