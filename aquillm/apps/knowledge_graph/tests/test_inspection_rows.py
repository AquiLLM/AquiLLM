from __future__ import annotations

from types import SimpleNamespace


class _FakeQuery:
    def __init__(self, rows):
        self.rows = tuple(rows)
        self.related = ()

    def select_related(self, *fields):
        self.related = fields
        return self

    def __getitem__(self, index):
        return self.rows[index]


def _run(*, pk: int, evaluation_only: bool):
    artifact = SimpleNamespace(pk=pk + 100)
    return SimpleNamespace(
        pk=pk,
        artifact_id=artifact.pk,
        artifact=artifact,
        build_kind="document",
        scope_type="document",
        scope_id=f"00000000-0000-4000-8000-{pk:012d}",
        stage="superseded" if evaluation_only else "active",
        status="cancelled" if evaluation_only else "succeeded",
        error_code="",
        rebuild_request_id=None,
        evaluation_only=evaluation_only,
    )


def test_build_inspection_rows_derives_exact_eval_completion(monkeypatch):
    from apps.knowledge_graph.services import builds, inspection_rows

    evaluation = _run(pk=1, evaluation_only=True)
    production = _run(pk=2, evaluation_only=False)
    observed = []

    def completed(artifact, run, *, build_kind):
        observed.append((artifact, run, build_kind))
        return True

    monkeypatch.setattr(builds, "_evaluation_occurrence_completed", completed)
    query = _FakeQuery((evaluation, production))

    rows = inspection_rows.bounded_build_inspection_rows(query, maximum=2)

    assert query.related == ("artifact",)
    assert tuple(row["pk"] for row in rows) == (1, 2)
    assert rows[0]["evaluation_completed"] is True
    assert rows[1]["evaluation_completed"] is False
    assert observed == [(evaluation.artifact, evaluation, "document")]
