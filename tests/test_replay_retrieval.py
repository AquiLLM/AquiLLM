import asyncio
import json
from types import SimpleNamespace

import pytest

from apps.chat.evals import retrieval_replay as replay


def test_manifest_required_and_validated(tmp_path):
    with pytest.raises(ValueError):
        replay.load_manifest(tmp_path / "missing")
    path = tmp_path / "manifest.json"
    for value in (
        {},
        {"version": 2, "questions": []},
        {"version": 1, "questions": [{"id": "Q1", "question": ""}]},
    ):
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError):
            replay.load_manifest(path)
    path.write_text(
        json.dumps({"version": 1, "questions": [{"id": "Q1", "question": "Why?"}]})
    )
    assert replay.load_manifest(path)[0]["id"] == "Q1"


def test_guard_denies_every_callable_including_rewrite():
    guard = replay.GenerationGuard(
        SimpleNamespace(rewrite=lambda: None, token_count=lambda: 3, model="x")
    )
    for name in ("rewrite", "token_count"):
        with pytest.raises(replay.GenerationDenied):
            getattr(guard, name)()
    assert guard.attempted
    assert guard.model == "x"


def test_mapping_rejects_missing_and_ambiguous_sources():
    target = {
        "source_sha256": replay.fingerprint("source"),
        "text_sha256": replay.fingerprint("answer"),
    }
    source = {"source": "source", "chunks": [{"chunk_id": 1, "text": "answer"}]}
    with pytest.raises(replay.MappingError):
        replay.resolve_targets([target], [])
    with pytest.raises(replay.MappingError):
        replay.resolve_targets([target], [source, source])
    assert replay.resolve_targets([target], [source])[0]["chunk_ids"] == [1]


def test_membership_and_span_survival_are_independent_and_private():
    target = {
        "source_sha256": replay.fingerprint("source"),
        "answer_sha256": replay.fingerprint("answer"),
        "start": 0,
        "end": 6,
    }
    targets = [dict(target, chunk_ids=[8], answer="answer")]
    events = [
        (
            "retrieval_stages",
            {
                "at": 1,
                "materialized_union": [
                    {"chunk_id": 8, "text": "answer elsewhere", "rank": 2}
                ],
                "post_rerank": [],
                "graph": {
                    "ready": None,
                    "candidates": [],
                    "branch_statuses": {"direct": "failed"},
                },
                "effective_limits": {"vector": 12},
            },
        )
    ]
    result = replay.summarize(events, [{"chunk_id": 8, "text": "elsewhere"}], targets)
    assert result["targets"][0] == {
        "source_sha256": target["source_sha256"],
        "pool_ranks": [2],
        "rerank_ranks": [],
        "packet_ranks": [1],
        "answer_span_covered": False,
    }
    serialized = json.dumps(result)
    assert "elsewhere" not in serialized and "chunk_id" not in serialized
    assert result["stages"][0]["graph"]["ready"] is None


@pytest.mark.parametrize(
    "outcome,failed,sdk",
    [("skipped", False, False), ("handled", True, False), ("handled", False, True)],
)
def test_failed_run_remains_failure(outcome, failed, sdk):
    assert not replay.turn_succeeded(
        outcome,
        packet_recorded=True,
        observer_failed=failed,
        sdk_started=sdk,
        generation_attempted=False,
    )


def test_settings_and_synthesis_restored_on_exception(monkeypatch):
    settings = SimpleNamespace(RAG_CACHE_ENABLED=True, KG_OVERLAY_ENABLED=True)
    config = SimpleNamespace(vector_top_k=12, trigram_top_k=12)
    pipeline = SimpleNamespace(synthesize_from_evidence=object())
    original = pipeline.synthesize_from_evidence
    monkeypatch.setenv("APP_RERANK_DOC_CHAR_LIMIT", "1600")
    with pytest.raises(RuntimeError):
        with replay.replay_runtime(
            "depth60-full", settings, config, pipeline, lambda: None
        ):
            assert settings.RAG_CACHE_ENABLED is False
            assert config.vector_top_k == 60
            raise RuntimeError("private text")
    assert settings.RAG_CACHE_ENABLED is True
    assert config.vector_top_k == 12
    assert pipeline.synthesize_from_evidence is original
    import os

    assert os.environ["APP_RERANK_DOC_CHAR_LIMIT"] == "1600"


def test_late_events_cannot_leak_into_next_turn():
    first, second = replay.EventRecorder(), replay.EventRecorder()
    first("rag_prepared_queries", {"at": 1})
    first.close()
    first("retrieval_stages", {"at": 2, "query": "private"})
    assert len(first.events) == 1 and not second.events


def test_compact_packet_text_is_measured():
    assert (
        replay.summarize(
            [],
            [{"i": 8, "x": "answer"}],
            [{"source_sha256": "a" * 64, "chunk_ids": [8], "answer": "answer"}],
        )["targets"][0]["answer_span_covered"]
        is True
    )


@pytest.mark.parametrize(
    "behavior",
    ["ok", "skip", "raise", "sdk", "generation", "rewrite", "search-error", "observer"],
)
def test_run_turn_records_and_fails_closed(behavior):
    from lib.evidence_observation import publish

    settings = SimpleNamespace(RAG_CACHE_ENABLED=True)
    config = SimpleNamespace(llm_interface=SimpleNamespace(rewrite=lambda: None))

    async def pipeline_run(consumer, llm, convo, **kwargs):
        if behavior == "raise":
            raise RuntimeError("secret")
        if behavior == "skip":
            return "skipped"
        if behavior == "sdk":
            publish("sdk_start", {})
        if behavior == "generation":
            try:
                llm.rewrite()
            except replay.GenerationDenied:
                pass
        if behavior == "rewrite":
            from apps.chat.services import rag_query

            rag_query._llm_rewrite_query("secret", convo)
        publish("rag_prepared_queries", {"queries": ["secret"], "candidate_top_k": 15})
        publish(
            "rag_search_outcomes",
            {"queries": [{"status": "error" if behavior == "search-error" else "ok"}]},
        )
        if behavior == "observer":
            from lib.evidence_observation import mark_failed

            mark_failed()
        publish("rag_final_selection", {"rows": [], "limits": {}})
        await pipeline.synthesize_from_evidence(llm, convo, SimpleNamespace(chunks=[]))
        return "handled"

    pipeline = SimpleNamespace(
        run_direct_rag_turn=pipeline_run, synthesize_from_evidence=object()
    )
    result = asyncio.run(
        replay.run_turn(
            {"id": "Q1", "question": "secret"},
            [],
            object(),
            [1],
            "baseline",
            settings,
            config,
            pipeline,
        )
    )
    assert result["success"] is (behavior == "ok")
    assert "secret" not in json.dumps(result)
    assert settings.RAG_CACHE_ENABLED is True


def test_cli_rejects_unbounded_inputs_and_requires_output():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "replay_cli", Path(__file__).parents[1] / "scripts/replay_retrieval.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    valid = [
        "--manifest",
        "questions.json",
        "--principal-id",
        "1",
        "--collection-ids",
        "1",
        "--output",
        "report.json",
    ]
    assert cli.parse_args(valid).repetitions == 1
    for suffix in (
        ["--repetitions", "11"],
        ["--principal-id", "0"],
        ["--collection-ids", *map(str, range(1, 66))],
    ):
        with pytest.raises(SystemExit):
            cli.parse_args(valid + suffix)
    with pytest.raises(SystemExit):
        cli.parse_args(valid[:-2])


def test_answer_mapping_validates_document_offsets():
    text = "prefix answer suffix"
    source = {
        "source": text,
        "chunks": [{"chunk_id": 1, "text": text, "start": 0, "end": len(text)}],
    }
    target = {
        "source_sha256": replay.fingerprint(text),
        "answer_sha256": replay.fingerprint("answer"),
        "start": 7,
        "end": 13,
    }
    assert replay.resolve_targets([target], [source])[0]["answer"] == "answer"
    with pytest.raises(replay.MappingError):
        replay.resolve_targets([{**target, "end": 12}], [source])


def test_cli_writes_failure_for_missing_manifest_without_database(tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "replay_failure_cli", Path(__file__).parents[1] / "scripts/replay_retrieval.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    output = tmp_path / "report.json"
    assert (
        cli.main(
            [
                "--manifest",
                str(tmp_path / "missing"),
                "--principal-id",
                "1",
                "--collection-ids",
                "1",
                "--output",
                str(output),
            ]
        )
        == 1
    )
    report = json.loads(output.read_text())
    assert report["success"] is False and report["error_type"] == "ValueError"


def test_exact_json_question_preserved(tmp_path):
    path = tmp_path / "questions.json"
    question = "Why is the model’s answer different?"
    path.write_text(
        json.dumps({"version": 1, "questions": [{"id": "Q1", "question": question}]}),
        encoding="utf-8",
    )
    assert replay.load_manifest(path)[0]["question"] == question


def test_real_pipeline_search_error_dict_is_not_success(monkeypatch):
    from apps.chat.services import rag_pipeline

    monkeypatch.setattr(
        rag_pipeline,
        "_run_vector_search",
        lambda *args: {"exception": "secret backend failure"},
    )
    settings = SimpleNamespace(RAG_CACHE_ENABLED=True)
    config = SimpleNamespace(llm_interface=SimpleNamespace())
    result = asyncio.run(
        replay.run_turn(
            {"id": "Q1", "question": "Search the documents for calibration"},
            [],
            object(),
            [1],
            "baseline",
            settings,
            config,
            rag_pipeline,
        )
    )
    assert result["success"] is False
    assert "secret backend failure" not in json.dumps(result)


def test_revision_override_is_validated_and_not_git_verified():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "revision_cli", Path(__file__).parents[1] / "scripts/replay_retrieval.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    valid = [
        "--manifest",
        "questions.json",
        "--principal-id",
        "1",
        "--collection-ids",
        "1",
        "--output",
        "report.json",
    ]
    assert cli.parse_args(valid + ["--revision", "a" * 40]).revision == "a" * 40
    assert cli.revision_metadata("a" * 40) == {
        "git_revision": "a" * 40,
        "revision_provenance": "operator_supplied",
    }
    assert cli.revision_metadata(None)["revision_provenance"] == "git_verified"
    for invalid in ("HEAD", "a" * 39, "z" * 40):
        with pytest.raises(SystemExit):
            cli.parse_args(valid + ["--revision", invalid])
