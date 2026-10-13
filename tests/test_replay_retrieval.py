import asyncio
import json
from types import SimpleNamespace

import pytest

from apps.chat.evals import retrieval_replay as replay
from apps.chat.tests.test_rag_source_document_tools import docs as docs


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


@pytest.mark.parametrize("fail", [False, True])
def test_graph_off_restores_graph_on_settings_and_zero_candidate_summary(fail):
    settings = SimpleNamespace(RAG_CACHE_ENABLED=True, KG_OVERLAY_ENABLED=True)
    pipeline = SimpleNamespace(synthesize_from_evidence=object())
    original = pipeline.synthesize_from_evidence
    try:
        with replay.replay_runtime(
            "graph-off", settings, SimpleNamespace(), pipeline, lambda: None
        ):
            assert settings.KG_OVERLAY_ENABLED is False
            if fail:
                raise RuntimeError("simulated failure")
    except RuntimeError:
        pass
    assert settings.KG_OVERLAY_ENABLED is True
    assert settings.RAG_CACHE_ENABLED is True
    assert pipeline.synthesize_from_evidence is original
    summary = replay.summarize(
        [("retrieval_stages", {"graph": {"candidates": []}})], [], []
    )
    assert summary["stages"][0]["graph"]["unique_count"] == 0


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
            from apps.chat.services import retrieval_query_runtime

            await retrieval_query_runtime._rewrite("secret", llm, None)
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


def test_revision_override_is_validated_and_not_git_verified(monkeypatch):
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

    def git_success(command, **kwargs):
        assert command == ["git", "rev-parse", "HEAD"]
        assert kwargs["check"] is True
        return SimpleNamespace(stdout="b" * 40 + "\n")

    monkeypatch.setattr(cli.subprocess, "run", git_success)
    assert cli.revision_metadata(None) == {
        "git_revision": "b" * 40,
        "revision_provenance": "git_verified",
    }

    def git_failure(*args, **kwargs):
        raise cli.subprocess.CalledProcessError(128, args[0])

    monkeypatch.setattr(cli.subprocess, "run", git_failure)
    with pytest.raises(cli.subprocess.CalledProcessError):
        cli.revision_metadata(None)
    for invalid in ("HEAD", "a" * 39, "z" * 40):
        with pytest.raises(SystemExit):
            cli.parse_args(valid + ["--revision", invalid])


def test_graph_reasons_keep_canonical_labels_and_reject_untrusted_values():
    from apps.knowledge_graph.retrieval.branch_contracts import (
        DirectBranchFailureReason,
        ExtendedBranchFailureReason,
        SharedBranchFailureReason,
    )

    labels = [
        DirectBranchFailureReason.DIRECT_BRANCH_TIMEOUT.value,
        ExtendedBranchFailureReason.EXTENDED_BRANCH_TIMEOUT.value,
        DirectBranchFailureReason.EXTRACTOR_TIMEOUT.value,
        SharedBranchFailureReason.BACKEND_UNAVAILABLE.value,
    ]

    class Untrusted:
        def __str__(self):
            raise AssertionError("untrusted reason stringified")

    result = replay.summarize(
        [
            (
                "retrieval_stages",
                {
                    "graph": {
                        "reasons": [
                            *labels,
                            "secret source and exception message",
                            {"secret": "text"},
                            Untrusted(),
                        ]
                    }
                },
            )
        ],
        [],
        [],
    )
    assert result["stages"][0]["graph"]["reasons"] == labels
    assert "secret" not in json.dumps(result)


@pytest.mark.parametrize("reasons", [None, "extractor_timeout", {"secret": "text"}, 1])
def test_invalid_graph_reason_container_is_not_copied(reasons):
    result = replay.summarize(
        [("retrieval_stages", {"graph": {"reasons": reasons}})], [], []
    )
    assert result["stages"][0]["graph"]["reasons"] is None


def test_baseline_disables_ambient_query_proposals(monkeypatch):
    from apps.chat.services.retrieval_query_runtime import query_proposals_enabled

    monkeypatch.setenv("RAG_QUERY_REWRITE_ENABLED", "1")
    monkeypatch.setenv("RAG_VOCABULARY_QUERY_ENABLED", "1")
    with replay.replay_runtime(
        "baseline",
        SimpleNamespace(RAG_CACHE_ENABLED=True),
        SimpleNamespace(),
        SimpleNamespace(synthesize_from_evidence=None),
        None,
    ):
        assert query_proposals_enabled() is False
    assert query_proposals_enabled() is True


def test_current_modes_are_independent_and_restored(monkeypatch):
    from apps.chat.services.rag_config import rag_preservation_config
    from apps.chat.services.retrieval_query_runtime import configured_proposal_controls

    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "legacy")
    settings = SimpleNamespace(
        RAG_CACHE_ENABLED=True,
        KG_OVERLAY_ENABLED=False,
        KG_GRAPH_DIRECT_ENABLED=False,
        KG_GRAPH_EXTENDED_ENABLED=False,
    )
    with replay.replay_runtime(
        "current",
        settings,
        SimpleNamespace(),
        SimpleNamespace(synthesize_from_evidence=None),
        None,
        rerank_text_mode="windowed",
        evidence_text_mode="source",
        document_capacity_mode="budgeted",
        query_mode="vocabulary",
        graph_mode="both",
    ):
        preservation = rag_preservation_config()
        assert preservation.error is None
        assert preservation.active
        assert preservation.rerank_text_mode == "windowed"
        assert preservation.evidence_text_mode == "source"
        assert preservation.document_capacity_mode == "budgeted"
        assert configured_proposal_controls()["vocabulary"] is True
        assert configured_proposal_controls()["rewrite"] is False
        assert settings.KG_OVERLAY_ENABLED and settings.KG_GRAPH_DIRECT_ENABLED
        assert settings.KG_GRAPH_EXTENDED_ENABLED
    assert rag_preservation_config().evidence_text_mode == "legacy"
    assert settings.KG_OVERLAY_ENABLED is False


def test_limited_packet_is_a_failed_quality_outcome():
    from lib.evidence_observation import publish

    async def run(consumer, llm, convo, **kwargs):
        publish("rag_prepared_queries", {"queries": ["private"]})
        publish("rag_search_outcomes", {"queries": [{"status": "ok"}]})
        publish("rag_final_selection", {"limits": {}})
        await pipeline.synthesize_from_evidence(
            llm, convo, SimpleNamespace(chunks=[], retrieval_status="context_limited")
        )
        return "handled"

    pipeline = SimpleNamespace(run_direct_rag_turn=run, synthesize_from_evidence=None)
    result = asyncio.run(
        replay.run_turn(
            {"id": "Q1", "question": "private"},
            [],
            object(),
            [1],
            "baseline",
            SimpleNamespace(RAG_CACHE_ENABLED=True),
            SimpleNamespace(llm_interface=SimpleNamespace()),
            pipeline,
        )
    )
    assert result["success"] is False
    assert result["retrieval_status"] == "context_limited"


def test_budget_summary_omits_ledger_arguments_and_identities():
    result = replay.summarize(
        [
            (
                "ledger_start",
                {"ledger_id": "private", "limits": {"retrieval_ms": 15000}},
            ),
            (
                "ledger",
                {
                    "arguments": ["private"],
                    "after": {
                        "actions": 1,
                        "sources": 2,
                        "remaining_ms": 1250,
                        "closed": "deadline",
                    },
                },
            ),
            (
                "turn_complete",
                {"actions": 1, "planner_calls": 0, "stop_reason": "deadline"},
            ),
            (
                "query_proposals",
                {
                    "configured": {"vocabulary": True, "rewrite": False},
                    "query_digests": ["a" * 64],
                    "configured_model": "private",
                },
            ),
        ],
        [],
        [],
    )
    assert result["budget"]["limits"] == {"retrieval_ms": 15000}
    assert result["budget"]["observed"]["remaining_ms"] == 1250
    assert result["budget"]["completion"]["stop_reason"] == "deadline"
    assert result["proposals"]["query_count"] == 1
    assert "private" not in json.dumps(result)


def test_cli_current_modes_require_explicit_treatment_label():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "current_cli", Path(__file__).parents[1] / "scripts/replay_retrieval.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    base = [
        "--manifest",
        "questions.json",
        "--principal-id",
        "1",
        "--collection-ids",
        "1",
        "--output",
        "report.json",
    ]
    with pytest.raises(SystemExit):
        cli.parse_args(base + ["--evidence-text-mode", "source"])
    args = cli.parse_args(
        base
        + [
            "--experiment",
            "current",
            "--evidence-text-mode",
            "source",
            "--rerank-text-mode",
            "windowed",
            "--query-mode",
            "vocabulary",
        ]
    )
    assert args.evidence_text_mode == "source"
    assert args.rerank_text_mode == "windowed"
    assert args.query_mode == "vocabulary"


@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize(
    "modes",
    [
        {"evidence_text_mode": "source"},
        {"rerank_text_mode": "windowed"},
        {"document_capacity_mode": "budgeted"},
        {"query_mode": "vocabulary"},
        {
            "evidence_text_mode": "source",
            "rerank_text_mode": "windowed",
            "document_capacity_mode": "budgeted",
            "query_mode": "vocabulary",
        },
    ],
)
def test_current_real_pipeline_replay_has_packet_and_no_generation(
    docs, monkeypatch, modes
):
    from apps.chat.services import rag_pipeline
    from apps.chat.tests.test_rag_preservation_routes import fake_search

    user, document, chunks, _ = docs
    fake_search(monkeypatch)
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "32000")
    settings = SimpleNamespace(RAG_CACHE_ENABLED=True)
    config = SimpleNamespace(
        llm_interface=SimpleNamespace(base_args={"model": "private"})
    )
    result = asyncio.run(
        replay.run_turn(
            {"id": "Q1", "question": "Summarize the selected documents"},
            [],
            user,
            [document.collection_id],
            "current",
            settings,
            config,
            rag_pipeline,
            **modes,
        )
    )
    assert result["success"] is True, result
    assert result["packet"]
    assert result["budget"]["limits"]["retrieval_ms"] == 15000
    assert result["generation_attempted"] is False
    assert result["sdk_started"] is False
    assert "private" not in json.dumps(result)


def test_current_single_search_timing_includes_operation_interval(monkeypatch):
    import lib.evidence_observation as observation

    clock = [10.0]
    monkeypatch.setattr(observation, "perf_counter", lambda: clock[0])

    def search(consumer, question, top_k):
        clock[0] = 10.375
        return {"retrieval_status": "no_results"}

    async def run(consumer, llm, convo, **kwargs):
        pipeline._run_vector_search(consumer, "private", 10)
        await pipeline.synthesize_from_evidence(
            llm, convo, SimpleNamespace(chunks=[], retrieval_status="no_results")
        )
        return "handled"

    pipeline = SimpleNamespace(
        _run_vector_search=search,
        run_direct_rag_turn=run,
        synthesize_from_evidence=None,
    )
    result = asyncio.run(
        replay.run_turn(
            {"id": "Q1", "question": "private"},
            [],
            object(),
            [1],
            "current",
            SimpleNamespace(RAG_CACHE_ENABLED=True),
            SimpleNamespace(llm_interface=SimpleNamespace()),
            pipeline,
            evidence_text_mode="source",
        )
    )
    assert result["success"] is True
    assert result["timings_ms"]["retrieval"] == 375.0


@pytest.mark.parametrize(
    "provider_name,expected_context",
    [
        ("ClaudeInterface", 32000),
        ("GeminiInterface", 36000),
        ("OpenAIInterface", 2048),
    ],
)
def test_guard_preserves_provider_capacity_and_denies_generation(
    monkeypatch,
    provider_name,
    expected_context,
):
    from apps.chat.services.rag_context_budget import provider_context_capacity

    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "2048")
    monkeypatch.setenv("CLAUDE_CONTEXT_LIMIT", "32000")
    monkeypatch.setenv("GEMINI_CONTEXT_LIMIT", "36000")
    interface = type(provider_name, (), {"get_message": lambda self: None})()
    guard = replay.GenerationGuard(interface)
    assert provider_context_capacity(guard)[0] == expected_context
    assert provider_context_capacity(guard) == provider_context_capacity(interface)
    with pytest.raises(replay.GenerationDenied):
        guard.get_message()
    assert guard.attempted is True


@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize("provider_name", ["ClaudeInterface", "GeminiInterface"])
def test_real_source_replay_uses_provider_specific_context_budget(
    docs,
    monkeypatch,
    provider_name,
):
    from apps.chat.services import rag_pipeline
    from apps.chat.tests.test_rag_preservation_routes import fake_search

    user, document, _, _ = docs
    fake_search(monkeypatch)
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "2048")
    monkeypatch.setenv("CLAUDE_CONTEXT_LIMIT", "32000")
    monkeypatch.setenv("GEMINI_CONTEXT_LIMIT", "36000")
    interface = type(provider_name, (), {"get_message": lambda self: None})()
    result = asyncio.run(
        replay.run_turn(
            {"id": "Q1", "question": "Summarize the selected documents"},
            [],
            user,
            [document.collection_id],
            "current",
            SimpleNamespace(RAG_CACHE_ENABLED=True),
            SimpleNamespace(llm_interface=interface),
            rag_pipeline,
            evidence_text_mode="source",
        )
    )
    assert result["success"] is True, result
    assert result["packet"]
    assert result["generation_attempted"] is False
    assert result["sdk_started"] is False


def test_effective_report_distinguishes_graph_implementation(monkeypatch):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "graph_mode_cli", Path(__file__).parents[1] / "scripts/replay_retrieval.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    config = SimpleNamespace(vector_top_k=12, trigram_top_k=12)
    # Effective loaded settings determine dispatch, even if ambient values differ.
    monkeypatch.setenv("KG_GRAPH_REMEDIATION_MODE", "shadow")
    reports = [
        cli.effective_config(
            SimpleNamespace(RAG_CACHE_ENABLED=False, KG_GRAPH_REMEDIATION_MODE=mode),
            config,
        )
        for mode in ("legacy", "v2")
    ]
    assert reports[0]["graph_remediation_mode"] == "legacy"
    assert reports[1]["graph_remediation_mode"] == "v2"
    assert json.dumps(reports[0], sort_keys=True) != json.dumps(
        reports[1], sort_keys=True
    )
