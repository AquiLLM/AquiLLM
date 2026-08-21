from __future__ import annotations

from pathlib import Path

SHELL = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_task21_hybrid_cloud_eval.sh"
)


def test_live_shell_seeds_builds_attests_then_runs_production_five_arms():
    source = SHELL.read_text(encoding="utf-8")
    precompose = source[: source.index("compose=(")]

    assert 'FIXTURE_MANIFEST="$WORK_ROOT/fixture-manifest.json"' in source
    assert 'RUNTIME_IDENTITY="$WORK_ROOT/runtime-identity.json"' in source
    assert '>"$ATTESTATION"' not in precompose
    assert '>"$LIVE_TRACE"' not in precompose
    assert (
        "activate_knowledge_graph_ontology --path research-v1.yaml "
        "--expected-checksum "
        "eb8d0c6b512216db2592f16898cd59ab76a2c95e9151c5fabfcc3f1be87a9059"
    ) in source
    assert "seed_knowledge_graph_eval_fixture --fixture-manifest" in source
    assert "-e KG_BUILD_ENABLED=0 -e KG_OVERLAY_ENABLED=0" in source
    assert "-e KG_EVAL_BYPASS_ALLOWED=1 -e COHERE_KEY=" in source
    assert "rebuild_knowledge_graph --collection" in source
    assert "inspect_knowledge_graph --request-id" in source
    assert "--wait --timeout-seconds 1800" in source
    assert "task21_hybrid_runtime_identity.py" in source
    assert '--fixture-manifest "/app/$WORK_REL/fixture-manifest.json"' in source
    assert '--runtime-identity "/app/$WORK_REL/runtime-identity.json"' in source
    assert "-e KG_OVERLAY_ENABLED=1 -e KG_MEMGRAPH_TRAVERSAL_ENABLED=1" in source
    assert "-e KG_GRAPH_DIRECT_ENABLED=1 -e KG_GRAPH_EXTENDED_ENABLED=1" in source
    generator_call = source.rindex(
        "apps.knowledge_graph.evals.task21_hybrid_live_observations"
    )
    assert source.index("seed_knowledge_graph_eval_fixture") < source.index(
        "rebuild_knowledge_graph --collection"
    ) < source.index("project_knowledge_graph --all") < source.index(
        "task21_hybrid_runtime_identity.py"
    ) < generator_call
