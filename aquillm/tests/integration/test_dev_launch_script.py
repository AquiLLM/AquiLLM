"""Integration tests for development launch scripts."""

from pathlib import Path


def test_dev_launch_script_uses_serial_vllm_startup():
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "deploy" / "scripts" / "start_dev.sh"
    contents = script.read_text(encoding="utf-8")

    assert "wait_for_service_healthy vllm" in contents
    assert "wait_for_service_healthy vllm_transcribe" in contents
    assert "wait_for_service_healthy vllm_embed" in contents
    assert "wait_for_service_healthy vllm_rerank" in contents
    assert "wait_for_service_healthy vllm_ocr" not in contents


def test_dev_launch_script_supports_optional_edge_startup():
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "deploy" / "scripts" / "start_dev.sh"
    contents = script.read_text(encoding="utf-8")

    assert 'USE_EDGE="${USE_EDGE:-0}"' in contents
    assert 'compose_up get_certs' in contents
    assert 'compose_up nginx' in contents


def test_dev_launch_script_keeps_projection_worker_opt_in():
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "deploy" / "scripts" / "start_dev.sh"
    contents = script.read_text(encoding="utf-8")

    assert 'USE_KNOWLEDGE_GRAPH="${USE_KNOWLEDGE_GRAPH:-0}"' in contents
    assert (
        'USE_KNOWLEDGE_GRAPH_PROJECTION="${USE_KNOWLEDGE_GRAPH_PROJECTION:-0}"'
        in contents
    )
    projection_guard = 'if [ "$USE_KNOWLEDGE_GRAPH_PROJECTION" = "1" ]; then'
    assert projection_guard in contents
    assert contents.index(projection_guard) < contents.index(
        "compose_up worker_knowledge_graph_projection"
    )
