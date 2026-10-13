"""Offline execution of the real citation-enforcement block; no app imports or IO.

Run from the repository root: rtk proxy python docs/audits/2026-10-10-rag-adversarial/citation_probe.py
AST loading avoids application startup, database access, model calls, and settings.
Only the provider response and unrelated helpers are replaced with local stubs.
"""

import ast
import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]
PROVIDERS = ROOT / "aquillm/lib/llm/providers"


def compile_module(body, filename):
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.Module(body=[future, *body], type_ignores=[])
    return compile(ast.fix_missing_locations(module), str(filename), "exec")


def citation_namespace():
    path = PROVIDERS / "rag_citations.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    namespace = {"re": re, "getenv": lambda name, default="": default}
    body = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.Assign, ast.AnnAssign))
    ]
    exec(compile_module(body, path), namespace)
    extracts = PROVIDERS / "rag_citation_extracts.py"
    extract_tree = ast.parse(extracts.read_text(encoding="utf-8"))
    helper = [
        node for node in extract_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_first_citation_token"
    ]
    assert len(helper) == 1, "Citation extraction helper changed; inspect probe."
    exec(compile_module(helper, extracts), namespace)
    return namespace


def enforcement_block():
    path = PROVIDERS / "complete_turn.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    turns = [
        node for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "complete_conversation_turn"
    ]
    assert len(turns) == 1, "Completion entry point changed; inspect probe."
    blocks = [
        node for node in turns[0].body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "enforce_citations and (not response_tool_call)"
    ]
    assert len(blocks) == 1, "Citation block changed; inspect probe."
    block = blocks[0]
    names = {node.id for node in ast.walk(block) if isinstance(node, ast.Name)}
    assert {"retry_invalid", "citations_valid", "retry_response"} <= names
    assignments = sorted({
        node.id for node in ast.walk(block)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    })
    wrapped = ast.AsyncFunctionDef(
        name="run_real_block",
        args=ast.arguments(
            posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
        ),
        body=[
            ast.Global(names=assignments), block,
            ast.Return(value=ast.Name(id="response_text", ctx=ast.Load())),
        ],
        decorator_list=[],
    )
    return compile_module([wrapped], path), block.lineno, block.end_lineno


async def probe(repaired_text):
    namespace = citation_namespace()
    citations = SimpleNamespace(**namespace)
    allowed = {"[doc:a chunk:1]"}
    calls = []

    async def get_message(**kwargs):
        calls.append(kwargs.get("_synthesis_phase"))
        return SimpleNamespace(text=repaired_text, tool_call={})

    namespace.update(
        citations=citations, enforce_citations=True, response_tool_call={},
        provider_stream_func=None, current_stage=lambda: "direct_synthesis",
        response_text="The capacity is 999 litres.", citation_allowlist=allowed,
        fb=SimpleNamespace(is_high_quality_summary=lambda text: False),
        _latest_user_requested_image=lambda conversation: False,
        conversation=None, response_from_compact_summary=False,
        citation_retry_prior_max_chars=5000, message_dicts=[], messages_for_bot=[],
        AssistantMessage=lambda **kwargs: kwargs, UserMessage=lambda **kwargs: kwargs,
        llm=SimpleNamespace(base_args={}, get_message=get_message),
        request_system_prompt="system", post_tool_max_tokens=512, max_tokens=512,
        _resolve_post_tool_max_tokens=lambda *args, **kwargs: 512,
        _post_tool_global_max=lambda maximum: maximum,
        stream_message_uuid="offline-probe", _extractive_evidence_ui_enabled=lambda: False,
    )
    compiled, start, end = enforcement_block()
    exec(compiled, namespace)
    output = await namespace["run_real_block"]()
    valid = citations.response_has_required_citations(
        output, allowed, require_cited_numeric_claims=True
    )
    assert calls == ["citation_repair"], calls
    assert output == repaired_text, output
    assert valid is False, "Behavior changed; citation probe now passes validation."
    return {
        "real_block_lines": [start, end], "repair_calls": calls,
        "output": output, "passes_actual_validator": valid,
    }


async def main():
    for answer in (
        "The capacity is 999 litres.",
        "A supported statement [doc:a chunk:1]. The capacity is 999 litres.",
    ):
        print(json.dumps(await probe(answer)))


if __name__ == "__main__":
    asyncio.run(main())
