"""Source-bound upstream correction for mixed speculative GDN gate ordering.

The recurrence and state metadata stay unchanged. Only mixed batches gather
their speculative gates with the same indices already used to gather QKV.
Upstream Apache-2.0 implementation, verified 2026-10-10 at immutable revision:
https://github.com/vllm-project/vllm/blob/276fbcff2717bd934cfa37c8a2e4c391f3e7237b/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py#L1309-L1319
Related upstream restoration: https://github.com/vllm-project/vllm/commit/38a1c179
"""
from __future__ import annotations

import ast
import functools
import hashlib
import importlib
import inspect
import json
import logging
import textwrap


log = logging.getLogger("aquillm.h100")
_MODULE = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
# AST of the complete _forward_core captured from vLLM 2dfaae752 after
# Genesis 34e269301. Ignore whitespace/comments, reject every semantic drift.
_FINGERPRINT = "8af7d81452b2f73fac77821e865714baa5c3e9a704d2d42cae562b7304642da1"
_PARAMETERS = ("self", "mixed_qkv", "b", "a", "core_attn_out")


def _source_fingerprint(tree):
    # ast.dump's empty-field defaults changed in Python 3.13. Include every
    # semantic field explicitly for the pinned serving Python 3.12 as well.
    def canonical(value):
        if isinstance(value, ast.AST):
            return {"type": type(value).__name__, "fields": {
                key: canonical(item) for key, item in ast.iter_fields(value)
                if not (key == "type_params" and item == [])}}
        if isinstance(value, list):
            return [canonical(item) for item in value]
        if value is Ellipsis:
            return {"literal": "Ellipsis"}
        return value
    encoded = json.dumps(canonical(tree), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def rewrite_gdn_method(original, source: str):
    """Validate the known full method, then apply exactly the upstream delta."""
    if not inspect.isfunction(original) or original.__closure__:
        raise ValueError("mixed GDN function contract mismatch")
    if tuple(inspect.signature(original).parameters) != _PARAMETERS:
        raise ValueError("mixed GDN signature contract mismatch")
    tree = ast.parse(textwrap.dedent(source))
    if _source_fingerprint(tree) != _FINGERPRINT:
        raise ValueError("post-Genesis mixed GDN method fingerprint mismatch")
    node = tree.body[0]
    if not isinstance(node, ast.FunctionDef) or node.name != "_forward_core" or node.decorator_list:
        raise ValueError("mixed GDN definition contract mismatch")
    branches = [n.body[0] for n in ast.walk(node) if isinstance(n, ast.If)
                and ast.unparse(n.test) == "spec_sequence_masks is not None"
                and n.body and isinstance(n.body[0], ast.If)
                and ast.unparse(n.body[0].test) ==
                    "attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0"]
    calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name)
             and n.func.id == "fused_sigmoid_gating_delta_rule_update"
             and any(k.arg == "q" and isinstance(k.value, ast.Name)
                     and k.value.id == "query_spec" for k in n.keywords)]
    if len(branches) != 1 or len(calls) != 1:
        raise ValueError("mixed GDN unique branch contract mismatch")
    branch = branches[0]
    # Pure-spec aliases add no allocations; pure-nonspec never enters this branch.
    branch.body[1:1] = ast.parse("a_spec = a\nb_spec = b").body
    branch.orelse[1:1] = ast.parse(
        "a_spec = a.index_select(0, spec_token_indx)\n"
        "b_spec = b.index_select(0, spec_token_indx)").body
    for kw in calls[0].keywords:
        if kw.arg in ("a", "b"):
            kw.value = ast.Name(id=kw.arg + "_spec", ctx=ast.Load())
    namespace = dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree), "<aquillm-h100-gdn-mixed>", "exec"), namespace)
    rewritten = functools.update_wrapper(namespace[node.name], original)
    rewritten._aquillm_h100_gdn_mixed = (_FINGERPRINT, original)
    return rewritten


def install_gdn_mixed() -> dict:
    """Startup only, after Genesis and the existing exact runtime gate.

    SystemExit deliberately crosses bootstrap's Exception handler: a candidate
    with an unfamiliar method must fail startup, not serve with a skipped fix.
    """
    try:
        module = importlib.import_module(_MODULE)
        cls = module.QwenGatedDeltaNetAttention
        original = cls._forward_core
        marker = getattr(original, "_aquillm_h100_gdn_mixed", None)
        if marker is not None:
            if (not isinstance(marker, tuple) or len(marker) != 2
                    or marker[0] != _FINGERPRINT
                    or getattr(original, "__wrapped__", None) is not marker[1]):
                raise ValueError("mixed GDN installed marker contract mismatch")
            return {"installed": True, "reason": "already_installed"}
        rewritten = rewrite_gdn_method(original, inspect.getsource(original))
    except (ImportError, AttributeError, ValueError, TypeError, OSError, SyntaxError) as error:
        raise SystemExit(f"AquiLLM H100 mixed GDN startup contract failed: {error}") from error
    cls._forward_core = rewritten
    log.warning("AQUILLM_H100 gdn_mixed_gate_alignment installed source=%s", _FINGERPRINT)
    return {"installed": True, "reason": "pinned_post_genesis_upstream_gate_alignment"}
