"""Salience-aware OpenAI-shaped message packing before provider preflight trim (fail-open)."""
from __future__ import annotations

import copy
import structlog
import re

from aquillm.metrics import context_pack_tokens
from dataclasses import dataclass
from typing import Any

from tiktoken import encoding_for_model

from lib.llm.providers.openai_tokens import context_reserve_tokens, estimate_prompt_tokens

logger = structlog.stdlib.get_logger(__name__)
_ENC = encoding_for_model("gpt-4o")


@dataclass
class ContextPackerConfig:
    budget_history_tokens: int = 12000
    budget_tool_evidence_tokens: int = 1400
    budget_retrieval_tokens: int = 3500
    pin_last_turns: int = 2
    max_snippets_per_doc: int = 3


def load_context_packer_config() -> ContextPackerConfig:
    try:
        from django.conf import settings

        return ContextPackerConfig(
            budget_history_tokens=int(getattr(settings, "CONTEXT_BUDGET_HISTORY_TOKENS", 12000)),
            budget_tool_evidence_tokens=int(
                getattr(settings, "CONTEXT_BUDGET_TOOL_EVIDENCE_TOKENS", 1400)
            ),
            budget_retrieval_tokens=int(getattr(settings, "CONTEXT_BUDGET_RETRIEVAL_TOKENS", 3500)),
            pin_last_turns=max(1, int(getattr(settings, "CONTEXT_PIN_LAST_TURNS", 2))),
            max_snippets_per_doc=max(1, int(getattr(settings, "CONTEXT_MAX_SNIPPETS_PER_DOC", 3))),
        )
    except Exception:
        return ContextPackerConfig()


def _is_tool_evidence(msg: dict[str, Any]) -> bool:
    if not isinstance(msg, dict):
        return False
    if str(msg.get("role", "")).lower() != "user":
        return False
    content = msg.get("content")
    if not isinstance(content, str):
        return False
    head = content.lstrip()[:160]
    first_line = head.split("\n", 1)[0]
    if first_line.startswith("Tool ") and "result:" in first_line:
        return True
    return first_line.startswith("Tool:")


def _assistant_has_tool_calls(msg: dict[str, Any]) -> bool:
    if str(msg.get("role", "")).lower() != "assistant":
        return False
    tc = msg.get("tool_calls")
    return isinstance(tc, list) and len(tc) > 0


def _compute_pinned_indices(msgs: list[dict[str, Any]], cfg: ContextPackerConfig) -> set[int]:
    n = len(msgs)
    pinned: set[int] = set()
    if n == 0:
        return pinned
    pinned.add(n - 1)
    i = n - 2
    if i >= 0 and _is_tool_evidence(msgs[i]):
        while i >= 0 and _is_tool_evidence(msgs[i]):
            pinned.add(i)
            i -= 1
        if i >= 0 and _assistant_has_tool_calls(msgs[i]):
            pinned.add(i)

    primary_user_idxs = [
        j
        for j in range(n)
        if str(msgs[j].get("role", "")).lower() == "user" and not _is_tool_evidence(msgs[j])
    ]
    if primary_user_idxs:
        tail_users = primary_user_idxs[-cfg.pin_last_turns :]
        cut = tail_users[0]
        for j in range(cut, n):
            pinned.add(j)
    return pinned


def _estimate_list_tokens(messages: list[dict[str, Any]]) -> int:
    return estimate_prompt_tokens(messages, _ENC)


def _dedupe_adjacent_tool_headers(msgs: list[dict[str, Any]]) -> bool:
    changed = False
    for i in range(len(msgs) - 1):
        if not (_is_tool_evidence(msgs[i]) and _is_tool_evidence(msgs[i + 1])):
            continue
        c0 = msgs[i].get("content")
        c1 = msgs[i + 1].get("content")
        if not isinstance(c0, str) or not isinstance(c1, str):
            continue
        lines0, lines1 = c0.splitlines(), c1.splitlines()
        k = 0
        n = min(len(lines0), len(lines1), 16)
        while k < n and lines0[k] == lines1[k]:
            k += 1
        if k >= 2:
            msgs[i + 1]["content"] = "\n".join(lines1[k:])
            changed = True
    return changed


def _collapse_excess_newlines(msgs: list[dict[str, Any]]) -> bool:
    changed = False
    for m in msgs:
        c = m.get("content")
        if isinstance(c, str) and re.search(r"\n{4,}", c):
            m["content"] = re.sub(r"\n{3,}", "\n\n", c)
            changed = True
    return changed


def _extractive_sentences_low_salience(
    msgs: list[dict[str, Any]],
    salience_scores: dict[int, float],
) -> bool:
    vals = sorted(salience_scores.values())
    median = vals[len(vals) // 2] if vals else 0.0
    changed = False
    for i, m in enumerate(msgs):
        if _is_tool_evidence(m):
            continue
        c = m.get("content")
        if not isinstance(c, str) or len(c) < 500:
            continue
        if salience_scores.get(i, 0.0) > median:
            continue
        parts = re.split(r"(?<=[.!?])\s+", c.strip())
        if len(parts) <= 2:
            continue
        short = " ".join(parts[:2]).strip()
        if len(short) + 24 < len(c):
            m["content"] = short + "\n[Context shortened.]"
            changed = True
    return changed


def _lm_compress_low_salience(
    msgs: list[dict[str, Any]],
    salience_scores: dict[int, float],
) -> bool:
    try:
        from lib.llm.optimizations.lm_lingua2_adapter import (
            compress_plain_text_for_prompt,
            lm_lingua2_enabled,
        )
    except Exception:
        return False
    if not lm_lingua2_enabled():
        return False
    vals = sorted(salience_scores.values())
    median = vals[len(vals) // 2] if vals else 0.0
    changed = False
    for i, m in enumerate(msgs):
        if _is_tool_evidence(m):
            continue
        c = m.get("content")
        if not isinstance(c, str):
            continue
        if salience_scores.get(i, 0.0) >= median:
            continue
        out = compress_plain_text_for_prompt(c)
        if out and len(out) < len(c):
            m["content"] = out
            changed = True
    return changed


def _run_staged_pruning(
    msgs: list[dict[str, Any]],
    salience_scores: dict[int, float] | None,
) -> list[str]:
    stages: list[str] = []
    if _dedupe_adjacent_tool_headers(msgs):
        stages.append("dedupe")
    if _collapse_excess_newlines(msgs):
        stages.append("boilerplate")
    if salience_scores and _extractive_sentences_low_salience(msgs, salience_scores):
        stages.append("extractive")
    if salience_scores and _lm_compress_low_salience(msgs, salience_scores):
        stages.append("lm_lingua2")
    return stages


def _prompt_budget_tokens(context_limit: int, max_tokens: int, slack: int) -> int:
    if context_limit <= 0 or max_tokens <= 0:
        return 0
    guard_tokens, estimator_pad_tokens = context_reserve_tokens(context_limit)
    slack = max(0, int(slack))
    raw = context_limit - int(max_tokens) - guard_tokens - estimator_pad_tokens - slack
    return max(0, raw)


def _tokens_for_indices(msgs: list[dict[str, Any]], indices: set[int]) -> int:
    sub = [msgs[i] for i in sorted(indices) if 0 <= i < len(msgs)]
    return _estimate_list_tokens(sub) if sub else 0


def _pack_core(
    system_text: str,
    message_dicts: list[dict[str, Any]],
    context_limit: int,
    max_tokens: int,
    cfg: ContextPackerConfig,
    slack: int,
    salience_scores: dict[int, float] | None,
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    msgs = copy.deepcopy(message_dicts)
    system_msg = [{"role": "system", "content": system_text}]
    budget = _prompt_budget_tokens(context_limit, max_tokens, slack)
    stats: dict[str, Any] = {
        "pinned_count": 0,
        "dropped_history": 0,
        "before_tokens": 0,
        "after_tokens": 0,
        "stage_fit": None,
        "stages_applied": [],
    }

    def total_prompt_tokens() -> int:
        return _estimate_list_tokens(system_msg + msgs)

    stats["before_tokens"] = total_prompt_tokens()
    if budget <= 0 or not msgs:
        stats["after_tokens"] = stats["before_tokens"]
        stats["pinned_count"] = len(_compute_pinned_indices(msgs, cfg))
        return msgs, max_tokens, stats

    hist_cap = max(256, cfg.budget_history_tokens)
    tool_cap = max(128, cfg.budget_tool_evidence_tokens)
    denom = hist_cap + tool_cap
    avail = budget - _estimate_list_tokens(system_msg)
    if avail < 256:
        avail = max(256, budget // 2)
    if denom > 0:
        scale = avail / denom
        hist_cap = max(256, int(hist_cap * scale))
        tool_cap = max(128, int(tool_cap * scale))

    stats["stages_applied"].extend(_run_staged_pruning(msgs, salience_scores))

    pinned = _compute_pinned_indices(msgs, cfg)
    stats["pinned_count"] = len(pinned)

    def tool_indices() -> set[int]:
        return {i for i in range(len(msgs)) if _is_tool_evidence(msgs[i])}

    def history_indices() -> set[int]:
        return {i for i in range(len(msgs)) if not _is_tool_evidence(msgs[i])}

    def drop_priority_order(idxs: list[int]) -> list[int]:
        if salience_scores is None:
            return sorted(idxs)
        return sorted(idxs, key=lambda i: (salience_scores.get(i, 0.0), i))

    def remove_best_candidate(pool: set[int]) -> bool:
        nonlocal pinned, salience_scores
        unpinned = [i for i in pool if i not in pinned]
        if not unpinned:
            return False
        victim = drop_priority_order(unpinned)[0]
        del msgs[victim]
        pinned = {p if p < victim else p - 1 for p in pinned if p != victim}
        if salience_scores:
            remapped: dict[int, float] = {}
            for k, v in salience_scores.items():
                if k == victim:
                    continue
                nk = k if k < victim else k - 1
                remapped[nk] = v
            salience_scores.clear()
            salience_scores.update(remapped)
        return True

    orig_len = len(msgs)
    while True:
        t_idx = tool_indices()
        h_idx = history_indices()
        tool_tok = _tokens_for_indices(msgs, t_idx)
        hist_tok = _tokens_for_indices(msgs, h_idx)
        tot = total_prompt_tokens()
        if tot <= budget and tool_tok <= tool_cap and hist_tok <= hist_cap:
            break
        if tool_tok > tool_cap:
            if remove_best_candidate(t_idx):
                continue
            tool_cap = tool_tok
            continue
        if hist_tok > hist_cap:
            if remove_best_candidate(h_idx):
                continue
            hist_cap = hist_tok
            continue
        if tot > budget:
            if not remove_best_candidate(set(range(len(msgs)))):
                break
            continue
        break

    stats["dropped_history"] = orig_len - len(msgs)

    trim_loops = 0
    while total_prompt_tokens() > budget and trim_loops < 48:
        victim: int | None = None
        best = 0
        for i, m in enumerate(msgs):
            if i in pinned:
                continue
            c = m.get("content")
            if isinstance(c, str) and len(c) > best:
                best = len(c)
                victim = i
        if victim is None:
            break
        overflow = max(1, total_prompt_tokens() - budget)
        trim_chars = max(128, overflow * 12)
        c = str(msgs[victim].get("content", ""))
        if len(c) <= trim_chars:
            msgs[victim]["content"] = "[Earlier context trimmed due to token limit.]"
        else:
            msgs[victim]["content"] = c[trim_chars:]
        trim_loops += 1
        stats["stages_applied"].append("hard_trim")

    stats["after_tokens"] = total_prompt_tokens()
    if stats["after_tokens"] <= budget:
        stats["stage_fit"] = stats["stages_applied"][-1] if stats["stages_applied"] else "pack"
    return msgs, max_tokens, stats


def pack_messages_for_budget(
    system_text: str,
    message_dicts: list[dict[str, Any]],
    context_limit: int,
    max_tokens: int,
    cfg: ContextPackerConfig | None = None,
    *,
    slack: int = 384,
    salience_scores: dict[int, float] | None = None,
) -> dict[str, Any]:
    """
    Return {"messages": packed, "max_tokens": adjusted, "stats": {...}}.
    On any error, returns the original list unchanged (fail-open).
    """
    cfg = cfg or ContextPackerConfig()
    try:
        active_scores = salience_scores
        if active_scores is None:
            from lib.llm.utils.context_salience import build_salience_scores

            built = build_salience_scores(message_dicts)
            active_scores = dict(built) if built else None
        msgs, mt, stats = _pack_core(
            system_text,
            message_dicts,
            context_limit,
            max_tokens,
            cfg,
            slack,
            active_scores,
        )
        context_pack_tokens.labels(stage="before").set(stats.get("before_tokens", 0))
        context_pack_tokens.labels(stage="after").set(stats.get("after_tokens", 0))
        logger.info(
            "obs.llm.context_pack",
            before_tokens=stats.get("before_tokens"),
            after_tokens=stats.get("after_tokens"),
            pinned_count=stats.get("pinned_count"),
            dropped_history=stats.get("dropped_history"),
            stage_fit=stats.get("stage_fit"),
            stages=",".join(stats.get("stages_applied") or []),
        )
        return {"messages": msgs, "max_tokens": mt, "stats": stats}
    except Exception as exc:
        logger.warning("obs.llm.context_pack_error", error_type=type(exc).__name__)
        return {
            "messages": copy.deepcopy(message_dicts),
            "max_tokens": max_tokens,
            "stats": {"error": type(exc).__name__, "fail_open": True},
        }


__all__ = [
    "ContextPackerConfig",
    "load_context_packer_config",
    "pack_messages_for_budget",
]
