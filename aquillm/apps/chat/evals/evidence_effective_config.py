"""Production-resolved arm treatments and secret-free comparison controls."""

from dataclasses import asdict
from os import getenv
from urllib.parse import urlsplit


def expected_treatment(mode):
    active = mode in ("preservation", "combined")
    expected = {
        "rerank_text_mode": "windowed" if active else "legacy",
        "evidence_text_mode": "source" if active else "legacy",
        "document_capacity_mode": "budgeted" if active else "legacy",
        "document_hard_cap": 0,
        "followup_evidence_enabled": active,
        "iterative_retrieval_enabled": active,
        "shadow_scoring": False,
        "error": None,
    }
    return {
        "mode": mode,
        "selection": "adaptive" if mode in ("selection", "combined") else "legacy",
        "preservation": expected,
    }


def resolved_treatment(mode):
    from apps.chat.services.rag_config import (
        evidence_selection_config,
        is_direct_rag_enabled,
    )
    from apps.chat.services.rag_preservation_config import preservation_config

    selector, preservation = evidence_selection_config(), preservation_config()
    expected = expected_treatment(mode)
    if (
        mode not in ("baseline", "selection", "preservation", "combined")
        or selector.error
        or preservation.error
        or not is_direct_rag_enabled()
        or selector.mode
        != ("adaptive" if mode in ("selection", "combined") else "legacy")
        or selector.shadow_scoring
        or asdict(preservation) != expected["preservation"]
    ):
        raise ValueError("requested arm differs from effective configuration")
    return expected


def additional_controls():
    from django.conf import settings

    from apps.chat.consumers import chat, chat_receive, utils
    from apps.chat.services import rag_config
    from apps.documents.services import chunk_rerank_config as rerank
    from apps.documents.services import rag_cache
    from lib.llm.providers.completion_policy_snapshot import completion_policy_snapshot
    from lib.llm.providers.openai_runtime_config import (
        context_limit,
        optional_float,
        request_limits,
        stream_enabled,
    )
    from lib.llm.providers.openai_tokens import context_reserve_tokens, env_int
    from lib.llm.providers.tool_budget import ToolBudgetConfig
    from lib.llm.turn_context import tool_timeout
    from lib.llm.utils import prompt_budget
    from lib.llm.utils.context_packer import load_context_packer_config

    endpoint = urlsplit(rerank.rerank_base_url())
    rerank_names = (
        "provider",
        "model",
        "model_revision",
        "vllm_model",
        "tokenizer",
        "tokenizer_revision",
        "code_revision",
        "timeout_seconds",
        "doc_char_limit",
        "pair_token_limit",
        "template_reserve_tokens",
        "score_concurrency",
    )
    return {
        "completion": completion_policy_snapshot(
            output_tokens=rag_config.synthesis_max_tokens()
        ),
        "tool_loop": {
            name: asdict(
                ToolBudgetConfig.from_env(max_func_calls=module.CHAT_MAX_FUNC_CALLS)
            )
            for name, module in (("connect", chat), ("receive", chat_receive))
        },
        "openai": {
            "request_limits": list(request_limits()),
            "stream": stream_enabled(),
            "context_limit": context_limit(),
            "reserves": context_reserve_tokens(context_limit()),
            "temperature": optional_float("OPENAI_TEMPERATURE"),
            "top_p": optional_float("OPENAI_TOP_P"),
            "system_role": getenv("OPENAI_SYSTEM_ROLE", "").strip().lower(),
            "thinking": (getenv("OPENAI_COMPAT_ENABLE_THINKING", "1") or "1")
            .strip()
            .lower()
            in ("1", "true", "yes", "on"),
            "compat_slack": env_int("OPENAI_COMPAT_PROMPT_SLACK_TOKENS", 256),
            "api_slack": env_int("OPENAI_API_PROMPT_SLACK_TOKENS", 384),
            "tool_min_completion": env_int("LLM_TOOL_MIN_COMPLETION_TOKENS", 128),
            "min_completion_overflow": env_int("LLM_MIN_COMPLETION_TOKENS", 384),
            "min_completion_timeout": env_int("LLM_MIN_COMPLETION_TOKENS", 256),
        },
        "context_packer": asdict(load_context_packer_config()),
        "selection_score_timeout_ms": rag_config.selection_scoring_timeout_ms(),
        "rerank": {
            **{key: getattr(rerank, "rerank_" + key)() for key in rerank_names},
            "endpoint": f"{endpoint.scheme}://{endpoint.hostname}:{endpoint.port}{endpoint.path}",
        },
        "chat": {
            key: getattr(utils, key)
            for key in (
                "CHAT_MAX_FUNC_CALLS",
                "CHAT_MAX_TOKENS",
                "MAX_IMAGES_PER_TOOL_RESULT",
                "LLM_IMAGE_MAX_DIMENSION",
                "LLM_IMAGE_MAX_BYTES",
            )
        },
        "outer_limits": {
            name: {
                key: getattr(module, key)
                for key in ("CHAT_MAX_FUNC_CALLS", "CHAT_MAX_TOKENS")
            }
            for name, module in (("connect", chat), ("receive", chat_receive))
        },
        "tools": {
            "timeout_seconds": tool_timeout(),
            "default_top_k": rag_config.tool_default_top_k(),
            "attach_selected": rag_config.attach_tools_when_collections_selected(),
            "max_figures": rag_config.max_figures_per_turn(),
            "query_rewrite": rag_config.query_rewrite_enabled(),
        },
        "cache": {
            "enabled": bool(getattr(settings, "RAG_CACHE_ENABLED", False)),
            **{
                key: getattr(rag_cache, key)()
                for key in (
                    "query_embed_ttl",
                    "doc_access_ttl",
                    "document_lookup_ttl",
                    "image_data_url_ttl",
                    "rerank_capability_ttl",
                )
            },
        },
        "prompt": {
            "context_packer": prompt_budget.context_packer_enabled(),
            "context_limit": prompt_budget.prompt_budget_context_limit(),
            "token_efficiency": prompt_budget.token_efficiency_enabled(),
            "slack": prompt_budget.prompt_budget_slack_tokens(),
            "max_tokens_cap": prompt_budget.prompt_budget_max_tokens_cap(),
        },
    }
