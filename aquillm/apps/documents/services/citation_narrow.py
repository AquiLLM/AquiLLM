"""
Narrow a citation: ask the project's configured LLM to extract the verbatim
quote from a chunk that most directly supports a claim in an assistant
message. The result is fed to the React PDF-citation modal's fuzzy locator
to tighten the in-PDF highlight from "whole chunk" to "the relevant span".

Same one-shot pattern as conversation auto-titling
(apps/chat/models/conversation.py:set_name) — reuse `apps.llm_interface`.
"""
from __future__ import annotations

from typing import Optional

import structlog
from asgiref.sync import async_to_sync
from django.apps import apps

logger = structlog.stdlib.get_logger(__name__)


SYSTEM_PROMPT = (
    "You help highlight the most relevant text inside a cited source.\n"
    "Given an ASSISTANT MESSAGE that cites a SOURCE chunk, output the\n"
    "verbatim text from the SOURCE that most directly supports the claim.\n"
    "\n"
    "Rules:\n"
    "- Output ONLY the quote text. No quotation marks around it, no\n"
    "  commentary, no headers — just the raw text.\n"
    "- Quote verbatim — character-for-character — what appears in SOURCE.\n"
    "  Do not paraphrase, do not fix typos, do not normalise whitespace.\n"
    "- Aim for 1-3 sentences (~30-300 characters).\n"
    "- If no single passage is more relevant than the rest, or the chunk\n"
    "  as a whole is the support, output an empty response.\n"
)


def _build_user_prompt(message_content: str, chunk_content: str) -> str:
    return (
        "ASSISTANT MESSAGE:\n"
        f"{message_content}\n"
        "\n"
        "SOURCE:\n"
        f"{chunk_content}\n"
    )


def narrow_citation(message_content: str, chunk_content: str) -> Optional[str]:
    """Return a verbatim quote from `chunk_content` that supports the
    assistant claim, or None on any failure / empty output.

    Caller is expected to fall back to highlighting the full chunk when
    None is returned.
    """
    if not message_content.strip() or not chunk_content.strip():
        return None

    llm_interface = apps.get_app_config('aquillm').llm_interface
    llm_args = {
        **llm_interface.base_args,
        'max_tokens': 600,
        'thinking_budget': 0,
        'system': SYSTEM_PROMPT,
        'messages': [{'role': 'user', 'content': _build_user_prompt(message_content, chunk_content)}],
    }

    @async_to_sync
    async def call() -> str:
        response = await llm_interface.get_message(**llm_args)
        return response.text or ""

    try:
        quote = call().strip()
    except Exception as exc:
        logger.warning("Citation narrow LLM call failed: %s", exc)
        return None

    if not quote:
        return None
    # Strip a single layer of surrounding quotes if the model added them.
    if len(quote) >= 2 and quote[0] == quote[-1] and quote[0] in ('"', "'"):
        quote = quote[1:-1].strip()
    return quote or None
