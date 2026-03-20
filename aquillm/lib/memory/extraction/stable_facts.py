"""
Stable fact extraction from conversation turns.

Extracts durable memory facts (preferences, goals, background) using either
LLM-based extraction or heuristic fallback.
"""

import json
import logging
import re
from os import getenv
from typing import Optional

import requests

from ..config import MEM0_TIMEOUT_SECONDS
from ..mem0.client import _normalize_openai_base_url

logger = logging.getLogger(__name__)


def _extract_json_object(text: str) -> dict:
    """Extract a JSON object from text, handling markdown code blocks."""
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()
    if not cleaned:
        return {}
    try:
        payload = json.loads(cleaned)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return {}
        try:
            payload = json.loads(match.group(0))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}


def extract_stable_facts(user_content: str, assistant_content: str) -> list[str]:
    """Extract durable memory facts locally so Mem0 write can skip server-side infer."""
    base_url = _normalize_openai_base_url(
        getenv(
            "MEM0_LLM_BASE_URL",
            getenv("MEM0_VLLM_BASE_URL", getenv("MEM0_OLLAMA_BASE_URL", "http://vllm:8000/v1")),
        )
    ).rstrip("/")
    model = getenv("MEM0_LLM_MODEL", "qwen3.5:4b-q8_0")
    api_key = getenv("MEM0_LLM_API_KEY", getenv("VLLM_API_KEY", ""))
    prompt = (
        "Extract durable memory candidates from this turn. "
        "Include: (1) stable user-specific preferences/goals/background, "
        "(2) explicit user requests to remember information going forward, and "
        "(3) durable task/domain facts that the user wants retained (project context, tooling choices, "
        "technical facts the user explicitly says to remember). "
        'For explicit remember directives (e.g. "remember X"), include the substantive fact X directly. '
        'Return STRICT JSON only in the form {"facts":["..."]}. '
        'If none exist, return {"facts":[]}. '
        "Do not include temporary one-off requests unless they are explicit long-term remember directives.\n\n"
        f"User: {user_content}\nAssistant: {assistant_content}"
    )
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            'Return STRICT JSON only in the form {"facts":["..."]}. '
                            'If none exist, return {"facts":[]}.'
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "stream": False,
            },
            timeout=MEM0_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        raw_payload = response.json()
        payload = raw_payload if isinstance(raw_payload, dict) else {}
        content = ""
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0] if isinstance(choices[0], dict) else {}
            message = first_choice.get("message") if isinstance(first_choice, dict) else {}
            if isinstance(message, dict):
                value = message.get("content")
                if isinstance(value, str):
                    content = value
        obj = _extract_json_object(content)
        facts = obj.get("facts", [])
        out: list[str] = []
        if isinstance(facts, list):
            for item in facts:
                if isinstance(item, str):
                    item = item.strip()
                    if item:
                        out.append(item)
        return out
    except Exception as exc:
        logger.warning("Stable-fact extraction failed: %s", exc)
        return []


def heuristic_facts_from_turn(user_content: str, assistant_content: str) -> list[str]:
    """
    Deterministic fallback for explicit memory intents when model extraction returns nothing.
    """
    user_text = (user_content or "").strip()
    if not user_text:
        return []

    lowered = user_text.lower()
    facts: list[str] = []

    if any(token in lowered for token in ("remember", "going forward", "from now on")):
        facts.append(f"User asked to remember: {user_text}")

    pattern_map = [
        (r"\bi prefer\b.+", "preference"),
        (r"\bi like\b.+", "preference"),
        (r"\bcall me\b.+", "name"),
        (r"\bmy name is\b.+", "name"),
        (r"\bi work on\b.+", "domain"),
        (r"\bi am working on\b.+", "domain"),
        (r"\bour stack is\b.+", "domain"),
        (r"\bwe use\b.+", "domain"),
        (r"\bthe project is\b.+", "domain"),
    ]
    for pattern, _label in pattern_map:
        match = re.search(pattern, user_text, flags=re.IGNORECASE)
        if match:
            candidate = match.group(0).strip()
            if candidate:
                facts.append(candidate[0].upper() + candidate[1:] if len(candidate) > 1 else candidate.upper())

    if facts and "remember" in lowered and assistant_content:
        assistant_snippet = re.sub(r"\s+", " ", assistant_content).strip()[:220]
        if assistant_snippet:
            facts.append(f"Remembered context: {assistant_snippet}")

    return list(dict.fromkeys(facts))


def has_remember_intent(text: str) -> bool:
    """Check if text contains explicit remember intent."""
    lowered = (text or "").lower()
    return any(token in lowered for token in ("remember", "going forward", "from now on"))


def normalize_remember_fact(user_content: str) -> str:
    """
    Normalize explicit remember directives to a fact string while preserving user content.
    """
    text = (user_content or "").strip()
    if not text:
        return ""
    text = re.sub(
        r"^\s*(please\s+)?remember(\s+this)?(\s+going\s+forward)?\s*[:,-]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    return text or (user_content or "").strip()


__all__ = [
    'extract_stable_facts',
    'heuristic_facts_from_turn',
    'has_remember_intent',
    'normalize_remember_fact',
]
