"""
Stable fact extraction from conversation turns.

Extracts durable memory facts (preferences, goals, background) using either
LLM-based extraction or heuristic fallback.
"""

import json
import re
from os import getenv

import structlog

import requests

from ..config import MEM0_TIMEOUT_SECONDS
from ..mem0.client import _normalize_openai_base_url
from ..types import MemoryIntent

INTENT_POLICY_VERSION = 'affirmative-source-v1'
_DIRECTIVE = re.compile(
    r'(?i)(?:^|(?<=[.!?;\n]))\s*(?:also\s+)?(?:(?:could|can|would|will)\s+you\s+)?'
    r'(?:please\s+)?remember(?:\s+this)?(?:\s+going\s+forward)?\s*[:,-]?\s*(?:that\s+)?'
)
_POLICY_DIRECTIVE = re.compile(
    r'(?i)\b(?:ignore|bypass|override|disable|skip|disregard|omit|never|without|do not|don\x27t)\b'
    r'.{0,100}\b(?:citation|citations|sources?|authorization|permissions?|policy|policies|system|safety)\b'
)


def is_substantive_fact(text: str) -> bool:
    """Memory content cannot amend source, citation or authorization policy."""
    return (isinstance(text, str) and 2 <= len(text.split()) and len(text) <= 2000
            and not _is_low_value_fact(text) and not _POLICY_DIRECTIVE.search(text))


def parse_memory_intent(text: str) -> MemoryIntent:
    """Accept a standalone imperative only; uncertainty never grants persistence."""
    text = text or ''
    accepted = []
    fenced = [(m.start(), m.end()) for m in re.finditer(r'```[\s\S]*?(?:```|$)|`[^`\n]*`', text)]
    quoted = [(m.start(), m.end()) for m in re.finditer(r'"[^"\n]*"|“[^”\n]*”|(?<!\w)\x27[^\x27\n]*\x27', text)]
    for match in _DIRECTIVE.finditer(text):
        start = match.end()
        stop = re.search(r'[.!?;\n](?:\s|$)|\s+(?:and|but)\s+(?:do not|don\x27t|never)\s+remember\b', text[start:], re.I)
        end = start + stop.start() + (1 if stop and text[start + stop.start()] in '.!?;' else 0) if stop else len(text)
        while end > start and text[end - 1].isspace():
            end -= 1
        payload = text[start:end]
        if any(a < end and b > match.start() for a, b in fenced + quoted):
            continue
        if any(c in payload for c in ['"', '“', '”', '`']) or re.search(r'(?i)\b(?:if|hypothetically|remember)\b', payload):
            continue
        if is_substantive_fact(payload):
            accepted.append(MemoryIntent('affirmative', payload, start, end))
    if len(accepted) == 1:
        return accepted[0]
    if len(accepted) > 1:
        return MemoryIntent('ambiguous')
    if not re.search(r'(?i)\bremember\b|going forward|from now on', text):
        return MemoryIntent('none')
    if fenced or quoted:
        return MemoryIntent('quoted')
    if re.search(r'(?i)\bif\b|hypothetically|would happen', text):
        return MemoryIntent('hypothetical')
    if re.search(r'(?i)\b(?:not|never|don\x27t|cannot|can\x27t)\s+remember', text):
        return MemoryIntent('negated')
    return MemoryIntent('ambiguous')


def validate_fact_candidates(user_content: str, candidates: list[str]) -> list[str]:
    """A model string is a proposal, never proof of provenance or consent."""
    intent = parse_memory_intent(user_content)
    if intent.kind != 'affirmative':
        return []
    source = _normalize_fact_text(intent.payload).casefold()
    return [fact for fact in _clean_fact_candidates(candidates)
            if is_substantive_fact(fact) and _normalize_fact_text(fact).casefold().rstrip('.!?') in source]

logger = structlog.stdlib.get_logger(__name__)
_REMEMBER_PREFIX_RE = re.compile(
    r"^\s*(please\s+)?remember(\s+this)?(\s+going\s+forward)?\s*[:,-]?\s*",
    flags=re.IGNORECASE,
)
_REMEMBER_THAT_RE = re.compile(r"^\s*that\s+", flags=re.IGNORECASE)
_LOW_VALUE_FACT_PATTERNS = (
    re.compile(r"^user asked to remember:", flags=re.IGNORECASE),
    re.compile(r"^remembered context:", flags=re.IGNORECASE),
    re.compile(r"\byou should remember this\b", flags=re.IGNORECASE),
    re.compile(r"\bremember this\b", flags=re.IGNORECASE),
    re.compile(r"\bi(?:'| a)?ll remember that\b", flags=re.IGNORECASE),
    re.compile(r"\bkeep that in mind\b", flags=re.IGNORECASE),
)


def _normalize_fact_text(text: str) -> str:
    """Normalize a candidate fact into a compact fact-first sentence."""
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    normalized = _REMEMBER_PREFIX_RE.sub("", normalized).strip()
    normalized = _REMEMBER_THAT_RE.sub("", normalized).strip()
    normalized = normalized.strip(" \t\r\n\"'")
    return normalized


def _is_low_value_fact(text: str) -> bool:
    """Reject vague or reflexive memory candidates."""
    normalized = _normalize_fact_text(text)
    lowered = normalized.lower().strip(" .!?,:;")
    if not lowered:
        return True
    if lowered in {"this", "that", "it", "remember this", "remember that"}:
        return True
    return any(pattern.search(normalized) for pattern in _LOW_VALUE_FACT_PATTERNS)


def _clean_fact_candidates(facts: list[str]) -> list[str]:
    """Normalize and deduplicate candidate facts while dropping obvious noise."""
    out: list[str] = []
    for fact in facts:
        normalized = _normalize_fact_text(fact)
        if _is_low_value_fact(normalized):
            continue
        if normalized and normalized not in out:
            out.append(normalized)
    return out


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
    intent = parse_memory_intent(user_content)
    if intent.kind != 'affirmative':
        return []
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
        f"User-authored persistence payload: {intent.payload}"
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
                "max_tokens": 512,
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
        return validate_fact_candidates(user_content, out)
    except Exception as exc:
        logger.warning(
            "obs.memory.fact_extraction_error",
            error_type=type(exc).__name__,
        )
        return []


def heuristic_facts_from_turn(user_content: str, assistant_content: str) -> list[str]:
    """
    Deterministic fallback for explicit memory intents when model extraction returns nothing.
    """
    user_text = (user_content or "").strip()
    if not user_text:
        return []

    intent = parse_memory_intent(user_text)
    if intent.kind not in {'affirmative', 'none'}:
        return []
    facts: list[str] = []

    if intent.kind == 'affirmative':
        return [intent.payload]

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
        match = re.match(pattern, user_text, flags=re.IGNORECASE)
        if match:
            candidate = match.group(0).strip()
            if candidate:
                facts.append(candidate[0].upper() + candidate[1:] if len(candidate) > 1 else candidate.upper())

    return _clean_fact_candidates(facts)


def clean_stable_facts(facts: list[str]) -> list[str]:
    """Normalize and filter externally supplied stable-fact candidates."""
    return _clean_fact_candidates(facts)


def has_remember_intent(text: str) -> bool:
    """Check if text contains explicit remember intent."""
    return parse_memory_intent(text).kind == 'affirmative'


def normalize_remember_fact(user_content: str) -> str:
    """
    Normalize explicit remember directives to a fact string while preserving user content.
    """
    return parse_memory_intent(user_content).payload or ''


__all__ = [
    'parse_memory_intent', 'validate_fact_candidates', 'is_substantive_fact',
    'clean_stable_facts',
    'extract_stable_facts',
    'heuristic_facts_from_turn',
    'has_remember_intent',
    'normalize_remember_fact',
]
