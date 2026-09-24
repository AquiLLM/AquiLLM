"""Map actual provider-shaped tool rows back to frozen Unicode source spans."""

import json
import re

from .evidence_quality_eval import text_digest

_CITATION = re.compile(r"\[doc:([^\s\]]+)\s+chunk:(\d+)\]")


def strings(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from strings(item)


def rows(value):
    """ToolMessage prefixes and Gemini content objects retain embedded JSON."""
    if isinstance(value, dict):
        if ("chunk_id" in value or "i" in value) and (
            "doc_id" in value or "d" in value
        ):
            yield value
        for item in value.values():
            yield from rows(item)
    elif isinstance(value, list):
        for item in value:
            yield from rows(item)


def packet_rows(payload):
    decoder = json.JSONDecoder()
    seen = set()
    for text in strings(payload):
        for match in re.finditer(r"\{", text):
            try:
                value, _ = decoder.raw_decode(text[match.start() :])
            except ValueError:
                continue
            for row in rows(value):
                key = json.dumps(row, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    yield row


def map_payload(payload, sources, hints=None):
    delivered, unknown = [], []
    for row in packet_rows(payload):
        key = str(row.get("doc_id", row.get("d"))), row.get("chunk_id", row.get("i"))
        source = sources.get(key)
        text = row.get("text", row.get("x", ""))
        citation = row.get("citation", row.get("ref"))
        if not source or not text or citation != f"[doc:{key[0]} chunk:{key[1]}]":
            unknown.append(key)
            continue
        offsets = (hints or {}).get(key)
        if offsets is None:
            marker = "\n...[truncated for context window]..."
            if text.endswith(marker):
                text = text.removesuffix(marker)
            # Exact legacy clipping can be located; ellipses/rewrites or repeated
            # occurrences are unknown, never inferred from candidate membership.
            matches = [m.start() for m in re.finditer(re.escape(text), source["text"])]
            offsets = (
                [(matches[0], matches[0] + len(text))] if len(matches) == 1 else []
            )
        if not offsets or any(source["text"][a:b] not in text for a, b in offsets):
            unknown.append(key)
            continue
        for start, end in offsets:
            delivered.append(
                {
                    "source_id": source["source_id"],
                    "revision": source["revision"],
                    "fingerprint": text_digest(source["text"]),
                    "start": start,
                    "end": end,
                    "text": source["text"][start:end],
                }
            )
    return delivered, unknown


def parse_citations(answer, sources):
    citations = []
    for document, chunk in _CITATION.findall(answer):
        source = sources.get((document, int(chunk)))
        citations.append(
            {
                "source_id": source["source_id"]
                if source
                else f"unmapped:{document}:{chunk}",
                "revision": source["revision"] if source else "unknown",
            }
        )
    return citations
