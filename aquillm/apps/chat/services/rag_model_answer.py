"""Bound model presentation without assigning semantic or media authority."""

import re
from html import unescape
from urllib.parse import unquote, urlsplit

from lib.llm.providers.rag_citations import find_invalid_citations

_RAW_HTML = re.compile(r"<(?:[!?]|/?[A-Za-z][A-Za-z0-9:-]*(?=[\s/>])[^<>]*>)")
_UNSAFE_PREFIX = re.compile(
    r"</?(?:script|style|iframe|frame|frameset|object|embed|img|image|picture|"
    r"video|audio|source|track|svg|math|link|meta|base|form|input|textarea|"
    r"select|button|template|plaintext|xmp|noscript|noembed|noframes)"
    r"(?=[\s/>]|$)",
    re.I,
)
_HTML_PREFIX = re.compile(r"</?[A-Za-z][A-Za-z0-9:-]*(?=[\s/>]|$)")
_ATTRIBUTE_PREFIX = re.compile(
    r"</?[A-Za-z][A-Za-z0-9:-]*[ \t]+[A-Za-z_:][A-Za-z0-9_.:-]*(?=[\s=/>]|$)"
)
_MATH_OR_CODE = re.compile(
    r"(?m:^ {0,3}\$\$[ \t]*\r?\n"
    r"(?:(?! {0,3}\${2,}[ \t]*\r?$)[^\r\n]*\r?\n)*"
    r" {0,3}\${2,}[ \t]*(?=\r?$))|"
    r"(?<!\\)\$\$[^\r\n]*?(?<!\\)\$\$|"
    r"(?<!\\)\$[^\r\n$]+(?<!\\)\$|(`+)[^\r\n]*?\1"
)
_REFERENCE = re.compile(
    r"(?m)^[ \t]*(?:>[ \t]*|[-+*][ \t]+|[0-9]{1,9}[.)][ \t]+)*"
    r"(\[[^\[\]\r\n]+\]):"
)
_IMAGE = re.compile(r"!\[[^\[\]\r\n]*\]\(([^()\s<>\"']+)\)")
_LINK = re.compile(r"(?<!!)\[[^\[\]\r\n]*\]\(([^()\s]+)\)")
_INTERNAL_ROUTE = re.compile(
    r"(?:^|/)(?:document(?:_image)?|aquillm/pdf|api/(?:citation[^/\s]*|chunks))/",
    re.I,
)


def _decoded_url(value):
    # Decode for detection only; never turn an authored URL into media authority.
    return unquote(unescape(value)).replace("\\", "/")


def _escape_ambiguous_prefixes(text):
    # Linkification adds HTML after this boundary. An unfinished literal prefix
    # must not consume that new markup. Closed math/code preserve their syntax;
    # the unsafe-tag check still applies globally before this display escaping.
    parts, start = [], 0
    for match in _MATH_OR_CODE.finditer(text):
        parts.append(
            _HTML_PREFIX.sub(lambda m: "&lt;" + m[0][1:], text[start : match.start()])
        )
        parts.append(match[0])
        start = match.end()
    parts.append(_HTML_PREFIX.sub(lambda m: "&lt;" + m[0][1:], text[start:]))
    return "".join(parts)


def prepare_model_answer(packet, text):
    """Keep prose and exact citations; normalize only captured inline images.

    Unsupported presentation raises a content-free reason. The caller still
    performs current authorization and final citation validation on the result.
    Reference definitions and raw HTML are deliberately outside this grammar.
    """
    if not isinstance(text, str) or re.search(r"[\x00-\x08\x0b-\x1f]", text):
        raise ValueError("invalid_presentation")
    allowed_citations = set(packet.citation_tokens)
    if (
        _RAW_HTML.search(text)
        or _UNSAFE_PREFIX.search(text)
        or _ATTRIBUTE_PREFIX.search(text)
        # Exact authorized citations are linkified before Markdown definitions.
        or any(m[1] not in allowed_citations for m in _REFERENCE.finditer(text))
    ):
        raise ValueError("invalid_presentation")
    invalid_citations = find_invalid_citations(text, allowed_citations)
    text = _escape_ambiguous_prefixes(text)

    images = tuple(_IMAGE.finditer(text))
    prose = _IMAGE.sub("", text)
    if "![" in prose:
        raise ValueError("invalid_media")
    # GFM also links plain absolute URLs. Reject authored document routes even
    # outside Markdown links; only server citation linkification creates them.
    if _INTERNAL_ROUTE.search(_decoded_url(prose)):
        raise ValueError("invalid_presentation")
    if re.search(r"\]\s*\(", _LINK.sub("", prose)):
        raise ValueError("invalid_presentation")
    for link in _LINK.finditer(prose):
        try:
            url = urlsplit(_decoded_url(link[1]))
        except ValueError as exc:
            raise ValueError("invalid_presentation") from exc
        if url.scheme not in {"http", "https", "mailto"}:
            raise ValueError("invalid_presentation")

    if images:
        from .rag_immutable_images import selected_image_lines

        try:
            lines = selected_image_lines(packet, {image[1] for image in images})
        except ValueError as exc:
            raise ValueError("invalid_media") from exc
        text = "\n\n".join(part for part in (prose.strip(), *lines) if part)
    # Citation fallback must not hide a rejected media or presentation request.
    if invalid_citations:
        raise ValueError("invalid_citation")
    return text
