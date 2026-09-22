"""Join streamed continuations without duplicated or malformed boundaries."""
import re

from . import image_context as imgctx


def _continuation_separator(partial_text: str, continuation_text: str) -> str:
    if not partial_text:
        return ""
    if partial_text.endswith(("\n", " ")):
        return ""
    if imgctx.has_unterminated_markdown_image(partial_text):
        return ""
    if continuation_text.startswith((")", "]", "/", ".", ",", ":", ";", "!", "?")):
        return ""
    tail = partial_text.rstrip()
    if tail.endswith(("&", "*", "(", "[", "{", "-", "—")):
        return ""
    if tail.endswith("**") or tail.count("**") % 2 == 1:
        return ""
    return "\n"


def _largest_suffix_prefix_overlap(left: str, right: str, min_chars: int = 1) -> int:
    max_size = min(len(left), len(right))
    for size in range(max_size, max(min_chars, 1) - 1, -1):
        if left.endswith(right[:size]):
            return size
    return 0


def _largest_common_prefix(left: str, right: str, min_chars: int = 1) -> int:
    max_size = min(len(left), len(right))
    for size in range(max_size, max(min_chars, 1) - 1, -1):
        if left[:size] == right[:size]:
            return size
    return 0


def _suffix_prefix_overlap_threshold(
    partial: str, continuation: str, overlap: int
) -> int:
    if overlap <= 0:
        return 0
    if overlap <= 12:
        return 3
    return max(3, min(96, min(len(partial), len(continuation)) // 8))


def _repair_continuation_seam(partial_text: str, merged_text: str) -> str:
    """Fix glued tokens and doubled percent signs at the partial/continuation boundary."""
    if not partial_text or not merged_text or len(partial_text) >= len(merged_text):
        return merged_text
    window_start = max(0, len(partial_text) - 48)
    window_end = min(len(merged_text), len(partial_text) + 48)
    window = merged_text[window_start:window_end]
    repaired = re.sub(r"(\d+(?:\.\d+)?%)(?:\1)+", r"\1", window)
    if repaired == window:
        return merged_text
    return merged_text[:window_start] + repaired + merged_text[window_end:]


def _trim_duplicate_continuation_prefix(
    partial_text: str, continuation_text: str
) -> str:
    partial = partial_text or ""
    continuation = continuation_text or ""
    if (not partial) or (not continuation):
        return continuation

    cont = continuation.lstrip("\r\n")
    partial_stripped = partial.lstrip("\r\n")

    if partial_stripped.startswith(cont):
        return ""

    if cont.startswith(partial):
        cont = cont[len(partial) :]
    else:
        prefix_overlap = _largest_common_prefix(partial_stripped, cont, min_chars=1)
        if prefix_overlap >= 24:
            cont = cont[prefix_overlap:]

    suffix_overlap = _largest_suffix_prefix_overlap(partial, cont, min_chars=1)
    threshold = _suffix_prefix_overlap_threshold(partial, cont, suffix_overlap)
    if suffix_overlap >= threshold:
        cont = cont[suffix_overlap:]

    return cont.lstrip("\r\n")
