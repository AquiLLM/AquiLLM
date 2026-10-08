"""Conservative visual routing; the model still chooses relevant figures."""

import re

CONDITIONAL_IMAGE_INSTRUCTION = (
    "Image selection rules: Answer in text by default. Set include_images=true "
    "on document retrieval tools only when the user requests a visual or viewing "
    "a specific plot, diagram, image, or visual comparison materially helps answer "
    "the question. Caption/OCR text remains available without fetching images. "
    "Search for the specific visual evidence needed; do not fetch figures merely "
    "because a retrieved paper contains them. From returned candidates, embed only "
    "figures that directly support the explanation, with a meaningful caption and "
    "the exact returned image URL. Do not append a gallery of unused candidates, "
    "decorate a text answer, or use figures to fill an evidence gap. Honor requests "
    "for text only or no images. Do not reuse images from earlier turns unless the "
    "user refers to them; retrieve the relevant figure again when needed."
)

_TARGET = (
    r"(?:figures?|figs?\.?|images?|pictures?|visuals?|plots?|graphs?|charts?|"
    r"diagrams?|photos?)"
)
_NO_IMAGES = re.compile(
    rf"\b(?:no|without)\s+(?:any\s+)?{_TARGET}\b|\btext[- ]only\b|"
    r"\b(?:do not|don't|dont|never)\s+"
    r"(?:show|display|render|include|retrieve|fetch|attach|add|embed|use|send|"
    r"provide|want|need)\s+"
    rf"(?:\w+\s+){{0,3}}{_TARGET}\b",
    re.I,
)
_ACTION = re.compile(
    r"\b(show|shown|display|render|include|explain|find|get|pull|open|compare|"
    r"interpret|see|view|want|would\s+like)\b",
    re.I,
)
_TARGET_RE = re.compile(rf"\b{_TARGET}\b", re.I)
_FOLLOWUP = re.compile(
    rf"\b(?:show|display|render)\s+(?:me\s+)?(?:it|them|that|those)\b|"
    rf"\b(?:that|those|previous|earlier|same)\s+{_TARGET}\b",
    re.I,
)
_VISUAL_QUESTION = re.compile(
    r"\b(?:what\s+does|what\s+do|how\s+does|how\s+do)\b[^.?!]*\blook\s+like\b|"
    r"\b(?:compare|describe|explain)\b[^.?!]*\b(?:shapes?|morpholog(?:y|ies))\b",
    re.I,
)


def requests_visuals(text: str) -> bool:
    """Route likely visual questions to tools; never force an image into output."""
    text = (text or "").replace("\u2019", "'")
    if _NO_IMAGES.search(text):
        return False
    # These uses name reasoning/technical concepts, not document illustrations.
    text = re.sub(
        r"\bfigure\s+out\b|\bgraphs?\s+(?:rag|algorithms?|theory|databases?)\b",
        "",
        text,
        flags=re.I,
    )
    return bool(
        (_TARGET_RE.search(text) and _ACTION.search(text))
        or re.search(r"\b(?:figure|fig\.?|plot|diagram|chart)\s+\d+\b", text, re.I)
        or _FOLLOWUP.search(text)
        or _VISUAL_QUESTION.search(text)
    )


def refers_to_previous_visual(text: str) -> bool:
    return requests_visuals(text) and bool(_FOLLOWUP.search(text or ""))
