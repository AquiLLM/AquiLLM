"""Whole-document tool response shaping."""

INCLUDE_IMAGES_DESCRIPTION = (
    "Optional; defaults to false. Set true only when the user requests a visual "
    "or a figure, plot, or diagram materially helps answer the current question. "
    "Text and image OCR remain searchable when false."
)


def image_document_tool_payload(
    *, full_text: str, title: str, display_url: str
) -> dict:
    """Result value for an image-backed document when returning full document text."""
    return {
        "text": full_text,
        "type": "image_document",
        "image_url": display_url,
    }


def image_document_instruction(*, title: str, display_url: str) -> str:
    return (
        "This image is optional visual evidence. Include it only if it directly "
        "supports the current answer or the user requested it; otherwise answer "
        "with text. When selected, explain its relevance and use the exact URL: "
        f"![{title}]({display_url})"
    )


__all__ = [
    "INCLUDE_IMAGES_DESCRIPTION",
    "image_document_instruction",
    "image_document_tool_payload",
]
