"""Whole-document tool response shaping."""


def image_document_tool_payload(*, full_text: str, title: str, display_url: str) -> dict:
    """Result value for an image-backed document when returning full document text."""
    return {
        "text": full_text,
        "type": "image_document",
        "image_url": display_url,
    }


def image_document_instruction(*, title: str, display_url: str) -> str:
    return (
        f"This is an image document. The image is shown above. "
        f"When showing this to the user, include the image using markdown: "
        f"![{title}]({display_url})"
    )


__all__ = ["image_document_instruction", "image_document_tool_payload"]
