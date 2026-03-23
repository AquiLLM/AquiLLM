"""Document listing helper for tool results."""


def titles_to_document_ids(docs) -> dict[str, str]:
    """Map document title -> id string for tool payloads."""
    return {doc.title: str(doc.id) for doc in docs}


__all__ = ["titles_to_document_ids"]
