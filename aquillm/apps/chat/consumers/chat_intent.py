"""Local-tool and document intent predicates for append configuration."""

from apps.chat.services.rag_intent import classify_chat_message


def _looks_like_explicit_document_search_request(message_content: str) -> bool:
    """True when the user explicitly asks the assistant to retrieve from documents."""
    intent = classify_chat_message(message_content or "", selected_collection_ids=[])
    return (
        intent.requires_rag and not intent.requires_local_tools and not intent.is_retry
    )


def _looks_like_local_tool_request(message_content: str) -> bool:
    """True for app-local non-document tools such as FITS processing."""
    return classify_chat_message(
        message_content or "", selected_collection_ids=[]
    ).requires_local_tools


def _looks_like_retry_request(message_content: str) -> bool:
    """True when the user asks to rerun the previous failed/unsatisfying turn."""
    return classify_chat_message(
        message_content or "", selected_collection_ids=[]
    ).is_retry
