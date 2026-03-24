"""Document- and search-related LLM tools (Django-bound).

Collection-backed tools resolve visible documents via
``Collection.get_user_accessible_documents`` (optional RAG doc-access cache when
``RAG_CACHE_ENABLED``).
"""
from __future__ import annotations

from asgiref.sync import async_to_sync
from django.contrib.auth.models import User

from aquillm.llm import LLMTool, ToolResultDict, llm_tool
from apps.chat.consumers.utils import truncate_tool_text
from apps.chat.refs import ChatRef, CollectionsRef
from apps.collections.models import Collection
from apps.documents.models import Document, DocumentChild, TextChunk
from lib.tools.documents.ids import resolve_doc_id_with_candidates
from lib.tools.documents.list_ids import titles_to_document_ids
from lib.tools.documents.whole_document import image_document_instruction, image_document_tool_payload
from lib.tools.search.context import format_adjacent_chunks_tool_result
from lib.tools.search.vector_search import pack_chunk_search_results

_NO_DOCS_EXCEPTION = {
    "exception": (
        "No documents to search! Either no collections were selected, or the selected "
        "collections are empty."
    )
}


def _accessible_document_ids(user: User, col_ref: CollectionsRef) -> list:
    docs = Collection.get_user_accessible_documents(
        user, Collection.objects.filter(id__in=col_ref.collections)
    )
    return [d.id for d in docs]


def _resolve_doc_uuid(doc_id: str, user: User, col_ref: CollectionsRef):
    return resolve_doc_id_with_candidates(doc_id, _accessible_document_ids(user, col_ref))


def vector_search_tool(user: User, col_ref: CollectionsRef) -> LLMTool:
    @llm_tool(
        param_descs={
            "search_string": (
                "The string to search by. Often it helps to phrase it as a question."
            ),
            "top_k": (
                "The number of results to return. Start with 5 for simple questions, 8-10 for broad "
                "or multi-part questions. Increase if the desired information is not found. "
                "Go no higher than 15."
            ),
        },
        required=["search_string", "top_k"],
        for_whom="assistant",
    )
    def vector_search(search_string: str, top_k: int) -> ToolResultDict:
        """
        Uses a combination of vector search, trigram search and reranking to search the documents
        available to the user. Prefer this tool when the question may span many documents; it does
        not require document UUIDs.
        Returns text chunks and image chunks. For image chunks, both the image and its OCR-extracted
        text are provided.
        When returning results to the user that include images, use markdown image syntax:
        ![description](image_url)
        """
        if top_k < 1 or top_k > 15:
            return {"exception": f"top_k must be between 1 and 15, got {top_k}"}
        if not search_string.strip():
            return {"exception": "search_string must not be empty"}
        docs = Collection.get_user_accessible_documents(
            user, Collection.objects.filter(id__in=col_ref.collections)
        )
        if not docs:
            return _NO_DOCS_EXCEPTION
        _, _, results = TextChunk.text_chunk_search(search_string, top_k, docs)
        titles_by_doc_id = {doc.id: doc.title for doc in docs}
        docs_by_doc_id = {doc.id: doc for doc in docs}

        return pack_chunk_search_results(
            results,
            titles_by_doc_id=titles_by_doc_id,
            docs_by_doc_id=docs_by_doc_id,
            truncate=truncate_tool_text,
            image_modality=TextChunk.Modality.IMAGE,
        )

    return vector_search


def document_list_ids_tool(user: User, col_ref: CollectionsRef) -> LLMTool:
    @llm_tool(
        for_whom="assistant",
        required=[],
        param_descs={},
    )
    def document_ids() -> ToolResultDict:
        """
        Get the names and IDs of all documents in the selected collections. When a user asks to see
        a document in full, or to search a single document, use this to get its ID. Copy UUIDs in full;
        they are easy to truncate by mistake.
        """
        docs = Collection.get_user_accessible_documents(
            user, Collection.objects.filter(id__in=col_ref.collections)
        )
        if not docs:
            return _NO_DOCS_EXCEPTION
        return {"result": titles_to_document_ids(docs)}

    return document_ids


def whole_document_tool(user: User, chat_ref: ChatRef, col_ref: CollectionsRef) -> LLMTool:
    @llm_tool(
        for_whom="assistant",
        required=["doc_id"],
        param_descs={"doc_id": "UUID (as as string) of the document to return in full"},
    )
    def whole_document(doc_id: str) -> ToolResultDict:
        """
        Get the full text of a document. Use when a user asks you to get a full document.
        For image documents, this includes both the extracted text and the image itself.
        When returning an image to the user, use markdown: ![description](image_url)
        """
        doc_uuid, error_msg = _resolve_doc_uuid(doc_id, user, col_ref)
        if doc_uuid is None:
            return {"exception": error_msg}
        doc: DocumentChild | None = Document.get_by_id(doc_uuid)
        if doc is None:
            return {"exception": f"Document {doc_id} does not exist!"}
        if not doc.collection.user_can_view(user):
            return {"exception": f"User cannot access document {doc_id}!"}
        token_count = async_to_sync(chat_ref.chat.llm_if.token_count)(chat_ref.chat.convo, doc.full_text)
        if token_count > 150000:
            return {"exception": f"Document {doc_id} is too large to open in this chat."}

        ret: ToolResultDict = {"result": doc.full_text}

        image_file = getattr(doc, "image_file", None)
        if image_file:
            display_url = f"/aquillm/document_image/{doc.id}/"
            ret["result"] = image_document_tool_payload(
                full_text=doc.full_text, title=doc.title, display_url=display_url
            )
            ret["_image_instruction"] = image_document_instruction(title=doc.title, display_url=display_url)

        return ret

    return whole_document


def search_single_document_tool(user: User, col_ref: CollectionsRef) -> LLMTool:
    @llm_tool(
        for_whom="assistant",
        required=["doc_id", "search_string", "top_k"],
        param_descs={
            "doc_id": "UUID (as a string) of the document to search.",
            "search_string": "String to search the contents of the document by.",
            "top_k": "Number of search results to return.",
        },
    )
    def search_single_document(doc_id: str, search_string: str, top_k: int) -> ToolResultDict:
        """
        Use vector search to search the text of a single document. If the user may mean many
        documents, prefer vector_search instead (no doc_id required).
        Returns text chunks and image chunks. For image chunks, both the image and its
        OCR-extracted text are provided.
        When returning results to the user that include images, use markdown image syntax:
        ![description](image_url)
        """
        if top_k < 1 or top_k > 15:
            return {"exception": f"top_k must be between 1 and 15, got {top_k}"}
        if not search_string.strip():
            return {"exception": "search_string must not be empty"}
        doc_uuid, error_msg = _resolve_doc_uuid(doc_id, user, col_ref)
        if doc_uuid is None:
            return {"exception": error_msg}
        doc = Document.get_by_id(doc_uuid)
        if doc is None:
            return {"exception": f"Document {doc_id} does not exist!"}
        if not doc.collection.user_can_view(user):
            return {"exception": f"User cannot access document {doc_id}!"}
        _, _, results = TextChunk.text_chunk_search(search_string, top_k, [doc])

        titles_by_doc_id = {doc.id: doc.title}
        docs_by_doc_id = {doc.id: doc}
        return pack_chunk_search_results(
            results,
            titles_by_doc_id=titles_by_doc_id,
            docs_by_doc_id=docs_by_doc_id,
            truncate=truncate_tool_text,
            image_modality=TextChunk.Modality.IMAGE,
        )

    return search_single_document


def more_context_tool(user: User) -> LLMTool:
    @llm_tool(
        for_whom="assistant",
        required=["adjacent_chunks", "chunk_id"],
        param_descs={
            "chunk_id": "ID number of the chunk for which more context is desired",
            "adjacent_chunks": (
                "How many chunks on either side to return. Start small and work up, if you think "
                "expanding the context will provide more useful info. Go no higher than 10."
            ),
        },
    )
    def more_context(chunk_id: int, adjacent_chunks: int) -> ToolResultDict:
        """
        Get adjacent text chunks on either side of a given chunk.
        Use this when a search returned something relevant, but it seemed like the information was cut off.
        """
        if adjacent_chunks < 1 or adjacent_chunks > 10:
            return {"exception": "Invalid value for adjacent_chunks!"}
        central_chunk = TextChunk.objects.filter(id=chunk_id).first()
        if central_chunk is None:
            return {"exception": f"Text chunk {chunk_id} does not exist!"}
        doc = Document.get_by_id(central_chunk.doc_id)
        if doc is None:
            return {"exception": f"Document for chunk {chunk_id} does not exist!"}
        if not doc.collection.user_can_view(user):
            return {"exception": f"User cannot access document containing {chunk_id}!"}
        central_chunk_number = central_chunk.chunk_number
        bottom = central_chunk_number - adjacent_chunks
        top = central_chunk_number + adjacent_chunks
        window = list(
            TextChunk.objects.filter(
                doc_id=central_chunk.doc_id, chunk_number__in=range(bottom, top + 1)
            )
            .order_by("chunk_number")
            .only("chunk_number", "content")
        )
        if not window:
            return {"exception": f"No nearby chunks found for chunk {chunk_id}."}
        return format_adjacent_chunks_tool_result(window, truncate=truncate_tool_text)

    return more_context


__all__ = [
    "document_list_ids_tool",
    "more_context_tool",
    "search_single_document_tool",
    "vector_search_tool",
    "whole_document_tool",
]
