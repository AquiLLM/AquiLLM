"""Adjacent context tool, with selected-scope source-mode preflight."""

from django.contrib.auth.models import User

from apps.chat.consumers.utils import truncate_tool_text
from apps.documents.models import TextChunk
from apps.documents.services.source_loading import (
    bounded_source_database,
    source_mode_enabled,
)
from aquillm.llm import LLMTool, ToolResultDict, llm_tool
from lib.tools.search.context import format_adjacent_chunks_tool_result

from .bounded_document_tools import bounded_document_result
from .source_documents import (
    can_view_document,
    document_metadata,
    required_source_runtime,
)


def _central_chunk(chunk_id):
    if not source_mode_enabled():
        return TextChunk.objects.filter(id=chunk_id).first()
    runtime = required_source_runtime()
    alias = runtime.authorization.database_alias
    with bounded_source_database(runtime, alias):
        return (
            TextChunk.objects.using(alias)
            .filter(id=chunk_id)
            .only(
                "id",
                "doc_id",
                "chunk_number",
            )
            .first()
        )


def more_context_tool(user: User) -> LLMTool:
    @llm_tool(
        for_whom="assistant",
        required=["adjacent_chunks", "chunk_id"],
        param_descs={
            "chunk_id": "ID number of the chunk for which more context is desired",
            "adjacent_chunks": (
                "How many chunks on either side to return. Sta"
                "rt small and work up, if you think "
                "expanding the context will provide more usefu"
                "l info. Go no higher than 10."
            ),
        },
    )
    def more_context(chunk_id: int, adjacent_chunks: int) -> ToolResultDict:
        """
        Get adjacent text chunks on either side of a given chunk.
        Use this when a search returned something relevant, but it seemed like the
        information was cut off.
        """
        if adjacent_chunks < 1 or adjacent_chunks > 10:
            return {"exception": "Invalid value for adjacent_chunks!"}
        central_chunk = _central_chunk(chunk_id)
        if central_chunk is None:
            return {"exception": f"Text chunk {chunk_id} does not exist!"}
        doc = document_metadata(central_chunk.doc_id)
        if doc is None:
            return {"exception": f"Document for chunk {chunk_id} does not exist!"}
        if not can_view_document(user, doc):
            return {"exception": f"User cannot access document containing {chunk_id}!"}
        central_chunk_number = central_chunk.chunk_number
        bottom = central_chunk_number - adjacent_chunks
        top = central_chunk_number + adjacent_chunks
        if source_mode_enabled():
            query = (
                TextChunk.objects.using(
                    required_source_runtime().authorization.database_alias
                )
                .filter(
                    doc_id=central_chunk.doc_id, chunk_number__in=range(bottom, top + 1)
                )
                .order_by("chunk_number")
            )
            from .source_tool_revalidation import revalidate_source_tool_result

            return revalidate_source_tool_result(
                bounded_document_result(doc, query, adjacent=True),
                user=user,
            )
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
