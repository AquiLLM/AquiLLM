"""Attach whole-document media only through the selected captured chunk."""
from apps.documents.services.representation_media_acquisition import chunk_media_url
from lib.tools.documents.whole_document import image_document_instruction, image_document_tool_payload
from lib.tools.search.vector_search import IMAGE_MARKDOWN_INSTRUCTION


def attach_whole_document_media(result, document, chunks, *, user, related_figures):
    text = result["result"]
    if getattr(document, "image_file", None):
        selected = next(iter(chunks), None)
        display_url = chunk_media_url(selected) if selected is not None else None
        if display_url:
            result["result"] = image_document_tool_payload(
                full_text=text, title=document.title, display_url=display_url)
            result["_image_instruction"] = image_document_instruction(
                title=document.title, display_url=display_url)
    else:
        from lib.llm.turn_context import current_turn
        if current_turn() is not None:
            from .source_figure_payloads import bounded_figure_payloads
            from .bounded_document_tools import _mark_figure_omission
            from apps.documents.services.representation_media_acquisition import media_runtime, _check_runtime
            runtime = media_runtime()
            _check_runtime(runtime)
            _, _, omitted = bounded_figure_payloads(document, user=user, runtime=runtime)
            if omitted:
                _mark_figure_omission(result)
            return
        figures = related_figures(document, user=user)
        if figures:
            result["result"] = {"type": "document_with_figures", "text": text, "figures": figures}
            result["_image_instruction"] = IMAGE_MARKDOWN_INSTRUCTION
