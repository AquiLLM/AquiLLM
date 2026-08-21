"""Permission-filtered figure payload helpers for document tools."""

from __future__ import annotations

from django.contrib.auth.models import User

from apps.documents.models import DocumentChild, DocumentFigure


def format_related_figure_payloads(
    figures, *, user: User, max_figures: int = 3
) -> list[dict]:
    payloads: list[dict] = []
    for figure in figures:
        collection = getattr(figure, "collection", None)
        can_view = (
            getattr(collection, "user_can_view", None)
            if collection is not None
            else None
        )
        if callable(can_view) and not can_view(user):
            continue
        image_file = getattr(figure, "image_file", None)
        if not getattr(image_file, "name", ""):
            continue
        caption = (
            str(getattr(figure, "extracted_caption", "") or "").strip()
            or str(getattr(figure, "full_text", "") or "").strip()
            or str(getattr(figure, "title", "") or "Figure").strip()
        )
        payloads.append(
            {
                "type": "image",
                "title": str(getattr(figure, "title", "") or "Figure"),
                "text": caption[:500],
                "image_url": f"/aquillm/document_image/{figure.id}/",
                "figure_index": int(getattr(figure, "figure_index", len(payloads)) or 0)
                + 1,
            }
        )
        if len(payloads) >= max_figures:
            break
    return payloads


def related_figure_payloads(
    doc: DocumentChild, *, user: User, max_figures: int = 3
) -> list[dict]:
    if isinstance(doc, DocumentFigure):
        return []
    try:
        from django.contrib.contenttypes.models import ContentType

        content_type = ContentType.objects.get_for_model(doc, for_concrete_model=False)
        figures = DocumentFigure.objects.filter(
            parent_content_type=content_type,
            parent_object_id=doc.id,
        ).order_by("figure_index", "title")[: max_figures * 3]
    except Exception:
        return []
    return format_related_figure_payloads(figures, user=user, max_figures=max_figures)


__all__ = ["format_related_figure_payloads", "related_figure_payloads"]
