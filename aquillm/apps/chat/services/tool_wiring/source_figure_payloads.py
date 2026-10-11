"""Whole-parent tools omit figures absent from their selected chunk packet."""
from django.contrib.contenttypes.models import ContentType

from apps.collections.services.retrieval_authorization import revalidate_retrieval_authorization_context
from apps.documents.models import DocumentFigure
from apps.documents.services.source_loading import bounded_source_database
from .source_documents import required_source_runtime


def bounded_figure_payloads(doc, *, user, max_figures=3, runtime=None):
    """Detect omitted auxiliaries without reading their captions or media.

    A whole-parent result contains only the parent's citation chunks. Even an
    authorized figure in the original selected scope is not in that packet.
    Acquiring its bytes here would spend budget on evidence that cannot pass
    the final exact-chunk media binding. Standalone figure tools use that binding.
    """
    runtime = runtime or required_source_runtime()
    alias = runtime.authorization.database_alias
    if isinstance(doc, DocumentFigure):
        return [], [], False
    with runtime.lock, bounded_source_database(runtime, alias):
        from apps.documents.services.representation_tool_scope import tool_scope
        frozen = tool_scope(runtime, using=alias)
        scope = revalidate_retrieval_authorization_context(context=runtime.authorization)
        if doc.id not in scope.document_ids:
            return [], [], True
        content_type = ContentType.objects.db_manager(alias).get_for_model(doc, for_concrete_model=False)
        omitted = DocumentFigure.objects.using(alias).filter(
            parent_content_type=content_type, parent_object_id=doc.id).exists()
        frozen.validate()
    return [], [], omitted


def revalidate_figure_payloads(payloads, provenance, *, user):
    """An old auxiliary envelope cannot mint selected-chunk authority at handoff."""
    runtime = required_source_runtime()
    with runtime.lock, bounded_source_database(runtime, runtime.authorization.database_alias):
        from apps.documents.services.representation_tool_scope import tool_scope
        tool_scope(runtime, using=runtime.authorization.database_alias, require_existing=True).validate()
    return [], []
