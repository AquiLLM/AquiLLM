"""Validate bounded graph body loads without changing requested graph order."""

from uuid import UUID

from .chunk_search_validation import candidate_identifier
from .source_loading import source_query_rows


def materialize_graph_rows(model_cls, novel_ids, allowed_doc_ids):
    loaded = source_query_rows(
        model_cls.objects.filter(pk__in=novel_ids, doc_id__in=allowed_doc_ids)
    )
    by_identifier = {}
    allowed = set(allowed_doc_ids)
    invalid = False
    for row in loaded:
        identifier = candidate_identifier(row)
        document_id = getattr(row, "doc_id", None)
        if (
            identifier not in novel_ids
            or identifier in by_identifier
            or type(document_id) is not UUID
            or document_id not in allowed
            or (hasattr(model_cls, "_meta") and not isinstance(row, model_cls))
        ):
            invalid = True
            continue
        by_identifier[identifier] = row
    if invalid or set(novel_ids).difference(by_identifier):
        return (), len(novel_ids)
    return tuple(by_identifier[identifier] for identifier in novel_ids), 0
