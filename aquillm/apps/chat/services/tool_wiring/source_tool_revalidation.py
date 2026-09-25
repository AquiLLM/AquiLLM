"""Current body/figure checks for whole-document and adjacency handoff."""

from copy import deepcopy

from apps.documents.models import TextChunk
from apps.documents.services.source_loading import source_query_rows
from lib.retrieval.evidence import fingerprint_source

from .source_documents import required_source_runtime
from .source_figure_payloads import revalidate_figure_payloads


def revalidate_source_tool_result(result, *, user):
    """Task5 must call this immediately before a normal-loop provider request."""
    from .bounded_document_tools import _mark_figure_omission, limited_document_result

    runtime = required_source_runtime()
    records = result.get("_source_provenance", [])
    rows = source_query_rows(
        TextChunk.objects.using(
            runtime.authorization.database_alias,
        ).filter(pk__in=[r["chunk_id"] for r in records])
    )
    current = {row.pk: row for row in rows}
    if any(
        (row := current.get(record["chunk_id"])) is None
        or str(row.doc_id) != record["document_id"]
        or row.chunk_number != record["chunk_number"]
        or fingerprint_source(row.content) != record["source_fingerprint"]
        for record in records
    ):
        return limited_document_result()
    refreshed = deepcopy(result)
    payload = refreshed.get("result")
    if isinstance(payload, dict) and payload.get("type") == "document_with_figures":
        figures, identities = revalidate_figure_payloads(
            payload["figures"],
            refreshed.get("_figure_provenance", []),
            user=user,
        )
        if len(figures) != len(payload["figures"]):
            _mark_figure_omission(refreshed)
        if figures:
            payload["figures"] = figures
            refreshed["_figure_provenance"] = identities
        else:
            refreshed["result"] = payload["text"]
            refreshed.pop("_figure_provenance", None)
            refreshed.pop("_image_instruction", None)
    return refreshed
