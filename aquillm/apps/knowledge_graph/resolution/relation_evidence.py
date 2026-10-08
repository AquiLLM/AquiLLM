"""Project relation support without reintroducing audited source exclusions."""

from .source_exclusions import EXCLUSION_FIELDS, exclusion_marker_is_valid


def relation_participation(mention_ids, relation_rows, source_artifact_ids, config):
    """Keep v1/subset support semantics; exclude only declared v2 relation IDs.

    Source artifacts and their runs are locked by the calling collection loader.
    The raw relation snapshot remains unchanged and still includes every row.
    """
    from django.db.models import F, Q

    from apps.knowledge_graph.models import GraphBuildRun

    from .collection import _bounded_batched_query_rows
    from .coreference import MAX_DOCUMENT_MENTIONS

    artifact_ids = tuple(sorted(set(source_artifact_ids)))
    if len(artifact_ids) > config.max_document_inputs:
        raise ValueError("relation exclusion source artifact cap exceeded")

    def audit_query(artifact_id_batch):
        return (
            GraphBuildRun.objects.filter(
                Q(stats__resolution_commit__version=2)
                | Q(stats__resolution_commit__has_any_keys=list(EXCLUSION_FIELDS)),
                artifact_id__in=artifact_id_batch,
                build_key=F("artifact__build_key"),
                build_generation=F("artifact__build_generation"),
            )
            .order_by("pk")
            .values_list("pk", "artifact_id", "stats__resolution_commit")
        )

    audits = _bounded_batched_query_rows(
        artifact_ids,
        audit_query,
        maximum=config.max_document_inputs,
        label="relation exclusion audit",
        row_key=lambda row: row[0],
        sort_key=lambda row: row[0],
    )
    excluded = set()
    seen = set()
    for _run_id, artifact_id, marker in audits:
        audit = {key: marker.get(key) for key in EXCLUSION_FIELDS}
        if (
            artifact_id in seen
            or type(marker.get("version")) is not int
            or not exclusion_marker_is_valid(marker, audit)
            or len(audit["excluded_mention_ids"]) > MAX_DOCUMENT_MENTIONS
        ):
            raise ValueError("source relation exclusion audit is invalid")
        seen.add(artifact_id)
        excluded.update(audit["excluded_relation_ids"])
        if len(excluded) > config.max_relations:
            raise ValueError("relation exclusion cap exceeded")
    participation = dict.fromkeys(mention_ids, 0)
    for relation_id, head_id, tail_id in relation_rows:
        if relation_id in excluded:
            continue
        if head_id in participation:
            participation[head_id] += 1
        if tail_id in participation:
            participation[tail_id] += 1
    return participation
