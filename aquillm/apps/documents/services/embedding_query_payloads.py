"""Pre-read admission for variable dense metadata under the original turn ledger."""
from math import isfinite

from django.db.models import F, Func, JSONField, Q, TextField
from django.db.models.functions import Cast, Length

from .source_loading import SourcePreparationLimited


class _UniqueManifestEntry(Func):
    output_field = JSONField()

    def as_sql(self, compiler, connection, **extra_context):
        manifest, manifest_params = compiler.compile(self.source_expressions[0])
        chunk, chunk_params = compiler.compile(self.source_expressions[1])
        sql = f"""(SELECT CASE WHEN count(*) = 1 THEN jsonb_agg(item)->0 ELSE NULL END
            FROM jsonb_array_elements(CASE WHEN jsonb_typeof({manifest}) = 'array'
                THEN {manifest} ELSE '[]'::jsonb END) AS matched(item)
            WHERE item->'chunk_id' = to_jsonb({chunk}))"""
        return sql, (*manifest_params, *manifest_params, *chunk_params)


class _TextSHA256(Func):
    template = "encode(sha256(convert_to(%(expressions)s, 'UTF8')), 'hex')"
    output_field = TextField()


_SCALARS = (
    "pk", "chunk_id", "generation_id", "generation__owner__representation_key",
    "input_sha256", "source_md5", "modality", "media_sha256", "distance",
)
_PAYLOADS = ("receipt", "serving_binding", "matched_manifest_entry")


def bounded_dense_payloads(rows, *, budget):
    """`rows` retains exact eligibility/source-MD5/order and <=45 SQL LIMIT."""
    query = rows.annotate(matched_manifest_entry=_UniqueManifestEntry(
        F("generation__manifest"), F("chunk_id")))
    annotations = {}
    for name in _PAYLOADS:
        text = Cast(F(name), TextField())
        annotations[f"{name}_length"] = Length(text)
        annotations[f"{name}_digest"] = _TextSHA256(text)
    query = query.annotate(**annotations)
    metadata = list(query.values(*_SCALARS, *annotations))
    admitted = [row for row in metadata if isfinite(row["distance"]) and all(
        type(row[f"{name}_length"]) is int and row[f"{name}_length"] > 0
        and row[f"{name}_digest"] is not None for name in _PAYLOADS)]
    if not admitted:
        return ()
    size = sum(row[f"{name}_length"] for row in admitted for name in _PAYLOADS)
    if not budget.reserve_text(size, kind="materialized"):
        raise SourcePreparationLimited("dense metadata allowance")
    # Exact metadata is repeated before hydration, including unique match count.
    # The original query's eligibility and source MD5 predicate also run again.
    unchanged = Q(pk__in=[])
    for row in admitted:
        unchanged |= Q(**row)
    payload = query.all()
    payload.query.clear_limits()
    return tuple(payload.filter(unchanged).values(*_SCALARS, "vector", *_PAYLOADS))
