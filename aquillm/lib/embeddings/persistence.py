"""Django write guards for vector/receipt pairs (raw SQL remains out of scope)."""

from django.db import models

from .provenance import valid_provenance


def prepare_embedding_save(instance, kwargs):
    fields = kwargs.get("update_fields")
    if fields is not None:
        fields = set(fields)
        if "embedding" not in fields:
            if (
                "embedding_provenance" in fields
                and instance.embedding_provenance is not None
            ):
                raise ValueError("Write embedding and provenance together")
            return
        kwargs["update_fields"] = fields | {"embedding_provenance"}
    instance.embedding_provenance = valid_provenance(
        instance.embedding, instance.embedding_provenance
    )


class EmbeddingQuerySet(models.QuerySet):
    """Clear unknown/stale receipts and keep supported vector writes atomic."""

    _validated_embedding_bulk_update = False

    def _clone(self):
        clone = super()._clone()
        clone._validated_embedding_bulk_update = self._validated_embedding_bulk_update
        return clone

    def update(self, **kwargs):
        if not self._validated_embedding_bulk_update:
            if "embedding" in kwargs:
                kwargs["embedding_provenance"] = valid_provenance(
                    kwargs["embedding"], kwargs.get("embedding_provenance")
                )
            elif kwargs.get("embedding_provenance") is not None:
                raise ValueError("Write embedding and provenance together")
        return super().update(**kwargs)

    def bulk_create(
        self,
        objs,
        batch_size=None,
        ignore_conflicts=False,
        update_conflicts=False,
        update_fields=None,
        unique_fields=None,
    ):
        objs = list(objs)
        for obj in objs:
            obj.embedding_provenance = valid_provenance(
                obj.embedding, obj.embedding_provenance
            )
        # Conflict updates must update the pair as well as initial inserts.
        fields = update_fields
        if fields:
            fields = set(fields)
            if "embedding" in fields:
                update_fields = list(fields | {"embedding_provenance"})
            elif "embedding_provenance" in fields:
                raise ValueError("Write embedding and provenance together")
        return super().bulk_create(
            objs,
            batch_size=batch_size,
            ignore_conflicts=ignore_conflicts,
            update_conflicts=update_conflicts,
            update_fields=update_fields,
            unique_fields=unique_fields,
        )

    def bulk_update(self, objs, fields, batch_size=None):
        objs, fields = list(objs), set(fields)
        if "embedding" in fields:
            for obj in objs:
                obj.embedding_provenance = valid_provenance(
                    obj.embedding, obj.embedding_provenance
                )
            fields.add("embedding_provenance")
        elif "embedding_provenance" in fields:
            raise ValueError("Write embedding and provenance together")
        # Django emits CASE expressions via update(); validate objects before
        # compiling SQL, then preserve those paired expressions on this clone.
        validated = self._chain()
        validated._validated_embedding_bulk_update = True
        return super(EmbeddingQuerySet, validated).bulk_update(
            objs, list(fields), batch_size
        )
