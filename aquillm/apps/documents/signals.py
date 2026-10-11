"""Clean derived chunks for instance, queryset, and cascading document deletes."""


def delete_document_chunks(sender, instance, using, **kwargs):
    from .services.representation_history import delete_legacy_remainder
    delete_legacy_remainder(instance, using=using)
