"""Clean derived chunks for instance, queryset, and cascading document deletes."""


def delete_document_chunks(sender, instance, using, **kwargs):
    from .models import TextChunk

    TextChunk.objects.using(using).filter(doc_id=instance.id).delete()
