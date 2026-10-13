"""Invalidate derived profile provenance when the originating message is deleted."""
from django.db.models.signals import post_delete
from django.dispatch import receiver
from apps.chat.models import Message


@receiver(post_delete, sender=Message, dispatch_uid='memory_profile_source_deleted')
def profile_source_deleted(sender, instance, **kwargs):
    from .services.profile_facts import invalidate_deleted_source
    invalidate_deleted_source(instance)
