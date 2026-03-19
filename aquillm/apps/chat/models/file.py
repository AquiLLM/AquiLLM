"""ConversationFile model - files attached to conversations."""
from django.db import models

from .conversation import WSConversation


class ConversationFile(models.Model):
    file = models.FileField(upload_to='conversation_files/')
    name = models.CharField(max_length=200)
    conversation = models.ForeignKey(WSConversation, on_delete=models.CASCADE, related_name='files')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    message_uuid = models.UUIDField(null=True, blank=True)

    class Meta:
        app_label = 'apps_chat'
        db_table = 'aquillm_conversationfile'

    def __str__(self):
        return f"File {self.file.name} for conversation {self.conversation.id}"
