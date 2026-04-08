"""WebSocket Conversation model."""
import structlog
import re
from typing import Optional

from django.apps import apps
from django.contrib.auth.models import User
from django.db import models
from django.utils import timezone

logger = structlog.stdlib.get_logger(__name__)


def get_default_system_prompt():
    """Pulls the default system prompt from the app config."""
    return apps.get_app_config('aquillm').system_prompt


class WSConversation(models.Model):
    owner = models.ForeignKey(User, related_name='ws_conversations', on_delete=models.CASCADE)
    system_prompt = models.TextField(default=get_default_system_prompt, blank=True)
    name = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(editable=False)
    updated_at = models.DateTimeField()

    class Meta:
        app_label = 'apps_chat'
        db_table = 'aquillm_wsconversation'

    def save(self, *args, **kwargs):
        if not self.pk:
            self.created_at = timezone.now()
        self.updated_at = timezone.now()
        return super().save(*args, **kwargs)

    @staticmethod
    def _clean_generated_title(title: Optional[str]) -> Optional[str]:
        if not title:
            return None
        cleaned = title.strip().strip('"').strip("'").strip('*').strip()
        return cleaned if cleaned else None

    @staticmethod
    def _is_generic_title(title: Optional[str]) -> bool:
        if not title:
            return True
        normalized = title.strip().lower()
        generic_titles = {'conversation', 'untitled conversation', 'new conversation', 'chat'}
        return normalized in generic_titles

    @staticmethod
    def _fallback_title_from_user_message(user_message: Optional[str]) -> str:
        if not user_message:
            return 'Conversation'
        cleaned = re.sub(r'\s+', ' ', user_message).strip()
        cleaned = cleaned.strip('"').strip("'").strip('*').strip()
        cleaned = cleaned.strip(".,:;!?- ")
        if not cleaned:
            return 'Conversation'

        words = cleaned.split(' ')
        title = ' '.join(words[:8]).strip()
        if len(words) > 8:
            title = f"{title}..."
        if title and title[0].isalpha():
            title = title[0].upper() + title[1:]
        return title or 'Conversation'

    def set_name(self):
        from asgiref.sync import async_to_sync

        system_prompt = """
        This is a conversation between a large langauge model and a user.
        Come up with a brief, roughly 3 to 10 word title for the conversation capturing what the user asked.
        Respond only with the title.
        As an example, if the conversation begins 'What is apple pie made of?', your response should be 'Apple Pie Ingredients'.
        The title should capture what is being asked, not what the assistant responded with.
        If there is not enough information to name the conversation, simply return 'Conversation'.
        """

        llm_interface = apps.get_app_config('aquillm').llm_interface
        first_user_message = self.db_messages.filter(role='user').order_by('sequence_number').values_list('content', flat=True).first()

        llm_args = {
            **llm_interface.base_args,
            'max_tokens': 30,
            'thinking_budget': 0,
            'system': system_prompt,
            'messages': [{'role': 'user', 'content': first_user_message or ''}]
        }

        @async_to_sync
        async def get_title():
            response = await llm_interface.get_message(**llm_args)
            return response.text

        title_text: Optional[str] = None
        try:
            title_text = get_title()
        except Exception as exc:
            logger.warning("obs.chat.auto_title_error", conversation_id=self.pk, error_type=type(exc).__name__, error=str(exc))

        title_text = self._clean_generated_title(title_text)
        if self._is_generic_title(title_text):
            self.name = self._fallback_title_from_user_message(first_user_message)
        else:
            self.name = title_text
        self.save()
