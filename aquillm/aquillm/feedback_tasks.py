import json
import logging

from celery import shared_task
from django.core.files.base import ContentFile
from django.db.models import Q
from django.utils import timezone

logger = logging.getLogger(__name__)


@shared_task(bind=True)
def generate_feedback_export(self, export_id):
    from .models import FeedbackExport, Message, WSConversation

    export = FeedbackExport.objects.get(pk=export_id)
    export.status = 'running'
    export.celery_task_id = self.request.id or ''
    export.save(update_fields=['status', 'celery_task_id'])

    try:
        # Find conversations that have at least one message with feedback
        feedback_messages = Message.objects.filter(
            Q(rating__isnull=False) | (Q(feedback_text__isnull=False) & ~Q(feedback_text=''))
        )
        conversation_ids = feedback_messages.values_list(
            'conversation_id', flat=True
        ).distinct()

        conversations = (
            WSConversation.objects
            .filter(pk__in=conversation_ids)
            .prefetch_related('db_messages')
            .select_related('owner')
        )

        export_data = {
            'exported_at': timezone.now().isoformat(),
            'conversations': [],
        }

        total_messages = 0
        for convo in conversations:
            messages = convo.db_messages.all().order_by('sequence_number')
            msg_list = []
            for msg in messages:
                msg_data = {
                    'message_uuid': str(msg.message_uuid),
                    'role': msg.role,
                    'content': msg.content,
                    'sequence_number': msg.sequence_number,
                    'created_at': msg.created_at.isoformat() if msg.created_at else None,
                    'rating': msg.rating,
                    'feedback_text': msg.feedback_text,
                }
                # Assistant-specific fields
                if msg.role == 'assistant':
                    msg_data.update({
                        'model': msg.model,
                        'stop_reason': msg.stop_reason,
                        'tool_call_id': msg.tool_call_id,
                        'tool_call_name': msg.tool_call_name,
                        'tool_call_input': msg.tool_call_input,
                        'usage': msg.usage,
                    })
                # Tool-specific fields
                elif msg.role == 'tool':
                    msg_data.update({
                        'tool_name': msg.tool_name,
                        'arguments': msg.arguments,
                        'for_whom': msg.for_whom,
                        'result_dict': msg.result_dict,
                    })
                msg_list.append(msg_data)

            total_messages += len(msg_list)
            export_data['conversations'].append({
                'conversation_id': convo.pk,
                'owner': convo.owner.username if convo.owner else None,
                'name': convo.name,
                'created_at': convo.created_at.isoformat() if convo.created_at else None,
                'messages': msg_list,
            })

        export_data['conversation_count'] = len(export_data['conversations'])
        export_data['message_count'] = total_messages

        json_bytes = json.dumps(export_data, indent=2, ensure_ascii=False).encode('utf-8')
        filename = f"feedback_export_{export.pk}_{timezone.now().strftime('%Y%m%d_%H%M%S')}.json"
        export.file.save(filename, ContentFile(json_bytes), save=False)
        export.conversation_count = len(export_data['conversations'])
        export.message_count = total_messages
        export.completed_at = timezone.now()
        export.status = 'completed'
        export.save(update_fields=[
            'file', 'conversation_count', 'message_count', 'completed_at', 'status',
        ])

        logger.info("Feedback export %s completed: %d conversations, %d messages",
                     export.pk, export.conversation_count, export.message_count)

    except Exception as exc:
        export.status = 'failed'
        export.error_message = str(exc)
        export.save(update_fields=['status', 'error_message'])
        raise
