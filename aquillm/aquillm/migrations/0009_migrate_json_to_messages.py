from django.db import migrations
import uuid as uuid_lib


def migrate_json_to_messages(apps, schema_editor):
    from django.db import transaction

    WSConversation = apps.get_model('aquillm', 'WSConversation')
    Message = apps.get_model('aquillm', 'Message')

    for convo in WSConversation.objects.exclude(convo__isnull=True):
        if not convo.convo:
            continue

        with transaction.atomic():
            # Extract system prompt from JSON blob
            convo.system_prompt = convo.convo.get('system', '')
            convo.save()

            # Convert each message in the JSON array to a Message row
            messages_to_create = []
            for seq, msg_data in enumerate(convo.convo.get('messages', [])):
                messages_to_create.append(Message(
                    conversation=convo,
                    message_uuid=msg_data.get('message_uuid', uuid_lib.uuid4()),
                    role=msg_data.get('role', 'user'),
                    content=msg_data.get('content', ''),
                    rating=msg_data.get('rating'),
                    sequence_number=seq,
                    # AssistantMessage fields
                    model=msg_data.get('model'),
                    stop_reason=msg_data.get('stop_reason'),
                    tool_call_id=msg_data.get('tool_call_id'),
                    tool_call_name=msg_data.get('tool_call_name'),
                    tool_call_input=msg_data.get('tool_call_input'),
                    usage=msg_data.get('usage', 0),
                    # ToolMessage fields
                    tool_name=msg_data.get('tool_name'),
                    arguments=msg_data.get('arguments'),
                    for_whom=msg_data.get('for_whom'),
                    result_dict=msg_data.get('result_dict'),
                ))
            Message.objects.bulk_create(messages_to_create)


class Migration(migrations.Migration):

    dependencies = [
        ('aquillm', '0008_wsconversation_system_prompt_message'),
    ]

    operations = [
        migrations.RunPython(migrate_json_to_messages, migrations.RunPython.noop),
    ]
