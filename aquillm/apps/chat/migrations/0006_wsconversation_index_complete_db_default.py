# Set a database-level default on index_complete so inserts that omit the column
# (e.g. a stale app process during a deploy) don't violate the NOT NULL constraint.
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('apps_chat', '0005_conversationchunk_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='wsconversation',
            name='index_complete',
            field=models.BooleanField(db_default=False, default=False),
        ),
    ]
