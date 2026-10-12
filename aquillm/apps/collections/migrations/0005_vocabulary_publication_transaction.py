"""Append only within the database-assigned original publication transaction."""
from django.db import migrations, models
from django.db.migrations.exceptions import IrreversibleError


def _tables(apps, editor):
    return {name: editor.quote_name(apps.get_model("apps_collections", name)._meta.db_table)
            for name in ("VocabularySet", "VocabularyVersion")}


def _replace_guards(apps, editor, *, provenance):
    tables = _tables(apps, editor)
    assignment = "NEW.publication_transaction_id := txid_current();" if provenance else ""
    admission = ("publication_transaction_id = txid_current()" if provenance
                 else "xmin::text::bigint = mod(txid_current(), 4294967296)")
    editor.execute(f"""
    CREATE OR REPLACE FUNCTION vocabulary_version_insert_v1() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      IF NOT EXISTS (SELECT 1 FROM {tables["VocabularySet"]} WHERE id = NEW.vocabulary_id AND owner_collection_id = NEW.owner_collection_id) THEN
        RAISE EXCEPTION 'vocabulary owner mismatch' USING ERRCODE = '23514';
      END IF;
      {assignment}
      RETURN NEW;
    END $$;
    CREATE OR REPLACE FUNCTION vocabulary_entry_insert_v1() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      IF NOT EXISTS (SELECT 1 FROM {tables["VocabularyVersion"]}
        WHERE id = NEW.version_id AND owner_collection_id = NEW.owner_collection_id
        AND {admission}) THEN
        RAISE EXCEPTION 'published vocabulary cannot be appended' USING ERRCODE = '23514';
      END IF;
      RETURN NEW;
    END $$;
    """)


def install(apps, editor):
    _replace_guards(apps, editor, provenance=True)


def reverse(apps, editor):
    table = _tables(apps, editor)["VocabularyVersion"]
    with editor.connection.cursor() as cursor:
        cursor.execute(f"SELECT EXISTS (SELECT 1 FROM {table})")
        if cursor.fetchone()[0]:
            raise IrreversibleError("Published vocabulary transaction provenance cannot be reversed.")
    _replace_guards(apps, editor, provenance=False)


class Migration(migrations.Migration):
    dependencies = [("apps_collections", "0004_vocabulary_versions_v1")]
    operations = [
        migrations.AddField(
            model_name="vocabularyversion", name="publication_transaction_id",
            field=models.BigIntegerField(default=0, editable=False),
        ),
        migrations.RunPython(install, reverse),
    ]
