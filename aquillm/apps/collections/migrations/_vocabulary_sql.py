"""Database guards for publication history, including deferred owner deletion."""
from django.conf import settings
from django.db.migrations.exceptions import IrreversibleError

def _tables(apps, editor):
    names = ("VocabularySet", "VocabularyDraft", "VocabularyVersion", "VocabularyEntry", "VocabularyAttachment", "Collection")
    result = {name: editor.quote_name(apps.get_model("apps_collections", name)._meta.db_table) for name in names}
    result["User"] = editor.quote_name(apps.get_model(settings.AUTH_USER_MODEL)._meta.db_table)
    return result

def install_guards(apps, editor):
    t = _tables(apps, editor)
    editor.execute(f"""
    CREATE FUNCTION vocabulary_owner_v1() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      IF TG_OP = 'UPDATE' AND NEW.owner_collection_id IS DISTINCT FROM OLD.owner_collection_id THEN
        RAISE EXCEPTION 'immutable vocabulary owner' USING ERRCODE = '23514';
      END IF;
      RETURN NEW;
    END $$;
    CREATE TRIGGER vocabulary_owner_v1 BEFORE UPDATE ON {t["VocabularySet"]}
      FOR EACH ROW EXECUTE FUNCTION vocabulary_owner_v1();

    CREATE FUNCTION vocabulary_draft_owner_v1() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      IF NEW.vocabulary_id IS DISTINCT FROM OLD.vocabulary_id THEN
        RAISE EXCEPTION 'immutable vocabulary draft owner' USING ERRCODE = '23514';
      END IF;
      RETURN NEW;
    END $$;
    CREATE TRIGGER vocabulary_draft_owner_v1 BEFORE UPDATE ON {t["VocabularyDraft"]}
      FOR EACH ROW EXECUTE FUNCTION vocabulary_draft_owner_v1();

    CREATE FUNCTION vocabulary_version_insert_v1() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      IF NOT EXISTS (SELECT 1 FROM {t["VocabularySet"]} WHERE id = NEW.vocabulary_id AND owner_collection_id = NEW.owner_collection_id) THEN
        RAISE EXCEPTION 'vocabulary owner mismatch' USING ERRCODE = '23514';
      END IF;
      RETURN NEW;
    END $$;
    CREATE TRIGGER vocabulary_version_insert_v1 BEFORE INSERT ON {t["VocabularyVersion"]}
      FOR EACH ROW EXECUTE FUNCTION vocabulary_version_insert_v1();

    CREATE FUNCTION vocabulary_entry_insert_v1() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      IF NOT EXISTS (SELECT 1 FROM {t["VocabularyVersion"]}
        WHERE id = NEW.version_id AND owner_collection_id = NEW.owner_collection_id
        AND xmin::text::bigint = mod(txid_current(), 4294967296)) THEN
        RAISE EXCEPTION 'published vocabulary cannot be appended' USING ERRCODE = '23514';
      END IF;
      RETURN NEW;
    END $$;
    CREATE TRIGGER vocabulary_entry_insert_v1 BEFORE INSERT ON {t["VocabularyEntry"]}
      FOR EACH ROW EXECUTE FUNCTION vocabulary_entry_insert_v1();

    CREATE FUNCTION vocabulary_version_update_v1() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      IF (to_jsonb(NEW) - 'published_by_id') IS DISTINCT FROM (to_jsonb(OLD) - 'published_by_id')
        OR OLD.published_by_id IS NULL OR NEW.published_by_id IS NOT NULL THEN
        RAISE EXCEPTION 'immutable vocabulary publication' USING ERRCODE = '23514';
      END IF;
      RETURN NEW;
    END $$;
    CREATE TRIGGER vocabulary_version_update_v1 BEFORE UPDATE ON {t["VocabularyVersion"]}
      FOR EACH ROW EXECUTE FUNCTION vocabulary_version_update_v1();

    CREATE FUNCTION vocabulary_actor_delete_v1() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      IF EXISTS (SELECT 1 FROM {t["User"]} WHERE id = OLD.published_by_id) THEN
        RAISE EXCEPTION 'publication actor may only be cleared by actor deletion' USING ERRCODE = '23514';
      END IF;
      RETURN NULL;
    END $$;
    CREATE CONSTRAINT TRIGGER vocabulary_actor_delete_v1 AFTER UPDATE ON {t["VocabularyVersion"]}
      DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION vocabulary_actor_delete_v1();

    CREATE FUNCTION vocabulary_entry_update_v1() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN RAISE EXCEPTION 'immutable vocabulary entry' USING ERRCODE = '23514'; END $$;
    CREATE TRIGGER vocabulary_entry_update_v1 BEFORE UPDATE ON {t["VocabularyEntry"]}
      FOR EACH ROW EXECUTE FUNCTION vocabulary_entry_update_v1();

    CREATE FUNCTION vocabulary_history_delete_v1() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      IF EXISTS (SELECT 1 FROM {t["Collection"]} WHERE id = OLD.owner_collection_id) THEN
        RAISE EXCEPTION 'published vocabulary history requires owner deletion' USING ERRCODE = '23514';
      END IF;
      RETURN NULL;
    END $$;
    CREATE CONSTRAINT TRIGGER vocabulary_version_delete_v1 AFTER DELETE ON {t["VocabularyVersion"]}
      DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION vocabulary_history_delete_v1();
    CREATE CONSTRAINT TRIGGER vocabulary_entry_delete_v1 AFTER DELETE ON {t["VocabularyEntry"]}
      DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION vocabulary_history_delete_v1();

    CREATE FUNCTION vocabulary_attachment_v1() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      IF TG_OP = 'UPDATE' AND (NEW.collection_id IS DISTINCT FROM OLD.collection_id
        OR NEW.vocabulary_id IS DISTINCT FROM OLD.vocabulary_id) THEN
        RAISE EXCEPTION 'immutable vocabulary attachment owner' USING ERRCODE = '23514';
      END IF;
      IF NOT EXISTS (SELECT 1 FROM {t["VocabularyVersion"]} WHERE id = NEW.pinned_version_id AND vocabulary_id = NEW.vocabulary_id) THEN
        RAISE EXCEPTION 'vocabulary attachment mismatch' USING ERRCODE = '23514';
      END IF;
      RETURN NEW;
    END $$;
    CREATE TRIGGER vocabulary_attachment_v1 BEFORE INSERT OR UPDATE ON {t["VocabularyAttachment"]}
      FOR EACH ROW EXECUTE FUNCTION vocabulary_attachment_v1();
    """)

def remove_guards(apps, editor):
    t = _tables(apps, editor)
    with editor.connection.cursor() as cursor:
        cursor.execute(f'SELECT EXISTS (SELECT 1 FROM {t["VocabularyVersion"]})')
        if cursor.fetchone()[0]:
            raise IrreversibleError("Published vocabulary history cannot be reversed.")
    for name in ("owner", "draft_owner", "version_insert", "entry_insert", "version_update", "actor_delete", "entry_update", "history_delete", "attachment"):
        editor.execute(f"DROP FUNCTION vocabulary_{name}_v1() CASCADE")
