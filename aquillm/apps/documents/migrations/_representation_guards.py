"""Frozen PostgreSQL history guards; no runtime model/service imports."""
from ._representation_identity_sql import IDENTITY_SQL
from ._representation_chunk_sql import CHUNK_SQL


DOCUMENT_MODELS = (
    "PDFDocument", "TeXDocument", "RawTextDocument", "VTTDocument",
    "HandwrittenNotesDocument", "ImageUploadDocument", "MediaUploadDocument",
    "DocumentFigure",
)


def install(apps, editor):
    quote = editor.quote_name
    models = [apps.get_model("apps_documents", name) for name in DOCUMENT_MODELS]
    branches = "\n".join(
        f"WHEN '{model._meta.label_lower}' THEN RETURN QUERY SELECT d.id, d.collection_id::bigint, "
        f"d.full_text_hash::text, d.full_text FROM {quote(model._meta.db_table)} d WHERE d.pkid = wanted_pk;"
        for model in models
    )
    counts = " UNION ALL ".join(
        f"SELECT id FROM {quote(model._meta.db_table)} WHERE id = wanted_uuid"
        for model in models
    )
    editor.execute(f"""
    CREATE FUNCTION representation_owner_v1(wanted_label text, wanted_pk bigint)
    RETURNS TABLE(document_id uuid, collection_id bigint, source_hash text, canonical_text text)
    LANGUAGE plpgsql STABLE AS $$ BEGIN CASE wanted_label {branches}
    ELSE RAISE EXCEPTION 'invalid_representation_owner'; END CASE; END $$;
    CREATE FUNCTION representation_owner_count_v1(wanted_uuid uuid) RETURNS bigint
    LANGUAGE sql STABLE AS $$ SELECT count(*) FROM ({counts}) AS owners $$;
    """)
    editor.execute(IDENTITY_SQL, params=None)
    editor.execute(CHUNK_SQL, params=None)
    for table in ("aquillm_representationstate", "aquillm_representationgeneration",
                  "aquillm_legacysourcesnapshot", "aquillm_textchunk"):
        editor.execute(f"""CREATE CONSTRAINT TRIGGER representation_delete_v1
        AFTER DELETE ON {quote(table)} DEFERRABLE INITIALLY DEFERRED
        FOR EACH ROW EXECUTE FUNCTION representation_delete_v1();""")
    for model in models:
        editor.execute(f"""CREATE TRIGGER representation_owner_update_v1
        BEFORE UPDATE ON {quote(model._meta.db_table)}
        FOR EACH ROW EXECUTE FUNCTION representation_owner_update_v1('{model._meta.label_lower}');""")
        editor.execute(f"""CREATE CONSTRAINT TRIGGER representation_owner_delete_v1
        AFTER DELETE ON {quote(model._meta.db_table)} DEFERRABLE INITIALLY DEFERRED
        FOR EACH ROW EXECUTE FUNCTION representation_owner_delete_v1('{model._meta.label_lower}');""")


def uninstall(apps, editor):
    for name in DOCUMENT_MODELS:
        table = editor.quote_name(apps.get_model("apps_documents", name)._meta.db_table)
        editor.execute(f"DROP TRIGGER IF EXISTS representation_owner_delete_v1 ON {table}")
        editor.execute(f"DROP TRIGGER IF EXISTS representation_owner_update_v1 ON {table}")
    for table in ("aquillm_representationstate", "aquillm_representationgeneration",
                  "aquillm_legacysourcesnapshot", "aquillm_textchunk"):
        editor.execute(f"DROP TRIGGER IF EXISTS representation_delete_v1 ON {table}")
    for table, name in (("aquillm_representationstate", "representation_state_v1"),
                        ("aquillm_representationgeneration", "representation_generation_v1"),
                        ("aquillm_legacysourcesnapshot", "representation_snapshot_v1"),
                        ("aquillm_textchunk", "representation_chunk_v1")):
        editor.execute(f"DROP TRIGGER IF EXISTS {name} ON {table}")
    for table in ("aquillm_representationstate", "aquillm_representationgeneration"):
        editor.execute(f"DROP TRIGGER IF EXISTS representation_pointer_v1 ON {table}")
    for signature in ("representation_owner_delete_v1()", "representation_owner_update_v1()", "representation_delete_v1()",
                      "representation_chunk_v1()", "representation_snapshot_v1()",
                      "representation_generation_v1()", "representation_state_v1()",
                      "representation_pointer_v1()", "representation_json_v1(jsonb)", "representation_string_v1(text)",
                      "representation_owner_count_v1(uuid)", "representation_owner_v1(text,bigint)"):
        editor.execute(f"DROP FUNCTION IF EXISTS {signature}")
