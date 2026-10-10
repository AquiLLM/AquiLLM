"""Reconcile metadata from the state-only app split without altering tables."""

import apps.documents.models.chunks
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("apps_collections", "0003_schema_head_and_generation_fencing"),
        ("apps_documents", "0005_chunkpublication"),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            database_operations=[],
            state_operations=[
                # aquillm.0017 and aquillm.0002 created these physical names;
                # apps_documents.0001 recorded different names only in state.
                migrations.RenameIndex(
                    model_name="documentfigure",
                    old_name="aquillm_doc_parent__f54ab6_idx",
                    new_name="aquillm_doc_parent__a1b2c3_idx",
                ),
                migrations.RenameIndex(
                    model_name="documentfigure",
                    old_name="aquillm_doc_source__cdd00f_idx",
                    new_name="aquillm_doc_source__d4e5f6_idx",
                ),
                migrations.RenameIndex(
                    model_name="textchunk",
                    old_name="aquillm_tex_doc_id_f31abe_idx",
                    new_name="aquillm_tex_doc_id_a7b188_idx",
                ),
                # Store the inherited field's template. Django expands it to
                # the same concrete reverse accessor on each model as before.
                *[
                    migrations.AlterField(
                        model_name=model_name,
                        name="collection",
                        field=models.ForeignKey(
                            on_delete=models.CASCADE,
                            related_name="%(class)s_documents",
                            to="apps_collections.collection",
                        ),
                    )
                    for model_name in (
                        "documentfigure",
                        "handwrittennotesdocument",
                        "imageuploaddocument",
                        "mediauploaddocument",
                        "pdfdocument",
                        "rawtextdocument",
                        "texdocument",
                        "vttdocument",
                    )
                ],
                migrations.AlterField(
                    model_name="textchunk",
                    name="doc_id",
                    field=models.UUIDField(
                        editable=False,
                        validators=[apps.documents.models.chunks.doc_id_validator],
                    ),
                ),
            ],
        ),
    ]
