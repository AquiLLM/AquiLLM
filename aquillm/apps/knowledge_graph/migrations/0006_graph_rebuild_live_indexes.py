from django.db import migrations, models

_LIVE_CHECKS = (
    (
        "GraphArtifact",
        "kg_artifact_eval_noncurrent",
        '"evaluation_only" = FALSE OR '
        '("rebuild_request_id" IS NOT NULL AND "status" <> \'active\')',
    ),
    (
        "GraphBuildRun",
        "kg_build_eval_noncurrent",
        '"evaluation_only" = FALSE OR '
        '("rebuild_request_id" IS NOT NULL '
        "AND \"stage\" <> 'active' AND \"status\" <> 'succeeded')",
    ),
)

_LIVE_FOREIGN_KEYS = (
    ("GraphArtifact", "kg_artifact_rebuild_request_fk"),
    ("GraphBuildRun", "kg_run_rebuild_request_fk"),
)

_LIVE_INDEXES = (
    (
        "GraphArtifact",
        "kg_art_rebuild_req_idx",
        ("rebuild_request_id",),
        False,
        "",
    ),
    (
        "GraphBuildRun",
        "kg_run_rebuild_req_idx",
        ("rebuild_request_id",),
        False,
        "",
    ),
    (
        "GraphArtifact",
        "kg_art_terminal_idx",
        ("status", "completed_at", "id"),
        False,
        "",
    ),
    (
        "GraphArtifact",
        "kg_art_superseded_idx",
        ("status", "superseded_at", "id"),
        False,
        "",
    ),
    (
        "GraphBuildRun",
        "kg_run_terminal_idx",
        ("status", "stage", "finished_at", "id"),
        False,
        "",
    ),
    (
        "GraphBuildRun",
        "kg_run_scope_gen_idx",
        ("build_kind", "scope_type", "scope_id", "build_generation"),
        False,
        "",
    ),
    (
        "GraphArtifact",
        "kg_artifact_request_scope_unique",
        ("rebuild_request_id", "scope_type", "scope_id"),
        True,
        '"rebuild_request_id" IS NOT NULL',
    ),
    (
        "GraphBuildRun",
        "kg_run_request_scope_unique",
        ("rebuild_request_id", "scope_type", "scope_id"),
        True,
        '"rebuild_request_id" IS NOT NULL',
    ),
)


def _constraint_for(name):
    if name == "kg_artifact_eval_noncurrent":
        return models.CheckConstraint(
            condition=(
                models.Q(evaluation_only=False)
                | (models.Q(rebuild_request__isnull=False) & ~models.Q(status="active"))
            ),
            name=name,
        )
    return models.CheckConstraint(
        condition=(
            models.Q(evaluation_only=False)
            | (
                models.Q(rebuild_request__isnull=False)
                & ~models.Q(stage="active")
                & ~models.Q(status="succeeded")
            )
        ),
        name=name,
    )


def _database_object_exists(schema_editor, table_name, object_name):
    with schema_editor.connection.cursor() as cursor:
        constraints = schema_editor.connection.introspection.get_constraints(
            cursor, table_name
        )
    return object_name in constraints


def _install_check_constraint(apps, schema_editor, *, model_name, name, expression):
    model = apps.get_model("apps_knowledge_graph", model_name)
    quote = schema_editor.quote_name
    table_name = model._meta.db_table
    if schema_editor.connection.vendor != "postgresql":
        if not _database_object_exists(schema_editor, table_name, name):
            schema_editor.add_constraint(model, _constraint_for(name))
        return

    with schema_editor.connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT constraint_row.convalidated
              FROM pg_catalog.pg_constraint AS constraint_row
              JOIN pg_catalog.pg_class AS table_row
                ON table_row.oid = constraint_row.conrelid
              JOIN pg_catalog.pg_namespace AS namespace_row
                ON namespace_row.oid = table_row.relnamespace
             WHERE namespace_row.nspname = current_schema()
               AND table_row.relname = %s
               AND constraint_row.conname = %s
            """,
            (table_name, name),
        )
        existing = cursor.fetchone()

    if existing is None:
        schema_editor.execute(
            f"ALTER TABLE {quote(table_name)} ADD CONSTRAINT {quote(name)} "
            f"CHECK ({expression}) NOT VALID"
        )
    if existing is None or not existing[0]:
        schema_editor.execute(
            f"ALTER TABLE {quote(table_name)} VALIDATE CONSTRAINT {quote(name)}"
        )


def _plain_uuid_field(model):
    field = models.UUIDField(
        blank=True,
        db_column="rebuild_request_id",
        db_index=False,
        editable=False,
        null=True,
    )
    field.set_attributes_from_name("rebuild_request")
    field.model = model
    return field


def _install_foreign_key(apps, schema_editor, *, model_name, name):
    model = apps.get_model("apps_knowledge_graph", model_name)
    target = apps.get_model("apps_knowledge_graph", "GraphRebuildRequest")
    quote = schema_editor.quote_name
    table_name = model._meta.db_table
    target_table = target._meta.db_table
    if schema_editor.connection.vendor != "postgresql":
        with schema_editor.connection.cursor() as cursor:
            constraints = schema_editor.connection.introspection.get_constraints(
                cursor, table_name
            )
        expected = (target_table, "id")
        if not any(
            details.get("columns") == ["rebuild_request_id"]
            and details.get("foreign_key") == expected
            for details in constraints.values()
        ):
            schema_editor.alter_field(
                model,
                _plain_uuid_field(model),
                model._meta.get_field("rebuild_request"),
                strict=True,
            )
        return

    with schema_editor.connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT constraint_row.conname, constraint_row.convalidated
              FROM pg_catalog.pg_constraint AS constraint_row
              JOIN pg_catalog.pg_class AS table_row
                ON table_row.oid = constraint_row.conrelid
              JOIN pg_catalog.pg_namespace AS namespace_row
                ON namespace_row.oid = table_row.relnamespace
              JOIN pg_catalog.pg_attribute AS column_row
                ON column_row.attrelid = table_row.oid
               AND column_row.attname = %s
             WHERE namespace_row.nspname = current_schema()
               AND table_row.relname = %s
               AND constraint_row.contype = 'f'
               AND constraint_row.conkey = ARRAY[column_row.attnum]::smallint[]
               AND constraint_row.confrelid = to_regclass(%s)
            """,
            ("rebuild_request_id", table_name, target_table),
        )
        existing = cursor.fetchone()

    constraint_name = name if existing is None else existing[0]
    if existing is None:
        schema_editor.execute(
            f"ALTER TABLE {quote(table_name)} ADD CONSTRAINT {quote(name)} "
            f"FOREIGN KEY ({quote('rebuild_request_id')}) "
            f"REFERENCES {quote(target_table)} ({quote('id')}) "
            "DEFERRABLE INITIALLY DEFERRED NOT VALID"
        )
    if existing is None or not existing[1]:
        schema_editor.execute(
            f"ALTER TABLE {quote(table_name)} "
            f"VALIDATE CONSTRAINT {quote(constraint_name)}"
        )


def _create_index(
    apps,
    schema_editor,
    *,
    model_name,
    name,
    columns,
    unique,
    where,
):
    model = apps.get_model("apps_knowledge_graph", model_name)
    quote = schema_editor.quote_name
    table_name = model._meta.db_table
    if schema_editor.connection.vendor == "postgresql":
        with schema_editor.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT index_row.indisvalid
                  FROM pg_catalog.pg_index AS index_row
                  JOIN pg_catalog.pg_class AS index_class
                    ON index_class.oid = index_row.indexrelid
                  JOIN pg_catalog.pg_namespace AS namespace_row
                    ON namespace_row.oid = index_class.relnamespace
                 WHERE namespace_row.nspname = current_schema()
                   AND index_class.relname = %s
                """,
                (name,),
            )
            existing = cursor.fetchone()
        if existing is not None and existing[0]:
            return
        if existing is not None:
            schema_editor.execute(f"DROP INDEX CONCURRENTLY {quote(name)}")
        concurrent = " CONCURRENTLY"
    else:
        if _database_object_exists(schema_editor, table_name, name):
            return
        concurrent = ""

    uniqueness = " UNIQUE" if unique else ""
    predicate = f" WHERE {where}" if where else ""
    schema_editor.execute(
        f"CREATE{uniqueness} INDEX{concurrent} IF NOT EXISTS {quote(name)} "
        f"ON {quote(table_name)} "
        f"({', '.join(quote(column) for column in columns)}){predicate}"
    )


def _install_live_schema(apps, schema_editor):
    if schema_editor.connection.vendor != "postgresql":
        for model_name, name in _LIVE_FOREIGN_KEYS:
            _install_foreign_key(
                apps,
                schema_editor,
                model_name=model_name,
                name=name,
            )
    for model_name, name, columns, unique, where in _LIVE_INDEXES:
        _create_index(
            apps,
            schema_editor,
            model_name=model_name,
            name=name,
            columns=columns,
            unique=unique,
            where=where,
        )
    if schema_editor.connection.vendor == "postgresql":
        for model_name, name in _LIVE_FOREIGN_KEYS:
            _install_foreign_key(
                apps,
                schema_editor,
                model_name=model_name,
                name=name,
            )
    for model_name, name, expression in _LIVE_CHECKS:
        _install_check_constraint(
            apps,
            schema_editor,
            model_name=model_name,
            name=name,
            expression=expression,
        )


def _remove_live_schema(apps, schema_editor):
    quote = schema_editor.quote_name
    for model_name, name, _expression in reversed(_LIVE_CHECKS):
        model = apps.get_model("apps_knowledge_graph", model_name)
        if not _database_object_exists(schema_editor, model._meta.db_table, name):
            continue
        if schema_editor.connection.vendor == "postgresql":
            schema_editor.execute(
                f"ALTER TABLE {quote(model._meta.db_table)} "
                f"DROP CONSTRAINT IF EXISTS {quote(name)}"
            )
        else:
            schema_editor.remove_constraint(model, _constraint_for(name))

    for model_name, name in reversed(_LIVE_FOREIGN_KEYS):
        model = apps.get_model("apps_knowledge_graph", model_name)
        if schema_editor.connection.vendor == "postgresql":
            schema_editor.execute(
                f"ALTER TABLE {quote(model._meta.db_table)} "
                f"DROP CONSTRAINT IF EXISTS {quote(name)}"
            )
        else:
            schema_editor.alter_field(
                model,
                model._meta.get_field("rebuild_request"),
                _plain_uuid_field(model),
                strict=True,
            )

    for _model_name, name, _columns, _unique, _where in reversed(_LIVE_INDEXES):
        concurrent = (
            " CONCURRENTLY" if schema_editor.connection.vendor == "postgresql" else ""
        )
        schema_editor.execute(f"DROP INDEX{concurrent} IF EXISTS {quote(name)}")


class Migration(migrations.Migration):
    # Every operation is restart-idempotent. PostgreSQL builds live-table indexes
    # concurrently and validates checks without a long write-blocking table scan.
    atomic = False

    dependencies = [
        ("apps_knowledge_graph", "0005_graph_rebuild_request"),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            database_operations=[
                migrations.RunPython(
                    _install_live_schema,
                    _remove_live_schema,
                    atomic=False,
                )
            ],
            state_operations=[
                migrations.AddConstraint(
                    model_name="graphartifact",
                    constraint=models.CheckConstraint(
                        condition=(
                            models.Q(evaluation_only=False)
                            | (
                                models.Q(rebuild_request__isnull=False)
                                & ~models.Q(status="active")
                            )
                        ),
                        name="kg_artifact_eval_noncurrent",
                    ),
                ),
                migrations.AddConstraint(
                    model_name="graphbuildrun",
                    constraint=models.CheckConstraint(
                        condition=(
                            models.Q(evaluation_only=False)
                            | (
                                models.Q(rebuild_request__isnull=False)
                                & ~models.Q(stage="active")
                                & ~models.Q(status="succeeded")
                            )
                        ),
                        name="kg_build_eval_noncurrent",
                    ),
                ),
                migrations.AddConstraint(
                    model_name="graphartifact",
                    constraint=models.UniqueConstraint(
                        fields=("rebuild_request", "scope_type", "scope_id"),
                        condition=models.Q(rebuild_request__isnull=False),
                        name="kg_artifact_request_scope_unique",
                    ),
                ),
                migrations.AddConstraint(
                    model_name="graphbuildrun",
                    constraint=models.UniqueConstraint(
                        fields=("rebuild_request", "scope_type", "scope_id"),
                        condition=models.Q(rebuild_request__isnull=False),
                        name="kg_run_request_scope_unique",
                    ),
                ),
                migrations.AddIndex(
                    model_name="graphartifact",
                    index=models.Index(
                        fields=["status", "completed_at", "id"],
                        name="kg_art_terminal_idx",
                    ),
                ),
                migrations.AddIndex(
                    model_name="graphartifact",
                    index=models.Index(
                        fields=["status", "superseded_at", "id"],
                        name="kg_art_superseded_idx",
                    ),
                ),
                migrations.AddIndex(
                    model_name="graphbuildrun",
                    index=models.Index(
                        fields=["status", "stage", "finished_at", "id"],
                        name="kg_run_terminal_idx",
                    ),
                ),
                migrations.AddIndex(
                    model_name="graphbuildrun",
                    index=models.Index(
                        fields=[
                            "build_kind",
                            "scope_type",
                            "scope_id",
                            "build_generation",
                        ],
                        name="kg_run_scope_gen_idx",
                    ),
                ),
            ],
        )
    ]
