"""Chat migration state must preserve the existing table and runtime defaults."""

from django.db.migrations.loader import MigrationLoader

from apps.chat.models import Message, WSConversation


def test_rating_index_state_matches_the_index_created_before_the_app_split():
    loader = MigrationLoader(None)
    legacy_state = loader.project_state(
        [("aquillm", "0008_wsconversation_system_prompt_message")]
    )
    current_state = loader.project_state()
    legacy_indexes = legacy_state.models[("aquillm", "message")].options["indexes"]
    current_indexes = current_state.models[("apps_chat", "message")].options["indexes"]

    assert [(index.name, index.fields) for index in current_indexes] == [
        (index.name, index.fields) for index in legacy_indexes
    ]


def test_migration_defaults_match_runtime_defaults_without_losing_legacy_backfill():
    loader = MigrationLoader(None)
    current_state = loader.project_state()
    legacy_state = loader.project_state([("apps_chat", "0004_message_app_version")])

    assert legacy_state.models[("apps_chat", "message")].fields[
        "app_version"
    ].default == "pre-0.1.0"
    for model, field_name in (
        (Message, "app_version"),
        (WSConversation, "system_prompt"),
    ):
        migrated_field = current_state.models[
            (model._meta.app_label, model._meta.model_name)
        ].fields[field_name]
        assert migrated_field.default is model._meta.get_field(field_name).default
