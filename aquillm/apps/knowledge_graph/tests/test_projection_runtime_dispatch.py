"""Related graph infrastructure regression scenarios."""

from apps.knowledge_graph.tests.test_projection_runtime import (
    _projection_environment,
    _projection_hook_environment,
    pytest,
    runtime,
)


def test_enabled_activation_dispatches_projection_outbox_after_commit(
    monkeypatch,
) -> None:
    from apps.knowledge_graph.projection import lifecycle, tasks

    callbacks = []
    dispatched = []
    monkeypatch.setattr(
        lifecycle,
        "enqueue_collection_projection_locked",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime.transaction,
        "on_commit",
        lambda callback, **_kwargs: callbacks.append(callback),
    )
    monkeypatch.setattr(
        tasks.reconcile_knowledge_graph_projections,
        "delay",
        lambda **kwargs: dispatched.append(kwargs),
    )

    assert runtime.enqueue_activated_collection_projection(
        7,
        9,
        source=_projection_hook_environment(),
    )
    assert dispatched == []
    assert len(callbacks) == 1

    callbacks[0]()

    assert dispatched == [{"collection_id": 7}]



def test_worker_postgres_factory_uses_the_live_configured_repository(
    monkeypatch,
) -> None:
    from apps.knowledge_graph.projection import worker

    settings = runtime.load_projection_runtime_settings(_projection_environment())
    repository = object()
    observed = []
    monkeypatch.setattr(worker, "_projection_settings", lambda: settings)
    state_repository = object()

    def factory(value, *, state_repository):
        observed.append((value, state_repository))
        return repository

    monkeypatch.setattr(worker, "postgres_projection_repository", factory)
    token = worker._STATE_REPOSITORY.set(state_repository)
    try:
        assert worker._postgres_repository() is repository
    finally:
        worker._STATE_REPOSITORY.reset(token)
    assert observed == [(settings, state_repository)]
    assert worker._state_using() == "projection_state"



def test_identifier_codec_factory_can_frame_a_persisted_key_version() -> None:
    environment = {
        **_projection_environment(),
        "KG_PROJECTION_IDENTIFIER_KEY_VERSION": "key-v2",
    }
    settings = runtime.load_projection_runtime_settings(environment)

    codec = runtime.projection_identifier_codec(settings, key_version="key-v1")

    assert codec.key_version == "key-v1"
    assert runtime.projection_identifier_codec(settings).key_version == "key-v2"



@pytest.mark.parametrize(
    ("value", "error"),
    ((1, TypeError), ("", ValueError), (" key-v1", ValueError)),
)
def test_identifier_codec_factory_rejects_noncanonical_version_without_secret(
    value, error
) -> None:
    settings = runtime.load_projection_runtime_settings(_projection_environment())

    with pytest.raises(error) as captured:
        runtime.projection_identifier_codec(settings, key_version=value)

    assert "identifier-secret" not in repr(captured.value)
