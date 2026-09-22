from __future__ import annotations

import math

import pytest
from test_collection_resolution import (
    _EMBEDDING_SIGNATURE,
    _document_entity,
    _ontology,
    _RecordingBackend,
    _session,
    _snapshot,
    _unit_vector,
)

from apps.knowledge_graph.resolution.collection import (
    CollectionEmbeddingSession,
    CollectionResolutionConfig,
    SignedEmbeddingBatch,
    embedding_text_hash,
    resolve_collection_entities,
)


def test_embedding_session_rejects_provider_or_model_signature_drift():
    expected = (
        f"local:model-a@rev:endpoint={'e' * 64}:dims=1024:"
        "prep=kg-entity-v1:max_chars=8192:batch=64"
    )
    backend = _RecordingBackend(
        {"Atlas": _unit_vector(1.0)},
        signature=(
            f"cohere:model-b@rev:endpoint={'f' * 64}:dims=1024:"
            "prep=kg-entity-v1:max_chars=8192:batch=64"
        ),
    )
    session = CollectionEmbeddingSession(
        expected_model_signature=expected,
        backend=backend,
    )

    with pytest.raises(ValueError, match="signature.*drift"):
        session.embed(("Atlas",))


@pytest.mark.parametrize(
    "batch_factory, message",
    [
        (
            lambda texts, signature: SignedEmbeddingBatch(
                vectors=(),
                text_hashes=(),
                indices=(),
                model_signature=signature,
            ),
            "one vector",
        ),
        (
            lambda texts, signature: SignedEmbeddingBatch(
                vectors=(tuple([math.nan] + [0.0] * 1023),),
                text_hashes=(embedding_text_hash(texts[0]),),
                indices=(0,),
                model_signature=signature,
            ),
            "finite",
        ),
        (
            lambda texts, signature: SignedEmbeddingBatch(
                vectors=(tuple(_unit_vector(1.0)),),
                text_hashes=("f" * 64,),
                indices=(0,),
                model_signature=signature,
            ),
            "order|hash",
        ),
    ],
)
def test_embedding_session_validates_count_finiteness_and_output_order(
    batch_factory, message
):
    signature = (
        f"local:model@rev:endpoint={'e' * 64}:dims=1024:"
        "prep=kg-entity-v1:max_chars=8192:batch=64"
    )

    def backend(texts):
        return batch_factory(texts, signature)

    session = CollectionEmbeddingSession(
        expected_model_signature=signature,
        backend=backend,
    )

    with pytest.raises(ValueError, match=message):
        session.embed(("Atlas",))


def test_embedding_session_rejects_missing_or_duplicate_provider_indices():
    def backend(texts):
        return SignedEmbeddingBatch(
            vectors=tuple(tuple(_unit_vector(1.0)) for _ in texts),
            text_hashes=tuple(embedding_text_hash(texts[0]) for _ in texts),
            indices=tuple(0 for _ in texts),
            model_signature=_EMBEDDING_SIGNATURE,
        )

    session = CollectionEmbeddingSession(
        expected_model_signature=_EMBEDDING_SIGNATURE,
        backend=backend,
    )

    with pytest.raises(ValueError, match="indices|binding|order"):
        session.embed(("Atlas", "Zephyr"))


def test_embedding_session_binds_reversed_provider_indices_to_exact_texts():
    atlas_vector = tuple(_unit_vector(1.0, 0.0))
    zephyr_vector = tuple(_unit_vector(0.0, 1.0))

    def backend(texts):
        assert texts == ("Atlas", "Zephyr")
        return SignedEmbeddingBatch(
            vectors=(zephyr_vector, atlas_vector),
            text_hashes=(
                embedding_text_hash("Zephyr"),
                embedding_text_hash("Atlas"),
            ),
            indices=(1, 0),
            model_signature=_EMBEDDING_SIGNATURE,
        )

    session = CollectionEmbeddingSession(
        expected_model_signature=_EMBEDDING_SIGNATURE,
        backend=backend,
    )

    result = session.embed(("Zephyr", "Atlas"))

    assert tuple(item.text for item in result) == ("Zephyr", "Atlas")
    assert result[0].vector == zephyr_vector
    assert result[1].vector == atlas_vector


def test_embedding_session_rejects_overlong_text_without_truncation_or_provider_call():
    calls = []

    def backend(texts):
        calls.append(texts)
        return SignedEmbeddingBatch(
            vectors=(tuple(_unit_vector(1.0)),),
            text_hashes=(embedding_text_hash(texts[0]),),
            indices=(0,),
            model_signature=_EMBEDDING_SIGNATURE,
        )

    session = CollectionEmbeddingSession(
        expected_model_signature=_EMBEDDING_SIGNATURE,
        backend=backend,
    )
    boundary = "x" * 8_192
    accepted = session.embed((boundary,))
    assert calls == [(boundary,)]
    assert accepted[0].text == boundary
    assert accepted[0].input_hash == embedding_text_hash(boundary)

    with pytest.raises(ValueError, match="maximum|8192|long"):
        session.embed(("x" * 8_193,))
    assert calls == [(boundary,)]


def test_embedding_session_batches_deterministically_and_fails_atomically():
    calls = []

    def backend(texts):
        calls.append(texts)
        if len(calls) == 2:
            raise RuntimeError("provider failed")
        return SignedEmbeddingBatch(
            vectors=tuple(tuple(_unit_vector(1.0)) for _ in texts),
            text_hashes=tuple(embedding_text_hash(text) for text in texts),
            indices=tuple(range(len(texts))),
            model_signature=_EMBEDDING_SIGNATURE.replace("batch=64", "batch=2"),
        )

    signature = _EMBEDDING_SIGNATURE.replace("batch=64", "batch=2")
    session = CollectionEmbeddingSession(
        expected_model_signature=signature,
        backend=backend,
        batch_size=2,
    )

    with pytest.raises(RuntimeError, match="provider failed"):
        session.embed(("Delta", "Alpha", "Charlie"))
    assert calls == [("Alpha", "Charlie"), ("Delta",)]

    calls.clear()
    with pytest.raises(RuntimeError, match="provider failed"):
        session.embed(("Delta", "Alpha", "Charlie"))
    assert calls[0] == ("Alpha", "Charlie")


def test_embedding_session_deduplicates_stably_and_records_input_hashes():
    session, backend = _session(
        {
            "Atlas": _unit_vector(1.0),
            "Zephyr": _unit_vector(0.0, 1.0),
        }
    )

    embedded = session.embed(("Zephyr", "Atlas", "Atlas"))

    assert backend.calls == [("Atlas", "Zephyr")]
    assert tuple(item.text for item in embedded) == ("Zephyr", "Atlas", "Atlas")
    assert embedded[1].input_hash == embedded[2].input_hash
    assert embedded[0].input_hash == embedding_text_hash("Zephyr")


def test_resolver_rejects_a_prewarmed_embedding_session():
    session, backend = _session({"Atlas": _unit_vector(1.0)})
    session.embed(("Atlas",))

    with pytest.raises(ValueError, match="fresh|prewarmed|cache"):
        resolve_collection_entities(
            _snapshot(),
            (_document_entity("a", "Atlas"),),
            _ontology(),
            embedding_session=session,
        )
    assert backend.calls == [("Atlas",)]


def test_embedding_candidate_decisions_remain_fanout_bounded():
    entities = tuple(
        _document_entity(index, f"Atlas variant {index}") for index in range(10, 110)
    )
    vectors = {
        entity.label: _unit_vector(1.0, (entity.entity_id % 10) / 1000)
        for entity in entities
    }
    session, _backend = _session(vectors)
    config = CollectionResolutionConfig(max_candidates_per_entity=3)
    result = resolve_collection_entities(
        _snapshot(config=config),
        entities,
        _ontology(),
        config=config,
        embedding_session=session,
    )

    embedding_decisions = tuple(
        decision
        for decision in result.decisions
        if decision.embedding_similarity is not None
        and "candidate_fanout_capped" not in decision.reason_codes
    )
    observed: dict[int, int] = {}
    for decision in embedding_decisions:
        observed[decision.left_entity_id] = observed.get(decision.left_entity_id, 0) + 1
        observed[decision.right_entity_id] = (
            observed.get(decision.right_entity_id, 0) + 1
        )
    assert max(observed.values(), default=0) <= 3
    assert len(result.decisions) <= len(entities) * 3
