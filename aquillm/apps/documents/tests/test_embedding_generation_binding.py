import pytest
from apps.documents.models import TextChunk
from apps.documents.services import embedding_index_store as store
from apps.documents.tests.test_embedding_generation import legacy as legacy, stage
from lib.embeddings.tests.test_document_provider import transport as transport
from lib.embeddings.document_provider import embed_document_batch
from lib.embeddings.document_contracts import build_budget

pytestmark = pytest.mark.django_db(transaction=True)


def test_response_from_other_engine_cannot_be_staged_as_same_model_space(
    legacy, transport
):
    from lib.embeddings.tests.test_document_contracts import profile
    from lib.embeddings.document_provider import (
        register_verified_profile,
        engine_manifest_for,
    )
    from lib.embeddings.utils import EmbeddingContractError

    _, lease, manifest, _ = stage(legacy, transport)
    other = profile(compute_precision="bf16")
    register_verified_profile(
        other,
        endpoint="http://fake-models:8000/v1",
        deployment_artifact_sha256="c" * 64,
        engine_manifest=engine_manifest_for(other),
        query_template="{text}",
        document_template="{text}",
    )
    results = embed_document_batch(
        (store.source_chunk(TextChunk.objects.get()),),
        space=other,
        budget=build_budget(),
    )
    with pytest.raises(EmbeddingContractError, match="serving_binding_mismatch"):
        store.stage_embedding_batch(
            lease,
            chunk_ids=(manifest[0]["chunk_id"],),
            results=results,
            input_sha256=(manifest[0]["prepared_input_sha256"],),
            using="default",
        )


def test_query_cache_restoration_requires_separate_serving_binding(transport):
    from lib.embeddings import document_provider as provider
    from lib.embeddings.document_binding import (
        serialize_query_observation,
        restore_query_observation,
    )
    from lib.embeddings.document_contracts import QueryEmbeddingObservationV1
    from lib.embeddings.utils import EmbeddingContractError
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits

    space, _ = transport
    observed = provider.observe_query_embedding(
        "query", space=space, budget=TurnBudget(TurnLimits())
    )
    cached = serialize_query_observation(observed, space=space)
    restored = restore_query_observation(cached, space=space)
    assert restored.receipt == observed.receipt
    assert restored.receipt["observed_identity"]["revision"] is None
    cached["serving_binding"]["space_key"] = "f" * 64
    with pytest.raises(EmbeddingContractError, match="serving_binding_mismatch"):
        restore_query_observation(cached, space=space)
    unbound = QueryEmbeddingObservationV1(
        observed.vector,
        observed.receipt,
        observed.prepared_input_sha256,
        observed.compatibility_key,
    )
    with pytest.raises(EmbeddingContractError, match="serving_binding_mismatch"):
        serialize_query_observation(unbound, space=space)


@pytest.mark.parametrize(
    "change", ["extra", "bool", "nan", "engine", "role", "modality", "media", "key"]
)
def test_query_cache_rejects_changed_envelope_and_noncanonical_vectors(
    transport, change
):
    from lib.embeddings import document_provider as provider
    from lib.embeddings.document_binding import (
        serialize_query_observation,
        restore_query_observation,
    )
    from lib.embeddings.utils import EmbeddingContractError
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits

    space, _ = transport
    observation = provider.observe_query_embedding(
        "query", space=space, budget=TurnBudget(TurnLimits())
    )
    cached = serialize_query_observation(observation, space=space)
    if change == "extra":
        cached["extra"] = "untrusted"
    elif change in ("bool", "nan"):
        cached["vector"][0] = True if change == "bool" else float("nan")
    elif change == "key":
        cached["compatibility_key"] = "f" * 64
    else:
        field = {
            "engine": "verification_sha256",
            "role": "role",
            "modality": "modality",
            "media": "media_sha256",
        }[change]
        cached["serving_binding"][field] = "foreign"
    with pytest.raises(EmbeddingContractError):
        restore_query_observation(cached, space=space)
