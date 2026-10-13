"""Real immutable image artifacts must obey the existing provider byte cap."""

from dataclasses import replace
from io import BytesIO
from random import Random

from PIL import Image
import pytest

from apps.ingestion.services.source_artifacts import persist_artifact
from apps.ingestion.tests.source_artifact_support import artifact_storage
from lib.embeddings import document_provider as provider
from lib.embeddings.document_contracts import build_budget
from lib.embeddings.provenance import input_digest, valid_provenance
from lib.embeddings.tests.test_document_provider import chunk, transport
from lib.embeddings.utils import EmbeddingContractError

pytestmark = pytest.mark.django_db


def image_chunk(side, seed):
    pixels = Random(seed).randbytes(side * side * 3)
    output = BytesIO()
    Image.frombytes("RGB", (side, side), pixels).save(output, format="PNG")
    ref = persist_artifact(
        output.getvalue(), member_path=None, container_sha256=None, using="default"
    )
    return replace(chunk("image evidence"), modality="image", media_artifact=ref)


def prepared_bytes(wire):
    # A lower bound on actual encoded request bytes, excluding JSON punctuation.
    values = (wire,) if isinstance(wire, str) else (wire["text"], wire["image"])
    return sum(len(value.encode("utf-8")) for value in values)


@pytest.mark.parametrize("sides", [(300,), (200, 200)])
def test_encoded_image_batch_over_original_byte_cap_never_dispatches(
    artifact_storage, transport, sides
):
    space, calls = transport
    original = tuple(image_chunk(side, index) for index, side in enumerate(sides))
    wires = tuple(provider.prepare_document(row, space=space)[0] for row in original)
    budget = build_budget(120000)
    assert budget.max_input_bytes == 262144
    assert len(original) <= budget.max_items
    assert sum(prepared_bytes(wire) for wire in wires) > budget.max_input_bytes
    if len(original) > 1:
        assert all(prepared_bytes(wire) < budget.max_input_bytes for wire in wires)
    with pytest.raises(EmbeddingContractError, match="document_batch_bytes"):
        provider.embed_document_batch(original, space=space, budget=budget)
    assert calls == []


@pytest.mark.parametrize("with_image", [False, True])
def test_under_cap_original_inputs_dispatch_once_with_exact_receipts(
    artifact_storage, transport, with_image
):
    space, calls = transport
    original = (image_chunk(64, 17), chunk("unchanged text")) if with_image else (
        chunk("unchanged text"),
    )
    wires = tuple(provider.prepare_document(row, space=space)[0] for row in original)
    budget = build_budget(120000)
    assert budget.max_input_bytes == 262144
    assert sum(prepared_bytes(wire) for wire in wires) < budget.max_input_bytes
    results = provider.embed_document_batch(original, space=space, budget=budget)
    assert len(calls) == 1
    assert calls[0]["input"] == list(wires)
    assert len(results) == len(original)
    for result, wire in zip(results, wires):
        assert valid_provenance(result.vector, result.provenance)
        assert result.provenance["prepared_input_sha256"] == input_digest(wire)
        assert result.provenance["dimensions"]["adaptation"] == "none"
        assert result.provenance["input_transformations"] == []
