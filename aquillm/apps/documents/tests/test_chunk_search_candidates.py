"""Contracts for the reusable hybrid candidate snapshot."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from math import fsum
from types import SimpleNamespace
from uuid import UUID

import pytest
from django.test import override_settings

from apps.documents.services.chunk_search_candidates import (
    AuthorizedDocumentScope,
    CandidateScopeLimit,
    collect_hybrid_candidate_snapshot,
    freeze_authorized_document_scope,
)
from apps.knowledge_graph.retrieval import (
    GraphExpansionConfig,
    GraphExpansionSeed,
    get_graph_expansion_config,
)

_DOC_A = UUID("11111111-1111-4111-8111-111111111111")
_DOC_B = UUID("22222222-2222-4222-8222-222222222222")
_ALGORITHM_SIGNATURE = "a" * 64


def _config(**overrides: object) -> GraphExpansionConfig:
    values: dict[str, object] = {
        "rrf_k": 60,
        "max_seeds": 64,
        "max_scope_documents": 10_000,
        "max_scope_collections": 128,
        "max_candidates": 20,
        "algorithm_signature": _ALGORITHM_SIGNATURE,
    }
    values.update(overrides)
    return GraphExpansionConfig(**values)


class _Rows:
    def __init__(
        self,
        rows: list[object] | tuple[object, ...],
        error: Exception | None = None,
    ):
        self._rows = tuple(rows)
        self._error = error

    def __iter__(self):
        if self._error is not None:
            raise self._error
        return iter(self._rows)


class _VectorChain:
    def __init__(self, root: _QueryRoot):
        self._root = root

    def defer(self, *fields: str):
        self._root.deferred = fields
        return self

    def order_by(self, *fields: object):
        self._root.vector_order = fields
        return self

    def __getitem__(self, item: slice):
        self._root.limits["vector"] = item.stop
        return _Rows(self._root.vector_rows, self._root.vector_error)


class _TextChain:
    def __init__(self, root: _QueryRoot):
        self._root = root
        self._kind = "trigram"

    def annotate(self, **kwargs: object):
        return self

    def filter(self, *args: object, **kwargs: object):
        if args:
            self._kind = "exact"
        return self

    def order_by(self, *fields: object):
        return self

    def __getitem__(self, item: slice):
        self._root.limits[self._kind] = item.stop
        rows = (
            self._root.trigram_rows
            if self._kind == "trigram"
            else self._root.exact_rows
        )
        return _Rows(rows)


class _QueryRoot:
    def __init__(
        self,
        *,
        vector_rows: list[object],
        trigram_rows: list[object],
        exact_rows: list[object],
        vector_error: Exception | None = None,
    ):
        self.vector_rows = vector_rows
        self.trigram_rows = trigram_rows
        self.exact_rows = exact_rows
        self.vector_error = vector_error
        self.limits: dict[str, int | None] = {
            "vector": None,
            "trigram": None,
            "exact": None,
        }
        self.deferred: tuple[str, ...] = ()
        self.vector_order: tuple[object, ...] = ()

    def exclude(self, **kwargs: object):
        return _VectorChain(self)

    def filter(self, *args: object, **kwargs: object):
        return _TextChain(self).filter(*args, **kwargs)


class _Manager:
    def __init__(self, root: _QueryRoot):
        self.root = root
        self.scopes: list[object] = []

    def filter_by_documents(self, documents: object):
        self.scopes.append(documents)
        return self.root

    def none(self):
        return _Rows(())


class _Modality:
    TEXT = "text"


def _model(root: _QueryRoot):
    return type(
        "CandidateModel",
        (),
        {"Modality": _Modality, "objects": _Manager(root)},
    )


def _chunk(identifier: int, *, doc_id: UUID = _DOC_A):
    return SimpleNamespace(pk=identifier, doc_id=doc_id, content=f"chunk-{identifier}")


def _collect(root: _QueryRoot, *, graph_config: GraphExpansionConfig | None):
    model = _model(root)
    snapshot = collect_hybrid_candidate_snapshot(
        model,
        "Explain HSC-PDR2",
        3,
        (SimpleNamespace(id=_DOC_A, collection_id=7),),
        query_embedding=[0.1, 0.2],
        graph_config=graph_config,
        app_config_getter=lambda _label: SimpleNamespace(
            vector_top_k=30,
            trigram_top_k=30,
        ),
    )
    return model, snapshot


def test_snapshot_preserves_source_order_baseline_first_occurrence_and_rrf() -> None:
    c1, c2, c3, c4 = (_chunk(identifier) for identifier in range(1, 5))
    root = _QueryRoot(
        vector_rows=[c2, c1, c2],
        trigram_rows=[c3, c2],
        exact_rows=[c4, c2],
    )

    model, snapshot = _collect(root, graph_config=_config(max_seeds=3))

    assert snapshot.vector_chunk_ids == (2, 1, 2)
    assert snapshot.trigram_chunk_ids == (3, 2)
    assert snapshot.exact_chunk_ids == (4, 2)
    assert tuple(row.pk for row in snapshot.baseline_candidates) == (2, 1, 3, 4)
    assert snapshot.pre_dedupe_count == 7
    assert tuple(seed.chunk_id for seed in snapshot.graph_seeds) == (2, 3, 4)
    assert tuple(seed.rank for seed in snapshot.graph_seeds) == (1, 2, 3)
    assert all(type(seed) is GraphExpansionSeed for seed in snapshot.graph_seeds)
    assert fsum(seed.restart_weight for seed in snapshot.graph_seeds) == pytest.approx(
        1.0
    )
    expected_two = 1 / 61 + 1 / 62 + 1 / 62
    expected_three = 1 / 61
    denominator = expected_two + expected_three + 1 / 61
    assert snapshot.graph_seeds[0].restart_weight == pytest.approx(
        expected_two / denominator
    )
    assert model.objects.scopes == [snapshot.documents] * 3
    assert root.deferred == ("embedding",)


def test_vector_db_failure_keeps_trigram_and_exact_candidates() -> None:
    trigram = _chunk(2)
    exact = _chunk(3)
    root = _QueryRoot(
        vector_rows=[_chunk(1)],
        trigram_rows=[trigram],
        exact_rows=[exact],
        vector_error=RuntimeError("vector operator unavailable"),
    )

    _model_cls, snapshot = _collect(root, graph_config=_config())

    assert snapshot.vector_chunk_ids == ()
    assert snapshot.trigram_chunk_ids == (2,)
    assert snapshot.exact_chunk_ids == (3,)
    assert tuple(row.pk for row in snapshot.baseline_candidates) == (2, 3)
    assert "vector operator unavailable" in (snapshot.vector_error or "")
    assert tuple(seed.chunk_id for seed in snapshot.graph_seeds) == (2, 3)


def test_snapshot_is_frozen_and_source_limits_match_legacy_tuning() -> None:
    root = _QueryRoot(
        vector_rows=[_chunk(1)],
        trigram_rows=[_chunk(2)],
        exact_rows=[_chunk(3)],
    )
    _model_cls, snapshot = _collect(root, graph_config=None)

    with pytest.raises(FrozenInstanceError):
        snapshot.vector_chunk_ids = ()  # type: ignore[misc]
    assert snapshot.graph_seeds == ()
    assert root.limits == {"vector": 8, "trigram": 8, "exact": 8}


class _DocumentWithoutLazyCollectionAccess:
    def __init__(self, identifier: UUID, collection_id: int):
        self.id = identifier
        self.collection_id = collection_id

    @property
    def collection(self):
        raise AssertionError(
            "scope freezing must not dereference a collection relation"
        )


def test_authorized_scope_comes_only_from_exact_document_scalars() -> None:
    doc_b = _DocumentWithoutLazyCollectionAccess(_DOC_B, 9)
    doc_a = _DocumentWithoutLazyCollectionAccess(_DOC_A, 7)

    scope = freeze_authorized_document_scope((doc_b, doc_a), _config())

    assert type(scope) is AuthorizedDocumentScope
    assert scope.documents == (doc_a, doc_b)
    assert scope.allowed_doc_ids == (_DOC_A, _DOC_B)
    assert scope.allowed_collection_ids == (7, 9)


@pytest.mark.parametrize(
    "documents",
    (
        (SimpleNamespace(id=str(_DOC_A), collection_id=7),),
        (SimpleNamespace(id=_DOC_A, collection_id=True),),
        (_DOC_A,),
        (
            SimpleNamespace(id=_DOC_A, collection_id=7),
            SimpleNamespace(id=_DOC_A, collection_id=7),
        ),
    ),
)
def test_authorized_scope_rejects_raw_malformed_or_duplicate_references(
    documents: tuple[object, ...],
) -> None:
    with pytest.raises(ValueError):
        freeze_authorized_document_scope(documents, _config())


def test_authorized_scope_checks_caps_before_constructing_request_tuples() -> None:
    observed = 0

    def documents():
        nonlocal observed
        for index in range(3):
            observed += 1
            yield SimpleNamespace(
                id=UUID(int=index + 1),
                collection_id=index + 1,
            )

    with pytest.raises(CandidateScopeLimit):
        freeze_authorized_document_scope(
            documents(),
            _config(max_scope_documents=2, max_scope_collections=2),
        )

    assert observed == 3


def test_exact_term_helpers_remain_compatible_for_conversation_search() -> None:
    from apps.chat.services import conversation_search
    from apps.documents.services import chunk_search

    assert chunk_search._salient_exact_terms is conversation_search._salient_exact_terms
    assert chunk_search._exact_term_query is conversation_search._exact_term_query


@override_settings(
    KG_OVERLAY_RRF_K=17,
    KG_OVERLAY_MAX_SEEDS=11,
    KG_OVERLAY_MAX_SCOPE_DOCUMENTS=19,
    KG_OVERLAY_MAX_SCOPE_COLLECTIONS=7,
    KG_OVERLAY_MAX_CANDIDATES=5,
)
def test_public_config_accessor_returns_validated_signature_snapshot() -> None:
    config = get_graph_expansion_config()

    assert type(config) is GraphExpansionConfig
    assert (
        config.rrf_k,
        config.max_seeds,
        config.max_scope_documents,
        config.max_scope_collections,
        config.max_candidates,
    ) == (17, 11, 19, 7, 5)
    assert len(config.algorithm_signature) == 64
    assert config.algorithm_signature != _ALGORITHM_SIGNATURE


@override_settings(KG_OVERLAY_MAX_EDGES=1_001)
def test_public_config_accessor_validates_all_task15_settings() -> None:
    with pytest.raises(ValueError, match="max_edges"):
        get_graph_expansion_config()
