import pytest

from apps.knowledge_graph.projection.serialization import canonical_projection_bytes
from apps.knowledge_graph.tests.test_projection_serialization import (
    ENTITY_A,
    ENTITY_B,
    _entity,
)


def test_duplicate_and_unsorted_record_arrays_are_rejected() -> None:
    first = _entity(ENTITY_A, 0.5)
    second = _entity(ENTITY_B, 0.25)
    with pytest.raises(ValueError, match="unique"):
        canonical_projection_bytes((first, first))
    with pytest.raises(ValueError, match="sorted"):
        canonical_projection_bytes((second, first))
