from __future__ import annotations

from contextlib import contextmanager

import numpy as np
from pgvector.django import VectorField


@contextmanager
def django_safe_vector_values(instance):
    """Temporarily adapt exact vectors for Django and pgvector validation."""

    restored: dict[str, object] = {}
    for field in instance._meta.local_fields:
        value = getattr(instance, field.attname)
        if isinstance(field, VectorField) and isinstance(value, (np.ndarray, tuple)):
            restored[field.attname] = value
            setattr(
                instance,
                field.attname,
                value.tolist() if isinstance(value, np.ndarray) else list(value),
            )
    try:
        yield
    finally:
        for field_name, value in restored.items():
            setattr(instance, field_name, value)


def immutable_field_value(field, value: object) -> object:
    """Return one exact comparison form for a pgvector model field value."""

    if not isinstance(field, VectorField) or value is None:
        return value
    try:
        stored = np.asarray(value, dtype=">f4")
    except (OverflowError, TypeError, ValueError):
        return ("invalid-vector", type(value).__module__, type(value).__qualname__)
    shape = tuple(int(dimension) for dimension in stored.shape)
    if stored.size > field.dimensions:
        return ("float32-vector", shape, int(stored.size))
    return ("float32-vector", shape, tuple(stored.reshape(-1).tolist()))
