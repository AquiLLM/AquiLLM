"""Small queryset doubles for retrieval diagnostic contract tests."""

from unittest.mock import MagicMock


class _Modality:
    TEXT = "text"


class _SliceList:
    def __init__(self, rows, count_val: int = 0):
        self._rows = rows
        self._count_val = count_val

    def __getitem__(self, s):
        return list(self._rows)

    def count(self):
        return self._count_val


class _VecChain:
    def __init__(self, rows, count_val: int = 0):
        self._rows = rows
        self._count_val = count_val

    def defer(self, *fields: str):
        return self

    def order_by(self, *a):
        return _SliceList(self._rows, self._count_val)

    def count(self):
        return self._count_val


class _ExcludeChain:
    """Represents the result of .exclude(embedding__isnull=True)."""

    def __init__(self, rows, count_val: int = 0):
        self._rows = rows
        self._count_val = count_val

    def defer(self, *fields: str):
        return self

    def order_by(self, *a):
        return _SliceList(self._rows, self._count_val)

    def count(self):
        return self._count_val


class _TriChain:
    def __init__(self, rows):
        self._rows = rows

    def annotate(self, **k):
        return self

    def filter(self, *args, **k):
        return self

    def order_by(self, *a):
        return _SliceList(self._rows)


class _QRoot:
    """Query root supporting vector, trigram, and count chains."""

    def __init__(self, vec_rows, tri_rows, count_val: int = 0):
        self._vec = vec_rows
        self._tri = tri_rows
        self._count_val = count_val

    def exclude(self, **k):
        return _ExcludeChain(self._vec, self._count_val)

    def filter(self, *args, **k):
        return _TriChain(self._tri)

    def count(self):
        return self._count_val


class _ModelCls:
    Modality = _Modality
    objects = MagicMock()


def _make_cfg():
    cfg = MagicMock()
    cfg.vector_top_k = 30
    cfg.trigram_top_k = 30
    return cfg
