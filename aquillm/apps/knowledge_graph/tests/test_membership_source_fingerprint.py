"""Bounded source proof contract; database differential tests live in support."""
from contextlib import contextmanager
import importlib

import pytest


def reader():
    return importlib.import_module(
        "apps.knowledge_graph.projection.membership_source_fingerprint"
    )


class Database:
    vendor = "postgresql"

    def __init__(self, rows):
        self.rows, self.calls = rows, []

    @contextmanager
    def cursor(self):
        self.calls.append("cursor")
        yield self

    def execute(self, sql, params):
        self.calls.append((sql, params))

    def fetchmany(self, size):
        self.calls.append(size)
        return self.rows[:size]


def run(monkeypatch, rows, **kwargs):
    module = reader()
    db = Database(rows)
    monkeypatch.setattr(module, "connections", {"source": db})
    result = module.load_membership_source_fingerprints(
        collection_ids=kwargs.pop("collection_ids", (7,)), using="source",
        check=kwargs.pop("check", lambda: None), **kwargs,
    )
    return result, db


def test_reader_exists():
    assert reader().SOURCE_FINGERPRINT_FORMAT == "kg-membership-source-sql-v1"


def test_one_bounded_statement_returns_immutable_duplicate_evidence(monkeypatch):
    result, db = run(monkeypatch, [(7, 1, 2, True, "a" * 64, "read committed")])
    assert result[7].duplicate_links is True
    assert result[7].source_rows == 2
    assert result[7].input_sha256 == "a" * 64
    assert len([call for call in db.calls if isinstance(call, tuple)]) == 1
    assert db.calls[-1] == 2
    assert db.calls[1][1] == [[7], [7]]
    with pytest.raises(AttributeError):
        result[7].entity_count = 2


@pytest.mark.parametrize("ids", [(True,), (0,), (-1,), (2**63,), (8, 7),
                                     (7, 7), [7], tuple(range(1, 130))])
def test_invalid_requested_sets_reject_before_database(monkeypatch, ids):
    with pytest.raises(ValueError, match="collection_ids"):
        run(monkeypatch, [], collection_ids=ids)


def test_empty_scope_avoids_database(monkeypatch):
    result, db = run(monkeypatch, [], collection_ids=())
    assert result == {} and db.calls == []


@pytest.mark.parametrize("rows", [[], [(8, 0, 0, False, "a" * 64, "read committed")],
    [(7, True, 1, False, "a" * 64, "read committed")],
    [(7, 2, 1, False, "a" * 64, "read committed")],
    [(7, 1, 2, False, "a" * 64, "read committed")],
    [(7, 0, 1, True, "a" * 64, "read committed")],
    [(7, 1, 1, True, "a" * 64, "read committed")],
    [(7, 0, 0, False, "A" * 64, "read committed")],
    [(7, 0, 0, False, "a" * 64, "repeatable read")],
    [(7, 0, 0, False, "a" * 64, "read committed")] * 2])
def test_invalid_or_stale_results_reject(monkeypatch, rows):
    with pytest.raises(ValueError):
        run(monkeypatch, rows)


def test_certification_requires_repeatable_read(monkeypatch):
    run(monkeypatch, [(7, 0, 0, False, "a" * 64, "repeatable read")],
        require_fresh=False)
    with pytest.raises(ValueError, match="isolation"):
        run(monkeypatch, [(7, 0, 0, False, "a" * 64, "read committed")],
            require_fresh=False)


@pytest.mark.parametrize("expiry", range(1, 10))
def test_original_check_expires_at_each_boundary(monkeypatch, expiry):
    calls = 0

    def check():
        nonlocal calls
        calls += 1
        if calls == expiry:
            raise TimeoutError("original deadline")

    with pytest.raises(TimeoutError, match="original deadline"):
        run(monkeypatch, [(7, 0, 0, False, "a" * 64, "read committed")], check=check)
