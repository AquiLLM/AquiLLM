"""Exact bounded frontier pages and one invocation's nonrefundable ledger."""

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from threading import RLock

from apps.knowledge_graph.projection import records as r
from apps.knowledge_graph.retrieval.graph_query_contracts import GraphEvidenceRefV2
from lib.retrieval.turn_budget import TurnBudget

from .frontier_window_budget_v2 import WindowReservationMixin


def _count(value, maximum):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError("frontier count exceeds its exact bound")


def _key(value):
    if (
        type(value) is not str
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("frontier key must be lowercase SHA256")


class FrontierBudgetExceeded(TimeoutError):
    """Intentional bounded partiality, never a schema error."""


class FrontierBudgetV2(WindowReservationMixin):
    """The same instance covers all authority groups and acquisition rounds."""

    def __setattr__(self, name, value):
        frozen = name.startswith("max_") or name in {"turn_budget", "deadline", "clock"}
        if frozen and name in self.__dict__:
            raise AttributeError("frontier captured limits and deadline are immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        *,
        turn_budget: TurnBudget,
        deadline: float,
        max_identities: int,
        max_edges: int,
        max_source_rows: int,
        max_queries: int,
        max_wire_bytes: int,
        clock: Callable[[], float],
    ):
        if type(turn_budget) is not TurnBudget or not callable(clock):
            raise TypeError("frontier requires the original TurnBudget and clock")
        if type(deadline) is not float or not isfinite(deadline) or deadline <= 0:
            raise ValueError("frontier deadline must be finite and positive")
        for value, maximum in (
            (max_identities, 200),
            (max_edges, 1000),
            (max_source_rows, 5000),
            (max_queries, 64),
            (max_wire_bytes, 1048576),
        ):
            _count(value, maximum)
        self.turn_budget, self.deadline, self.clock = turn_budget, deadline, clock
        self.max_identities, self.max_edges = max_identities, max_edges
        self.max_source_rows, self.max_queries = max_source_rows, max_queries
        self.max_wire_bytes = max_wire_bytes
        self.identities = self.edges = self.source_rows = 0
        self.queries = self.wire_bytes = 0
        self._identities, self._edges, self._lock = set(), set(), RLock()
        self._uncertain, self._reserved_rows = False, 0
        self._authority_reservations = {}
        self._window = None

    def check_active(self):
        with self._lock:
            self.turn_budget.check_active()
            if self._uncertain:
                raise FrontierBudgetExceeded("frontier transport accounting uncertain")
            if self.clock() >= self.deadline or self.turn_budget.remaining_ms() <= 0:
                raise TimeoutError("frontier original deadline expired")

    def reserve_query(self):
        with self._lock:
            self.check_active()
            self._require_no_window()
            if (
                self.queries
                + sum(q for _, q, _ in self._authority_reservations.values())
                >= self.max_queries
            ):
                return False
            self.queries += 1
            return True

    def charge_page(self, *, scanned_rows: int, wire_bytes: int):
        # Charge even when late or invalid. Admission is a separate operation.
        for value in (scanned_rows, wire_bytes):
            if type(value) is not int or value < 0:
                raise ValueError("physical charges must be nonnegative exact integers")
        with self._lock:
            self.source_rows += scanned_rows
            self.wire_bytes += wire_bytes
            return (
                self.source_rows <= self.max_source_rows
                and self.wire_bytes <= self.max_wire_bytes
            )

    def admit_identity(self, key: str):
        _key(key)
        with self._lock:
            self.check_active()
            if key in self._identities:
                return True
            if self.identities >= self.max_identities:
                return False
            self._identities.add(key)
            self.identities += 1
            return True

    def admit_edge(self, key: tuple[str, str, str]):
        if (
            type(key) is not tuple
            or len(key) != 3
            or any(type(v) is not str or not v for v in key)
        ):
            raise ValueError("edge key must be an exact semantic triple")
        _key(key[0])
        _key(key[2])
        with self._lock:
            self.check_active()
            if key in self._edges:
                return True
            if self.edges >= self.max_edges:
                return False
            self._edges.add(key)
            self.edges += 1
            return True

    def reserve_authority(self, rows, queries):
        """Protect one session's final fresh proof in the shared ledger."""
        _count(rows, 5000)
        _count(queries, 64)
        with self._lock:
            self.check_active()
            self._require_no_window()
            held_rows = sum(r for r, _, _ in self._authority_reservations.values())
            held_queries = sum(q for _, q, _ in self._authority_reservations.values())
            if (
                self.source_rows + self._reserved_rows + held_rows + rows
                > self.max_source_rows
                or self.queries + held_queries + queries > self.max_queries
            ):
                raise FrontierBudgetExceeded("frontier final authority cannot fit")
            token = object()
            self._authority_reservations[token] = rows, queries, 0
            return token

    def reserve_authority_wire(self, token, wire_bytes):
        _count(wire_bytes, 1048576)
        with self._lock:
            self.check_active()
            self._require_no_window()
            rows, queries, previous = self._authority_reservations[token]
            if wire_bytes > self.available_wire_bytes() + previous:
                raise FrontierBudgetExceeded("frontier final wire authority cannot fit")
            self._authority_reservations[token] = rows, queries, wire_bytes

    def available_wire_bytes(self):
        with self._lock:
            self.check_active()
            return (
                self.max_wire_bytes
                - self.wire_bytes
                - sum(wire for _, _, wire in self._authority_reservations.values())
                - (self._window.wire if self._window is not None else 0)
            )

    def available_read_rows(self):
        with self._lock:
            self.check_active()
            return (
                self.max_source_rows
                - self.source_rows
                - self._reserved_rows
                - sum(r for r, _, _ in self._authority_reservations.values())
                - (self._window.rows if self._window is not None else 0)
            )

    def reserve_read(self, maximum_rows, *, authority=None):
        """Reserve the physical worst case before a serial datastore dispatch."""
        _count(maximum_rows, 5000)
        with self._lock:
            self.check_active()
            self._require_no_window()
            held_rows = sum(r for r, _, _ in self._authority_reservations.values())
            held_queries = sum(q for _, q, _ in self._authority_reservations.values())
            if authority is not None:
                rows, queries, wire = self._authority_reservations[authority]
                if rows < maximum_rows or queries < 1:
                    raise ValueError("frontier final authority reservation missing")
                held_rows -= maximum_rows
                held_queries -= 1
            if (
                self.source_rows + self._reserved_rows + held_rows + maximum_rows
                > self.max_source_rows
                or self.queries + held_queries >= self.max_queries
            ):
                raise FrontierBudgetExceeded("frontier physical work exhausted")
            if authority is not None:
                if queries == 1 and rows == maximum_rows:
                    del self._authority_reservations[authority]
                else:
                    self._authority_reservations[authority] = (
                        rows - maximum_rows,
                        queries - 1,
                        wire,
                    )
            self.queries += 1
            self._reserved_rows += maximum_rows

    def charge_read(self, *, reserved_rows, scanned_rows):
        with self._lock:
            if self._reserved_rows < reserved_rows:
                raise ValueError("frontier physical reservation is missing")
            self._reserved_rows -= reserved_rows
            return self.charge_page(scanned_rows=scanned_rows, wire_bytes=0)


@dataclass(frozen=True, slots=True)
class FrontierCursorV2:
    request_digest: str
    generation_key: str
    depth: int
    after_key: tuple[str, ...]

    def __post_init__(self):
        _key(self.request_digest)
        _key(self.generation_key)
        _count(self.depth, 2)
        if (
            type(self.after_key) is not tuple
            or not 1 <= len(self.after_key) <= 16
            or any(type(v) is not str or len(v) > 256 for v in self.after_key)
        ):
            raise ValueError("frontier cursor must contain bounded lexical keys")


@dataclass(frozen=True, slots=True)
class FrontierPageV2:
    nodes: tuple[r.ProjectedEntityV1, ...]
    memberships: tuple[r.AutomaticCanonicalMembershipV1, ...]
    relations: tuple[r.ProjectedPhysicalRelationV1, ...]
    relation_semantics: tuple[r.ProjectedRelationSemanticsV1, ...]
    documents: tuple[r.ProjectedDocumentMembershipV1, ...]
    chunks: tuple[r.ProjectedChunkMembershipV1, ...]
    relation_evidence: tuple[r.ProjectedRelationEvidenceV1, ...]
    mention_evidence: tuple[r.ProjectedEntityMentionEvidenceV1, ...]
    provenance: tuple[r.ProjectedArtifactProvenanceV1, ...]
    evidence_refs: tuple[GraphEvidenceRefV2, ...]
    next_cursor: FrontierCursorV2 | None
    scanned_rows: int
    complete: bool

    def __post_init__(self):
        kinds = (
            r.ProjectedEntityV1,
            r.AutomaticCanonicalMembershipV1,
            r.ProjectedPhysicalRelationV1,
            r.ProjectedRelationSemanticsV1,
            r.ProjectedDocumentMembershipV1,
            r.ProjectedChunkMembershipV1,
            r.ProjectedRelationEvidenceV1,
            r.ProjectedEntityMentionEvidenceV1,
            r.ProjectedArtifactProvenanceV1,
            GraphEvidenceRefV2,
        )
        names = tuple(self.__dataclass_fields__)[:10]
        for name, kind in zip(names, kinds, strict=True):
            value = getattr(self, name)
            if (
                type(value) is not tuple
                or len(value) > 5000
                or any(type(v) is not kind for v in value)
            ):
                raise TypeError("frontier page requires exact bounded source records")
        _count(self.scanned_rows, 5000)
        if type(self.complete) is not bool or self.complete != (
            self.next_cursor is None
        ):
            raise ValueError(
                "frontier sequence must have exactly one terminal discriminator"
            )
        if (
            self.next_cursor is not None
            and type(self.next_cursor) is not FrontierCursorV2
        ):
            raise TypeError("frontier next cursor must be exact")
