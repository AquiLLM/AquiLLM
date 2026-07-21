"""Thread-safe absolute-position forced-token replay state."""

from __future__ import annotations

from collections.abc import Sequence
from threading import Lock

from .decoding import BLANK_TOKEN_ID


class ReplayState:
    """Store one immutable transcript snapshot for a single decoder replay.

    The decoder prompt occupies absolute position zero, so generated transcript
    token ``n`` is requested at absolute position ``n + 1``.  All later
    positions return the RNN-T terminal blank.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._token_ids: tuple[int, ...] | None = None
        self._is_real = False

    def replace_real(self, token_ids: Sequence[int]) -> None:
        """Atomically replace replay state with a real transcription result."""

        snapshot = self._validated_snapshot(token_ids)
        with self._lock:
            self._token_ids = snapshot
            self._is_real = True

    def replace_profiling(self, token_ids: Sequence[int]) -> None:
        """Set profiling state unless a real transcription snapshot already exists."""

        snapshot = self._validated_snapshot(token_ids)
        with self._lock:
            if not self._is_real:
                self._token_ids = snapshot

    def forced_ids(self, positions: Sequence[int]) -> list[int]:
        """Return forced IDs for absolute decoder positions from one snapshot."""

        requested_positions = self._validated_positions(positions)
        with self._lock:
            snapshot = self._token_ids
        if snapshot is None:
            raise RuntimeError("replay state has not been initialized")

        return [
            snapshot[position - 1] if position <= len(snapshot) else BLANK_TOKEN_ID
            for position in requested_positions
        ]

    def reset(self) -> None:
        """Clear all request-local state after a completion or abort."""

        with self._lock:
            self._token_ids = None
            self._is_real = False

    @staticmethod
    def _validated_snapshot(token_ids: Sequence[int]) -> tuple[int, ...]:
        snapshot = tuple(token_ids)
        if any(
            isinstance(token_id, bool) or not isinstance(token_id, int)
            for token_id in snapshot
        ):
            raise TypeError("token IDs must be integers")
        return snapshot

    @staticmethod
    def _validated_positions(positions: Sequence[int]) -> tuple[int, ...]:
        requested_positions = tuple(positions)
        for position in requested_positions:
            if isinstance(position, bool) or not isinstance(position, int):
                raise TypeError("position must be a positive integer")
            if position <= 0:
                raise ValueError("position must be a positive integer")
        return requested_positions
