"""Shared CPU-importable tensor contracts. Validation never reads device data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def _positive_int(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class KVSpec:
    dtype: str
    head_dim: int
    num_q_heads: int
    num_kv_heads: int
    block_size: int
    key_packed_size: int
    value_data_bytes: int

    def __post_init__(self):
        if self.dtype != "turboquant_k8v4":
            raise ValueError("only turboquant_k8v4 is supported")
        for name in ("head_dim", "num_q_heads", "num_kv_heads", "block_size"):
            _positive_int(name, getattr(self, name))
        if self.num_q_heads % self.num_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if self.head_dim % 2 or self.key_packed_size != self.head_dim:
            raise ValueError("k8v4 requires one key byte per dimension and an even head dimension")
        if self.value_data_bytes != self.head_dim // 2:
            raise ValueError("k8v4 requires one value byte per pair of dimensions")


@dataclass(frozen=True)
class SplitPlan:
    max_splits: int
    buckets: tuple[tuple[int, int], ...]

    def __post_init__(self):
        _positive_int("max_splits", self.max_splits)
        previous = -1
        for upper, splits in self.buckets:
            if type(upper) is not int or upper < 0 or upper <= previous:
                raise ValueError("split bucket bounds must be increasing nonnegative integers")
            _positive_int("active splits", splits)
            if splits > self.max_splits:
                raise ValueError("active splits exceed max_splits")
            previous = upper


@dataclass
class VerifyBatch:
    q: Tensor
    kv_cache: Tensor
    block_table: Tensor
    seq_lens: Tensor
    raw_k: Tensor
    raw_v: Tensor
    scale: float
    spec: KVSpec


@dataclass
class AttentionState:
    output: Tensor
    lse: Tensor  # FP32 natural logarithm, [Q,Hq].


@dataclass(frozen=True)
class RouteDecision:
    eligible: bool
    reason: str
