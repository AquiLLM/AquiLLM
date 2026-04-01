"""Adaptive tool-call budget policy primitives for LLMInterface.spin()."""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from json import dumps
from os import getenv
from typing import Any, Optional

import structlog

logger = structlog.stdlib.get_logger(__name__)


def _normalize_tool_name(name: str | None) -> str:
    return str(name or "").strip().lower()


def _parse_positive_int(raw: str | None, default: int) -> int:
    try:
        value = int(raw or str(default))
    except Exception:
        value = default
    return max(1, value)


def parse_csv_positive_int_map(raw: str | None, *, setting_name: str) -> dict[str, int]:
    """Parse `tool_name:value` CSV settings with warning-on-invalid behavior."""
    if raw is None or not str(raw).strip():
        return {}
    parsed: dict[str, int] = {}
    for token in str(raw).split(","):
        item = token.strip()
        if not item:
            continue
        if ":" not in item:
            logger.warning("Ignoring invalid %s entry without ':'", setting_name)
            continue
        key_raw, value_raw = item.split(":", 1)
        key = _normalize_tool_name(key_raw)
        if not key:
            logger.warning("Ignoring invalid %s entry with empty tool name", setting_name)
            continue
        try:
            value = int(value_raw.strip())
        except Exception:
            logger.warning("Ignoring invalid %s entry for tool=%s (value not int)", setting_name, key)
            continue
        if value <= 0:
            logger.warning("Ignoring invalid %s entry for tool=%s (value must be >0)", setting_name, key)
            continue
        parsed[key] = value
    return parsed


@dataclass(frozen=True)
class ToolBudgetConfig:
    per_tool_limit_default: int
    per_tool_limits: dict[str, int]
    repeat_signature_break_threshold: int
    no_progress_break_threshold: int
    budget_units_per_turn: Optional[int]
    tool_cost_weights: dict[str, int]

    @classmethod
    def from_env(cls, *, max_func_calls: int) -> "ToolBudgetConfig":
        per_tool_limit_default = _parse_positive_int(getenv("LLM_MAX_CALLS_PER_TOOL_NAME"), 2)
        per_tool_limits = parse_csv_positive_int_map(
            getenv("LLM_TOOL_CALL_LIMITS"),
            setting_name="LLM_TOOL_CALL_LIMITS",
        )
        repeat_signature_break_threshold = _parse_positive_int(
            getenv("LLM_REPEAT_TOOL_BREAK_THRESHOLD"),
            3,
        )
        no_progress_break_threshold = _parse_positive_int(
            getenv("LLM_TOOL_NO_PROGRESS_BREAK_THRESHOLD"),
            2,
        )

        budget_units_per_turn: Optional[int]
        raw_budget_units = getenv("LLM_TOOL_BUDGET_UNITS_PER_TURN")
        if raw_budget_units is None or not str(raw_budget_units).strip():
            budget_units_per_turn = None
        else:
            try:
                parsed_budget_units = int(raw_budget_units)
            except Exception:
                logger.warning(
                    "Invalid LLM_TOOL_BUDGET_UNITS_PER_TURN=%r; disabling weighted budget",
                    raw_budget_units,
                )
                budget_units_per_turn = None
            else:
                budget_units_per_turn = max(1, parsed_budget_units)
        tool_cost_weights = parse_csv_positive_int_map(
            getenv("LLM_TOOL_COST_WEIGHTS"),
            setting_name="LLM_TOOL_COST_WEIGHTS",
        )
        if budget_units_per_turn is None:
            tool_cost_weights = {}

        return cls(
            per_tool_limit_default=per_tool_limit_default,
            per_tool_limits=per_tool_limits,
            repeat_signature_break_threshold=repeat_signature_break_threshold,
            no_progress_break_threshold=no_progress_break_threshold,
            budget_units_per_turn=budget_units_per_turn,
            tool_cost_weights=tool_cost_weights,
        )

    def resolve_per_tool_limit(self, tool_name: str) -> int:
        key = _normalize_tool_name(tool_name)
        return self.per_tool_limits.get(key, self.per_tool_limit_default)

    def resolve_tool_cost_weight(self, tool_name: str) -> int:
        key = _normalize_tool_name(tool_name)
        return self.tool_cost_weights.get(key, 1)


@dataclass(frozen=True)
class ToolCallObservation:
    tool_name: str
    signature: str
    latest_result_dict: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class ToolBudgetDecision:
    should_continue: bool
    stop_reason: Optional[str] = None


@dataclass
class _ToolBudgetState:
    tool_name_call_counts: dict[str, int] = field(default_factory=dict)
    budget_units_consumed: int = 0
    repeated_signature_count: int = 0
    no_progress_streak: int = 0
    previous_signature: Optional[str] = None
    last_result_hash_by_tool: dict[str, str] = field(default_factory=dict)


class ToolBudgetPolicy:
    """Mutable policy state for a single conversation turn's tool loop."""

    def __init__(self, config: ToolBudgetConfig):
        self.config = config
        self.state = _ToolBudgetState()
        self.stop_reason: Optional[str] = None

    def observe_tool_call(self, observation: ToolCallObservation) -> ToolBudgetDecision:
        tool_name = _normalize_tool_name(observation.tool_name)
        signature = observation.signature
        is_repeat_signature = self.state.previous_signature == signature
        self._update_repeat_signature(signature)
        no_progress = self._detect_no_progress(tool_name, observation, is_repeat_signature=is_repeat_signature)
        self._update_no_progress_streak(no_progress)

        next_count = self.state.tool_name_call_counts.get(tool_name, 0) + 1
        per_tool_limit = self.config.resolve_per_tool_limit(tool_name)
        if next_count >= per_tool_limit:
            return self._break("per_tool_limit_reached")
        if self.state.repeated_signature_count >= self.config.repeat_signature_break_threshold:
            return self._break("repeat_signature_break")
        if self.state.no_progress_streak >= self.config.no_progress_break_threshold:
            return self._break("no_progress_break")

        if self.config.budget_units_per_turn is not None:
            weight = self.config.resolve_tool_cost_weight(tool_name)
            would_consume = self.state.budget_units_consumed + weight
            if would_consume > self.config.budget_units_per_turn:
                return self._break("budget_units_exhausted")
            self.state.budget_units_consumed = would_consume

        self.state.tool_name_call_counts[tool_name] = next_count
        return ToolBudgetDecision(should_continue=True)

    def summary(self) -> dict[str, Any]:
        return {
            "tool_counts": dict(self.state.tool_name_call_counts),
            "budget_units_consumed": self.state.budget_units_consumed,
            "budget_units_per_turn": self.config.budget_units_per_turn,
            "stop_reason": self.stop_reason,
            "no_progress_streak": self.state.no_progress_streak,
            "repeated_signature_count": self.state.repeated_signature_count,
        }

    def _break(self, reason: str) -> ToolBudgetDecision:
        self.stop_reason = reason
        return ToolBudgetDecision(should_continue=False, stop_reason=reason)

    def _update_repeat_signature(self, signature: str) -> None:
        if self.state.previous_signature == signature:
            self.state.repeated_signature_count += 1
        else:
            self.state.repeated_signature_count = 0
        self.state.previous_signature = signature

    def _update_no_progress_streak(self, no_progress: bool) -> None:
        if no_progress:
            self.state.no_progress_streak += 1
        else:
            self.state.no_progress_streak = 0

    def _detect_no_progress(
        self,
        tool_name: str,
        observation: ToolCallObservation,
        *,
        is_repeat_signature: bool,
    ) -> bool:
        result_dict = observation.latest_result_dict if isinstance(observation.latest_result_dict, dict) else None
        has_exception = bool(result_dict and str(result_dict.get("exception", "")).strip())
        if has_exception:
            return True

        result_hash = _hash_result_dict(result_dict)
        previous_hash = self.state.last_result_hash_by_tool.get(tool_name)
        if result_hash is not None:
            self.state.last_result_hash_by_tool[tool_name] = result_hash
        same_result_payload = result_hash is not None and previous_hash == result_hash
        if same_result_payload:
            return True

        if is_repeat_signature and result_hash is None:
            return True

        return False


def _hash_result_dict(result_dict: Optional[dict[str, Any]]) -> Optional[str]:
    if not isinstance(result_dict, dict):
        return None
    try:
        canonical = dumps(result_dict, sort_keys=True, ensure_ascii=True, default=str)
    except Exception:
        return None
    return sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "ToolBudgetConfig",
    "ToolBudgetDecision",
    "ToolBudgetPolicy",
    "ToolCallObservation",
    "parse_csv_positive_int_map",
]
