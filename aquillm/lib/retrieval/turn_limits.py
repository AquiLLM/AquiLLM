"""Validation of immutable pilot limits; no runtime budget state."""

PILOT_MAX = {
    "actions": 3,
    "unique_sources": 45,
    "materialized_codepoints": 250_000,
    "tokenized_codepoints": 1_000_000,
    "acquisition_pairs": 90,
    "final_pairs": 45,
    "in_flight_pairs": 6,
    "planner_calls": 2,
    "planner_call_ms": 5_000,
    "planner_output_tokens": 1024,
    "retrieval_ms": 30_000,
    "final_scoring_ms": 3_000,
}


def validate_limits(self, pilot_max) -> None:
    for name in self.__dataclass_fields__:
        value = getattr(self, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= pilot_max[name]
        ):
            raise ValueError(f"{name} must be within the pilot allowance")
    if self.acquisition_pairs + self.final_pairs > 135:
        raise ValueError("overall pair allowance exceeds 135")
    if self.final_scoring_ms > self.retrieval_ms:
        raise ValueError("final scoring reserve exceeds retrieval deadline")
