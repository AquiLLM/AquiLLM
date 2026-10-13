"""Validation of immutable pilot limits; no runtime budget state."""


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
