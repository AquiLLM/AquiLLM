"""Provider-neutral typed topology failure boundary."""

from .contracts import TopologyFailureReason


class TopologyLoadError(RuntimeError):
    """Expose one fixed topology reason without leaking a raw backend error."""

    def __init__(self, reason: TopologyFailureReason):
        if type(reason) is not TopologyFailureReason:
            raise TypeError("reason must be an exact TopologyFailureReason")
        self.reason = reason
        super().__init__(reason.value)


__all__ = ["TopologyLoadError"]
