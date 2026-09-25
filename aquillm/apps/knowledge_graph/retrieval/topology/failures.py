"""Provider-neutral typed topology failure boundary."""

from .contracts import TopologyFailureReason


class TopologyLoadError(RuntimeError):
    """Expose one fixed topology reason without leaking a raw backend error."""

    def __init__(self, reason: TopologyFailureReason):
        if type(reason) is not TopologyFailureReason:
            raise TypeError("reason must be an exact TopologyFailureReason")
        self.reason = reason
        super().__init__(reason.value)


class TopologyResultCapError(RuntimeError):
    """Signal an attested request-local provider result limit."""

    def __init__(self) -> None:
        super().__init__("topology result cap reached")


__all__ = ["TopologyLoadError", "TopologyResultCapError"]
