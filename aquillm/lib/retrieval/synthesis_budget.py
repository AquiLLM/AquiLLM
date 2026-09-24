"""One frozen evidence capability, separate time limit, original work counters."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SynthesisLease:
    budget: object
    payload: str
    deadline: float
    max_calls: int
    max_output_tokens: int
    authority: object = None
    identities: tuple = ()
    calls: int = 0
    output_tokens: int = 0
    dispatches: list = field(default_factory=list)

    def check_active(self):
        self.budget.check_active()
        if self.budget._synthesis is not self or self.remaining_seconds() <= 0:
            raise ValueError("synthesis allowance exhausted")

    def remaining_seconds(self):
        return max(0.0, self.deadline - self.budget._clock())

    def can_publish(self):
        self.check_active()
        return True

    def reserve_text(self, count, *, kind):
        if kind != "tokenized" or type(count) is not int or count <= 0:
            raise ValueError("invalid synthesis tokenization")
        with self.budget._lock:
            self.check_active()
            if (
                self.budget._text[kind] + count
                > self.budget.limits.tokenized_codepoints
            ):
                return False
            self.budget._text[kind] += count
            return True

    def start_dispatch(self, output_tokens, kind="initial"):
        with self.budget._lock:
            self.check_active()
            if (
                self.calls >= self.max_calls
                or self.output_tokens + output_tokens > self.max_output_tokens
            ):
                raise ValueError("synthesis dispatch allowance exhausted")
            object.__setattr__(self, "calls", self.calls + 1)
            object.__setattr__(
                self, "output_tokens", self.output_tokens + output_tokens
            )
            self.dispatches.append({"kind": kind, "started": self.budget._clock()})
            return self.calls - 1

    def finish_dispatch(self, index):
        with self.budget._lock:
            item = self.dispatches[index]
            item["duration_ms"] = (self.budget._clock() - item["started"]) * 1000


def seal_synthesis(
    budget,
    payload,
    *,
    calls,
    output_tokens,
    timeout_seconds,
    authority=None,
    identities=(),
):
    """Called only after app-owned final authorization and exact payload assembly.

    Reentry may reuse this exact immutable input, never reset any allowance.
    Retrieval publication remains closed even while synthesis is active.
    """
    with budget._lock:
        budget.check_active()
        if budget._synthesis is not None:
            if (
                budget._synthesis.payload != payload
                or budget._synthesis.authority is not authority
                or budget._synthesis.identities != identities
            ):
                raise ValueError("frozen evidence changed")
            return budget._synthesis
        if (
            not budget._open()
            or not payload
            or min(calls, output_tokens, timeout_seconds) <= 0
        ):
            raise ValueError("evidence was not frozen within retrieval deadline")
        lease = SynthesisLease(
            budget,
            payload,
            budget._clock() + timeout_seconds,
            calls,
            output_tokens,
            authority,
            tuple(identities),
        )
        budget._synthesis = lease
        budget._closed_reason = "sealed"
        from lib.evidence_observation import publish

        publish(
            "synthesis_sealed",
            {
                "ledger_id": str(id(budget)),
                "max_calls": calls,
                "max_output_tokens": output_tokens,
                "timeout_seconds": timeout_seconds,
                "remaining_retrieval_ms": max(
                    0, (budget._deadline - budget._clock()) * 1000
                ),
            },
        )
        return lease
