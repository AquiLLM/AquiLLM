"""Publication validates before opening a state cursor and preserves caller checks."""

import pytest

from apps.knowledge_graph.projection.membership_certificate_store import (
    publish_membership_certificate,
)


def test_invalid_payload_never_reaches_database():
    with pytest.raises(ValueError):
        publish_membership_certificate(
            payload={},
            signature="a" * 64,
            previous_signature=None,
            using="does-not-exist",
            check=lambda: None,
        )


def test_original_check_interrupts_before_database():
    class Deadline(Exception):
        pass

    def check():
        raise Deadline()

    with pytest.raises(Deadline):
        publish_membership_certificate(
            payload={},
            signature="a" * 64,
            previous_signature=None,
            using="does-not-exist",
            check=check,
        )
