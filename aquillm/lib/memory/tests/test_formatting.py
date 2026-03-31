from lib.memory.formatting import format_memories_for_system


class _ProfileFactWithBrokenStr:
    fact = "My name is Jack"

    def __str__(self) -> str:
        raise AssertionError("format_memories_for_system should not call __str__ when .fact exists")


def test_format_memories_uses_fact_attr_without_stringifying_profile_object():
    rendered = format_memories_for_system(
        profile_facts=[_ProfileFactWithBrokenStr()],
        episodic_memories=[],
    )

    assert "My name is Jack" in rendered
