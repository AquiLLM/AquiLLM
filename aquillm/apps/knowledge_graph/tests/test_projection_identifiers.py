from uuid import UUID

import pytest

from apps.knowledge_graph.projection import (
    HmacSha256ProjectionIdentifierCodec,
    OpaqueProjectionKey,
    ProjectionIdentifierDomain,
)

TEST_KEY = b"task21-projection-test-key"
EXPECTED_DOMAINS = {
    "collection",
    "artifact",
    "entity",
    "relation",
    "evidence",
    "relation_mention",
    "entity_mention",
    "entity_mapping",
    "document",
    "chunk",
    "canonical_link_decision",
    "automatic_canonical_identity",
}


class _HostileDigest:
    def __len__(self) -> int:
        return 64

    def __iter__(self):
        return iter("0" * 64)


class _DigestSubclass(str):
    pass


@pytest.fixture
def codec() -> HmacSha256ProjectionIdentifierCodec:
    return HmacSha256ProjectionIdentifierCodec(
        key=TEST_KEY,
        key_version="test-key-v1",
    )


@pytest.mark.parametrize(
    "value",
    [_HostileDigest(), _DigestSubclass("0" * 64)],
)
def test_opaque_projection_key_requires_exact_builtin_string(value: object) -> None:
    with pytest.raises(TypeError, match="built-in str"):
        OpaqueProjectionKey(
            domain=ProjectionIdentifierDomain.ENTITY,
            value=value,  # type: ignore[arg-type]
        )


def test_domain_enum_is_closed_and_every_domain_encodes(
    codec: HmacSha256ProjectionIdentifierCodec,
) -> None:
    assert {domain.value for domain in ProjectionIdentifierDomain} == EXPECTED_DOMAINS

    identifiers = []
    for domain in ProjectionIdentifierDomain:
        generation = (
            None
            if domain is ProjectionIdentifierDomain.AUTOMATIC_CANONICAL_IDENTITY
            else "generation-a"
        )
        identifier = codec.encode(domain, generation=generation, source=11)
        assert isinstance(identifier, OpaqueProjectionKey)
        assert type(identifier.domain) is ProjectionIdentifierDomain
        assert identifier.domain is domain
        assert len(identifier.value) == 64
        assert identifier.value == identifier.value.lower()
        identifiers.append(identifier)

    assert len(set(identifiers)) == len(EXPECTED_DOMAINS)


def test_key_version_separates_identifiers() -> None:
    first = HmacSha256ProjectionIdentifierCodec(
        key=TEST_KEY,
        key_version="test-key-v1",
    )
    second = HmacSha256ProjectionIdentifierCodec(
        key=TEST_KEY,
        key_version="test-key-v2",
    )

    assert first.encode(
        ProjectionIdentifierDomain.ENTITY,
        generation="generation-a",
        source=11,
    ) != second.encode(
        ProjectionIdentifierDomain.ENTITY,
        generation="generation-a",
        source=11,
    )


@pytest.mark.parametrize(
    "domain",
    [
        domain
        for domain in ProjectionIdentifierDomain
        if domain is not ProjectionIdentifierDomain.AUTOMATIC_CANONICAL_IDENTITY
    ],
)
def test_generation_separates_generation_scoped_domains(
    codec: HmacSha256ProjectionIdentifierCodec,
    domain: ProjectionIdentifierDomain,
) -> None:
    assert codec.encode(domain, generation="generation-a", source=11) != codec.encode(
        domain,
        generation="generation-b",
        source=11,
    )


def test_automatic_membership_is_equal_across_generations(
    codec: HmacSha256ProjectionIdentifierCodec,
) -> None:
    domain = ProjectionIdentifierDomain.AUTOMATIC_CANONICAL_IDENTITY

    assert codec.encode(domain, source="canonical-a") == codec.encode(
        domain,
        generation="generation-a",
        source="canonical-a",
    )
    assert codec.encode(
        domain,
        generation="generation-a",
        source="canonical-a",
    ) == codec.encode(
        domain,
        generation="generation-b",
        source="canonical-a",
    )


def test_literal_compatibility_vector(
    codec: HmacSha256ProjectionIdentifierCodec,
) -> None:
    domain = ProjectionIdentifierDomain.AUTOMATIC_CANONICAL_IDENTITY

    assert codec.frame(domain, source="canonical-a").hex() == (
        "0000001070726f6a656374696f6e2d69642d7631"
        "0000000b746573742d6b65792d7631"
        "0000001c6175746f6d617469635f63616e6f6e6963616c5f6964656e74697479"
        "000000012d"
        "0000000475746638"
        "0000000b63616e6f6e6963616c2d61"
    )
    assert codec.encode(domain, source="canonical-a").value == (
        "88b2c4e9b12b4320d5f44bfbc0542c275ac2117197b16702a5da1a9aaea1a54c"
    )


def test_supported_source_types_have_unambiguous_encodings(
    codec: HmacSha256ProjectionIdentifierCodec,
) -> None:
    domain = ProjectionIdentifierDomain.ENTITY
    generation = "generation-a"
    canonical_uuid = "12345678-1234-5678-9234-567812345678"

    assert codec.encode(domain, generation=generation, source=11) != codec.encode(
        domain,
        generation=generation,
        source="11",
    )
    assert codec.encode(
        domain,
        generation=generation,
        source=UUID(canonical_uuid),
    ) == codec.encode(
        domain,
        generation=generation,
        source=canonical_uuid,
    )


@pytest.mark.parametrize("source", [-(2**63), 2**63 - 1])
def test_integer_sources_accept_signed_64_bit_boundaries(
    codec: HmacSha256ProjectionIdentifierCodec,
    source: int,
) -> None:
    identifier = codec.encode(
        ProjectionIdentifierDomain.ENTITY,
        generation="generation-a",
        source=source,
    )

    assert isinstance(identifier, OpaqueProjectionKey)


@pytest.mark.parametrize("source", [-(2**63) - 1, 2**63])
def test_integer_sources_reject_values_outside_signed_64_bit_range(
    codec: HmacSha256ProjectionIdentifierCodec,
    source: int,
) -> None:
    with pytest.raises(ValueError, match="signed 64-bit"):
        codec.encode(
            ProjectionIdentifierDomain.ENTITY,
            generation="generation-a",
            source=source,
        )


@pytest.mark.parametrize("source", [True, False, b"11", bytearray(b"11"), object()])
def test_ambiguous_or_arbitrary_sources_are_rejected(
    codec: HmacSha256ProjectionIdentifierCodec,
    source: object,
) -> None:
    with pytest.raises(TypeError):
        codec.encode(
            ProjectionIdentifierDomain.ENTITY,
            generation="generation-a",
            source=source,
        )


@pytest.mark.parametrize(
    "source",
    [
        "12345678123456789234567812345678",
        "12345678-1234-5678-9234-56781234567A",
        "{12345678-1234-5678-9234-567812345678}",
    ],
)
def test_noncanonical_uuid_source_strings_are_rejected(
    codec: HmacSha256ProjectionIdentifierCodec,
    source: str,
) -> None:
    with pytest.raises(ValueError, match="canonical UUID"):
        codec.encode(
            ProjectionIdentifierDomain.ENTITY,
            generation="generation-a",
            source=source,
        )


@pytest.mark.parametrize("generation", [None, "", " ", True, b"generation-a"])
def test_generation_is_mandatory_and_canonical_for_scoped_domains(
    codec: HmacSha256ProjectionIdentifierCodec,
    generation: object,
) -> None:
    expected_error = (
        ValueError if generation is None or type(generation) is str else TypeError
    )
    with pytest.raises(expected_error):
        codec.encode(
            ProjectionIdentifierDomain.ENTITY,
            generation=generation,
            source=11,
        )


@pytest.mark.parametrize("source", ["", " "])
def test_empty_sources_are_rejected(
    codec: HmacSha256ProjectionIdentifierCodec,
    source: str,
) -> None:
    with pytest.raises(ValueError):
        codec.encode(
            ProjectionIdentifierDomain.ENTITY,
            generation="generation-a",
            source=source,
        )


def test_domain_must_be_the_closed_enum(
    codec: HmacSha256ProjectionIdentifierCodec,
) -> None:
    with pytest.raises(TypeError):
        codec.encode("entity", generation="generation-a", source=11)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"key": b"", "key_version": "test-key-v1"}, ValueError),
        ({"key": "secret", "key_version": "test-key-v1"}, TypeError),
        ({"key": TEST_KEY, "key_version": ""}, ValueError),
        ({"key": TEST_KEY, "key_version": True}, TypeError),
        (
            {"key": TEST_KEY, "key_version": "test-key-v1", "codec_version": ""},
            ValueError,
        ),
    ],
)
def test_codec_configuration_rejects_empty_or_ambiguous_values(
    kwargs: dict[str, object],
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        HmacSha256ProjectionIdentifierCodec(**kwargs)  # type: ignore[arg-type]
