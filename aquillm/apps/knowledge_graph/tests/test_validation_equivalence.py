"""Preserve the closed key/text domains when accelerating their validation."""

import pytest

from apps.knowledge_graph.projection.serialization import _key
from apps.knowledge_graph.retrieval.topology.contracts import TopologyQueryName
from apps.knowledge_graph.retrieval.topology.gateway_contracts import (
    GatewayRequestSizeError,
    TopologyGatewayRequestV1,
    TopologyGatewaySuccessV1,
    _safe_text,
    decode_request,
    decode_response,
    encode_request,
    encode_response,
)


class _StringSubclass(str):
    pass


@pytest.mark.parametrize("value", [None, 0, True, b"a" * 64, _StringSubclass("a" * 64)])
def test_key_and_text_reject_non_builtin_strings_before_other_checks(value):
    with pytest.raises(TypeError, match="^key must be a built-in str$"):
        _key(value, "key")
    with pytest.raises(TypeError, match="^text must be an exact string$"):
        _safe_text(value, "text", 0, size_error=GatewayRequestSizeError)


@pytest.mark.parametrize("value", ["0" * 64, "f" * 64, "0123456789abcdef" * 4])
def test_key_accepts_exact_lowercase_ascii_hexadecimal(value):
    assert _key(value, "key") is None


@pytest.mark.parametrize(
    "value",
    [
        "",
        "a" * 63,
        "a" * 65,
        "A" * 64,
        "g" * 64,
        "0" * 63 + "\n",
        " " + "0" * 63,
        "0" * 63 + " ",
        "\uff10" * 64,
        "\u0660" * 64,
        "a" * 63 + "\ud800",
    ],
)
def test_key_rejects_length_case_whitespace_and_non_ascii_hex_defects(value):
    with pytest.raises(
        ValueError, match="^key must be a lowercase SHA-256 hexadecimal key$"
    ):
        _key(value, "key")


def test_every_c0_del_and_surrogate_is_forbidden_with_size_error_precedence():
    # A missing surrogate half or an off-by-one character range must fail here.
    for codepoint in (*range(0x20), 0x7F, *range(0xD800, 0xE000)):
        value = "a" + chr(codepoint) + "b"
        with pytest.raises(ValueError) as error:
            _safe_text(value, "text", 3)
        assert type(error.value) is ValueError
        assert str(error.value) == "text contains forbidden control text"
        for error_class in (ValueError, GatewayRequestSizeError):
            with pytest.raises(error_class) as error:
                _safe_text(value, "text", 2, size_error=error_class)
            assert type(error.value) is error_class
            assert str(error.value) == "text exceeds its text cap"


@pytest.mark.parametrize(
    "value",
    [
        "",
        " ",
        "~",
        "\x80",
        "\ud7ff",
        "\ue000",
        "\uffff",
        "\U0010ffff",
        "e\u0301",
        "\U0001f600",
        'quote"\\',
    ],
)
def test_allowed_unicode_uses_codepoint_length_without_normalizing(value):
    assert _safe_text(value, "text", len(value)) is None
    if value:
        with pytest.raises(
            GatewayRequestSizeError, match="^text exceeds its text cap$"
        ):
            _safe_text(
                value, "text", len(value) - 1, size_error=GatewayRequestSizeError
            )


def test_unicode_gateway_round_trips_preserve_literal_canonical_bytes():
    text = 'e\u0301\U0001f600"\\'
    request = TopologyGatewayRequestV1(
        TopologyQueryName.RELATION_TOPOLOGY, {"text": text}, 1.0, 1
    )
    response = TopologyGatewaySuccessV1(({"text": text},))
    expected_text = b'e\xcc\x81\xf0\x9f\x98\x80\\"\\\\'
    expected_request = (
        b'{"deadline":1.0,"max_records":1,"parameters":{"text":"'
        + expected_text
        + b'"},"query":"relation_topology"}'
    )
    expected_response = b'{"ok":true,"rows":[{"text":"' + expected_text + b'"}]}'
    assert encode_request(request) == expected_request
    assert encode_response(response) == expected_response
    assert encode_request(decode_request(expected_request)) == expected_request
    assert encode_response(decode_response(expected_response)) == expected_response
