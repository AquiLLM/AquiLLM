"""Shared bounded topology wire budgets, independent of Django/provider code."""

from typing import Final

# Fits 128 generations (including maximum UTF-8/escaped version tokens), 10,000
# authorized document triples plus repeated key lists, and 64 weighted seeds.
# The request-capacity regression verifies the canonical nested wire encoding.
MAX_REQUEST_BYTES: Final = 4_194_304
MAX_PARAMETER_JSON_BYTES: Final = 3_145_728
MAX_RESPONSE_BYTES: Final = 1_048_576
