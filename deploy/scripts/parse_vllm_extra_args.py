#!/usr/bin/env python3
"""Parse VLLM_EXTRA_ARGS into argv-safe tokens."""

from __future__ import annotations

import json
import shlex
import sys


def _normalize_json_like_token(token: str) -> str:
    # Accept values that may arrive doubly escaped via env file parsing:
    # e.g. '{\"method\":\"ngram\"}' -> '{"method":"ngram"}'
    candidates = [
        token,
        token.strip("'\""),
        token.replace('\\"', '"'),
        token.strip("'\"").replace('\\"', '"'),
    ]

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, (dict, list)):
            return candidate

    return token


def parse_extra_args(raw: str) -> list[str]:
    if not raw:
        return []

    tokens = shlex.split(raw, posix=True)
    return [_normalize_json_like_token(token) for token in tokens]


def main() -> int:
    raw = sys.argv[1] if len(sys.argv) > 1 else ""
    for token in parse_extra_args(raw):
        sys.stdout.buffer.write(token.encode("utf-8"))
        sys.stdout.buffer.write(b"\0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

