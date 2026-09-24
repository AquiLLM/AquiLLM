"""Conservative local estimates; these are not provider tokenizer guarantees."""

from math import ceil

from tiktoken import encoding_for_model

_ENCODING = encoding_for_model("gpt-4o")


def estimate_text_tokens(text: str) -> int:
    return ceil(len(_ENCODING.encode(text, disallowed_special=())) * 1.2)
