"""Exact at-most-25600-bit captured competing-pair mask; no inferred locality."""


def encode_mask(pairs, generations, identities):
    mask = bytearray((generations * identities + 7) // 8)
    for generation, identity in pairs:
        offset = generation * identities + identity
        mask[offset // 8] |= 1 << (offset % 8)
    return mask.hex()


def decode_mask(value, generations, identities):
    bits = generations * identities
    if (
        type(value) is not str
        or len(value) != 2 * ((bits + 7) // 8)
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("frontier pair mask encoding invalid")
    mask = bytes.fromhex(value)
    if bits % 8 and mask[-1] >> (bits % 8):
        raise ValueError("frontier pair mask padding invalid")
    return tuple(
        (offset // identities, offset % identities)
        for offset in range(bits)
        if mask[offset // 8] & (1 << (offset % 8))
    )
