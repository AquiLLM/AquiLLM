"""Conservative planning geometry; records are not proof of runtime registration."""
from math import lcm, prod

from kv_cache_config import ConfigurationError, MODEL, MAX_BYTES

IDENTITY = {
    "model": MODEL, "revision": "2d783431e303148fc6e16622fac5edac83a6b5c4",
    "quantization": "awq", "kv_dtype": "turboquant_k8v4",
    "tensor_parallel_size": 1, "pipeline_parallel_size": 1, "mtp_depth": 4,
}


def integer(value, name, *, zero=False):
    if type(value) is not int or not (0 if zero else 1) <= value <= MAX_BYTES:
        raise ConfigurationError(f"{name} must be a bounded {'nonnegative' if zero else 'positive'} integer")
    return value


def dimensions(value, name):
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"{name} must be a nonempty list")
    return [integer(v, name) for v in value]


def validate_layout(record):
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise ConfigurationError("runtime layout record schema_version=1 is required; automatic geometry is unsupported")
    if record.get("identity") != IDENTITY:
        raise ConfigurationError("layout identity must match pinned model/revision/quantization/topology/MTP")
    groups = record.get("groups", [])
    if not isinstance(groups, list) or any(not isinstance(g, dict) for g in groups) or {g.get("kind") for g in groups} != {"attention", "recurrent"}:
        raise ConfigurationError("layout requires separate attention and recurrent groups")
    chunk, counts = 1, {"attention": 0, "recurrent": 0}
    for group in groups:
        kind = group["kind"]
        counts[kind] += integer(group.get("layers"), "layers")
        span = integer(group.get("logical_block_tokens"), "logical_block_tokens")
        page = integer(group.get("page_bytes"), "page_bytes")
        chunk = lcm(chunk, span)
        if chunk > 262144:
            raise ConfigurationError("common chunk span exceeds model context")
        if kind == "attention":
            shape = dimensions(group.get("shape"), "attention shape")
            stride = dimensions(group.get("stride"), "attention stride")
            if group.get("dtype") != "uint8" or len(shape) != 4 or shape[2] != 4 or shape[3] < 388:
                raise ConfigurationError("K8V4 requires 4D uint8 NHD with four heads and >=388 physical bytes/slot")
            if shape[1] != span:
                raise ConfigurationError("4D K8V4 logical block differs from physical token axis; subpaging adapter unvalidated")
            expected = [shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3], 1]
            if stride != expected or page != stride[0]:
                raise ConfigurationError("unsupported attention stride/page padding; preserve bytes until addressing is validated")
        else:
            if group.get("dtype") != "bfloat16" or page % 256:
                raise ConfigurationError("recurrent pages require bfloat16 and 256-byte alignment")
            states = group.get("states", [])
            if len(states) != 2:
                raise ConfigurationError("recurrent group requires conv and SSM state descriptions")
            previous_end = 0
            for state in states:
                offset = integer(state.get("offset_bytes"), "state offset", zero=True)
                size = prod(dimensions(state.get("shape"), "state shape")) * 2
                if offset % 2 or offset < previous_end or offset + size > page:
                    raise ConfigurationError("recurrent state overlap/alignment/page bounds invalid")
                previous_end = offset + size
    if counts != {"attention": 16, "recurrent": 48}:
        raise ConfigurationError("layout layer counts must match 16 attention and 48 recurrent layers")
    return chunk
