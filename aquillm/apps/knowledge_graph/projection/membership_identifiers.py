"""Private per-invocation integer encoding; never caches membership authority."""

from . import identifiers as ids

Domain = ids.ProjectionIdentifierDomain


def _current_binding():
    native = ids.HmacSha256ProjectionIdentifierCodec
    return (
        native,
        native.__init__,
        native.__getattribute__,
        native.encode,
        native.frame,
        native._identifier_input,
        native._key,
        native.key_version,
        native.codec_version,
        ids._ProjectionIdentifierInput,
        ids._ProjectionIdentifierInput.__init__,
        ids._ProjectionIdentifierInput.__getattribute__,
        ids._ProjectionIdentifierInput.__new__,
        ids._ProjectionIdentifierInput.codec_version,
        ids._ProjectionIdentifierInput.key_version,
        ids._ProjectionIdentifierInput.domain,
        ids._ProjectionIdentifierInput.generation,
        ids._ProjectionIdentifierInput.source_kind,
        ids._ProjectionIdentifierInput.canonical_source,
        ids.OpaqueProjectionKey,
        ids.OpaqueProjectionKey.__init__,
        ids.OpaqueProjectionKey.__getattribute__,
        ids.OpaqueProjectionKey.__new__,
        ids.OpaqueProjectionKey.__post_init__,
        ids.OpaqueProjectionKey.domain,
        ids.OpaqueProjectionKey.value,
        ids.ProjectionIdentifierDomain,
        ids.ProjectionIdentifierDomain.ENTITY,
        ids.ProjectionIdentifierDomain.ENTITY.value,
        ids.ProjectionIdentifierDomain.AUTOMATIC_CANONICAL_IDENTITY,
        ids.ProjectionIdentifierDomain.AUTOMATIC_CANONICAL_IDENTITY.value,
        ids.UUID,
        ids.UUID.__str__,
        ids.UUID.__init__,
        ids.UUID.__getattribute__,
        ids.hmac_new,
        ids.sha256,
        ids._canonical_generation,
        ids._canonicalize_source,
        ids._canonical_uuid_or_text,
        ids._canonical_token,
        ids._canonical_text,
        ids._frame_field,
        ids._SIGNED_64_BIT_MIN,
        ids._SIGNED_64_BIT_MAX,
    )


def _generation_state(value):
    if type(value) is str:
        return str, value
    if type(value) is ids.UUID:
        return ids.UUID, value.int
    return type(value), None


class MembershipKeyEncoder:
    """Keep prepared HMAC states and canonical-ID memo inside this invocation."""

    def __init__(self, codec, generation):
        self.codec, self.generation = codec, generation
        self._generic = False
        self._fields = self._generation = self._prefixes = None
        self._canonical = {}

    def _discard(self):
        self._generic = True
        self._fields = self._generation = self._prefixes = None
        self._canonical.clear()
        return False

    def _ready(self):
        if self._generic:
            return False
        try:
            current = _current_binding()
            if (
                type(self.codec) is not ids._NATIVE_HMAC_ENCODING_BINDING[0]
                or len(current) != len(ids._NATIVE_HMAC_ENCODING_BINDING)
                or any(
                    a is not b
                    for a, b in zip(current, ids._NATIVE_HMAC_ENCODING_BINDING)
                )
            ):
                return self._discard()
            fields = self.codec._key, self.codec.key_version, self.codec.codec_version
            generation = _generation_state(self.generation)
            if (
                type(fields[0]) is not bytes
                or not fields[0]
                or type(fields[1]) is not str
                or type(fields[2]) is not str
            ):
                return self._discard()
            if self._fields is not None:
                if fields != self._fields or generation != self._generation:
                    return self._discard()
                return True
            for value in fields[1:]:
                ids._canonical_token(value, field_name="version")
            suffix = (1).to_bytes(4, "big") + b"0"
            entity = self.codec.frame(
                Domain.ENTITY, generation=self.generation, source=0
            )
            canonical = self.codec.frame(Domain.AUTOMATIC_CANONICAL_IDENTITY, source=0)
            if not entity.endswith(suffix) or not canonical.endswith(suffix):
                return self._discard()
            self._prefixes = tuple(
                ids.hmac_new(fields[0], frame[: -len(suffix)], ids.sha256)
                for frame in (entity, canonical)
            )
            self._fields, self._generation = fields, generation
            return True
        except (AttributeError, TypeError, ValueError):
            return self._discard()  # Generic encoding retains original validation.

    def _integer(self, source):
        return type(source) is int and -(2**63) <= source <= 2**63 - 1

    def _encoded(self, source, index):
        value = str(source).encode("utf-8")
        prepared = self._prefixes[index].copy()
        prepared.update(len(value).to_bytes(4, "big") + value)
        return prepared.hexdigest()

    def entity(self, source):
        if self._integer(source) and self._ready():
            return self._encoded(source, 0)
        return self.codec.encode(
            Domain.ENTITY, generation=self.generation, source=source
        ).value

    def canonical(self, source):
        if self._integer(source) and self._ready():
            if source not in self._canonical:
                self._canonical[source] = self._encoded(source, 1)
            return self._canonical[source]
        return self.codec.encode(
            Domain.AUTOMATIC_CANONICAL_IDENTITY, source=source
        ).value
