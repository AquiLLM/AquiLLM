"""One narrow state-role publication call; source freshness belongs to readers."""

import json
import re

from django.db import connections

from .membership_certificate_contracts import validate_membership_certificate_payload

_HEX = re.compile(r"[0-9a-f]{64}\Z")


def publish_membership_certificate(
    *, payload: dict, signature: str, previous_signature: str | None, using: str, check
) -> bool:
    check()
    validate_membership_certificate_payload(payload)
    for value in (signature,) + (
        () if previous_signature is None else (previous_signature,)
    ):
        if type(value) is not str or _HEX.fullmatch(value) is None:
            raise ValueError("membership certificate signature is invalid")
    if type(signature) is not str or type(using) is not str or not using:
        raise ValueError("membership certificate publication request is invalid")
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    check()
    with connections[using].cursor() as cursor:
        check()
        cursor.execute(
            "SELECT public.kg_publish_membership_certificate_v1(%s::jsonb,%s,%s)",
            [encoded, signature, previous_signature],
        )
        check()
        row = cursor.fetchone()
        check()
        if row is None or len(row) != 1 or type(row[0]) is not bool:
            raise ValueError("membership certificate publication result is invalid")
        result = row[0]
    check()
    return result
