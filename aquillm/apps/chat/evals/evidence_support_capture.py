"""Private exact publication capture, independent of provider answer claims."""


class PublicationCapture:
    def __init__(self):
        self.streams, self.deltas = {}, {}
        self.invalid = False

    def outbound(self, payload):
        stream = payload.get("stream")
        if (
            isinstance(stream, dict)
            and stream.get("role") == "assistant"
            and stream.get("content")
        ):
            if stream.get("done") is not True:
                self.invalid = True
            self._add(self.streams, stream)
        for message in payload.get("delta", {}).get("messages", []):
            if (
                message.get("role") == "assistant"
                and not message.get("tool_call_name")
                and message.get("content")
            ):
                self._add(self.deltas, message)

    def _add(self, target, message):
        identity, content = str(message.get("message_uuid", "")), message.get("content")
        if not identity or not isinstance(content, str):
            self.invalid = True
            return
        target.setdefault(identity, set()).add(content)

    def result(self, message_id, persisted):
        identity = str(message_id) if message_id else None
        streams, deltas = (
            self.streams.get(identity, set()),
            self.deltas.get(identity, set()),
        )
        return {
            "final_message_id": identity,
            "final_stream": next(iter(streams)) if len(streams) == 1 else None,
            "final_delta": next(iter(deltas)) if len(deltas) == 1 else None,
            "persisted_answer": persisted,
            "publication_capture_complete": (
                not self.invalid
                and len(streams) == len(deltas) == 1
                and isinstance(persisted, str)
            ),
        }


def persisted_content(conversation_id, message_id):
    """Query actual PostgreSQL rows; ambiguous identity cannot attest persistence."""
    from django.db.models import Q

    from apps.chat.models import Message

    if not message_id:
        return None
    values = list(
        Message.objects.filter(
            Q(tool_call_name__isnull=True) | Q(tool_call_name=""),
            conversation_id=conversation_id,
            message_uuid=message_id,
            role="assistant",
        ).values_list("content", flat=True)[:2]
    )
    return values[0] if len(values) == 1 else None


def packet_support(packet):
    from apps.chat.services.rag_answer_context import _evidence_digest, _scope_digest

    from .evidence_observation_json import normalize

    support = normalize(packet.answer_support) if packet.answer_support else None
    return {
        "support_decision": support,
        "support_packet_identity": {
            "evidence_digest": _evidence_digest(packet),
            "scope_digest": _scope_digest(packet),
        },
        "support_question": packet.query,
        "source_scope": {
            "kind": packet.search_scope,
            "digest": support.get("scope_digest") if support else None,
        },
        "policy_identity": support.get("calibration_id") if support else None,
        "score_metadata": [
            {"scoring_kind": c.scoring_kind, "score_status": c.score_interpretation}
            for c in packet.selection.candidates
        ]
        if packet.selection
        else [],
    }


def delivery_binding(packet, payload, delivered):
    from apps.chat.services.rag_answer_context import _evidence_digest, _scope_digest

    from .evidence_quality_eval import digest

    return {
        "evidence_digest": _evidence_digest(packet),
        "scope_digest": _scope_digest(packet),
        "payload_digest": digest(payload),
        "delivered_digest": digest(delivered),
    }


def support_delivery_matches(row):
    """Final support must refer to exactly the last actual SDK delivery."""
    import re

    from .evidence_quality_eval import digest

    binding = row.get("support_delivery_binding")
    support = row.get("support_decision")
    payloads = row.get("sdk_payloads")
    return (
        isinstance(binding, dict)
        and isinstance(support, dict)
        and isinstance(payloads, list)
        and bool(payloads)
        and all(
            isinstance(binding.get(k), str)
            and re.fullmatch(r"[0-9a-f]{64}", binding[k]) is not None
            for k in (
                "evidence_digest",
                "scope_digest",
                "payload_digest",
                "delivered_digest",
            )
        )
        and all(
            binding[k] == support.get(k) for k in ("evidence_digest", "scope_digest")
        )
        and all(
            binding[k] == row.get("support_packet_identity", {}).get(k)
            for k in ("evidence_digest", "scope_digest")
        )
        and binding["scope_digest"] == row.get("source_scope", {}).get("digest")
        and binding["payload_digest"] == digest(payloads[-1])
        and binding["delivered_digest"] == digest(row.get("delivered"))
    )


async def publication_result(capture, conversation_id, message_id, case, manifest):
    from channels.db import database_sync_to_async

    from .evidence_quality_eval import digest

    persisted = await database_sync_to_async(persisted_content)(
        conversation_id, message_id
    )
    runtime = manifest.get("runtime", {})
    fields = ("answer_digest", "tokenizer_digest", "input_preparation_digest")
    # A digest of missing runtime fields cannot become a verified identity.
    runtime_identity = (
        digest({k: runtime[k] for k in fields})
        if all(runtime.get(k) for k in fields)
        else None
    )
    return {
        **capture.result(message_id, persisted),
        "primary_question": case["question"],
        "domain": case["genre"],
        "depth": case["depth"],
        "runtime_identity": runtime_identity,
    }
