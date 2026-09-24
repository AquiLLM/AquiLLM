"""Private live-run observations; no fabricated retrieval or provider responses."""

import json
from contextlib import contextmanager
from time import perf_counter
from unittest.mock import patch

from lib.evidence_observation import active, publish

from .evidence_observation_json import normalize, normalize_observation
from .evidence_quality_delivery import map_payload, parse_citations


@contextmanager
def transport_observation(unattributed, worker_warmup=None):
    """Observe real requests transport. Preserve bytes, arguments and responses."""
    import requests

    original = requests.sessions.Session.send

    def send(session, request, **kwargs):
        start = perf_counter()
        body = request.body
        try:
            payload = json.loads(body) if body else {}
        except (ValueError, TypeError):
            payload = {}
        scoring = any(key in payload for key in ("text_1", "documents"))
        response = None
        try:
            response = original(session, request, **kwargs)
            return response
        finally:
            if scoring:
                item = {
                    "duration_ms": (perf_counter() - start) * 1000,
                    "pairs": len(payload.get("documents", payload.get("text_2", [])))
                    if isinstance(payload.get("documents", payload.get("text_2")), list)
                    else 1,
                    "status": getattr(response, "status_code", None),
                }
                if active():
                    publish("rerank_http", item)
                else:
                    from threading import current_thread

                    if (
                        worker_warmup is not None
                        and current_thread().name == "pair-capability"
                    ):
                        worker_warmup.append(item)
                    else:
                        unattributed.append(item)

    with patch.object(requests.sessions.Session, "send", send):
        yield


class Trace:
    def __init__(self, sources):
        self.sources = sources
        self.started = perf_counter()
        self.events = []
        self.delivered, self.unknown, self.sdk_payloads = [], [], []
        self.done = self.sdk_started = None
        self.final = {}
        self.timings = {}
        self.rank_fallback = None
        self.coverage = None
        self.upstream = []
        self.window_scores = []

    def sink(self, event, data):
        from apps.documents.services.source_loading import current_source_runtime

        runtime = current_source_runtime()
        data = normalize(data)
        self.events.append({"event": event, **data})
        if event == "retrieval_sources":
            mapped, _ = map_payload(
                {"content": json.dumps({"result": data["rows"]})}, self.sources
            )
            self.upstream.extend(mapped)
        if event == "window_score":
            self.window_scores.append(data)
        if event == "rag_metrics":
            self.timings.update(
                {k.removesuffix("_ms"): v for k, v in data.items() if k.endswith("_ms")}
            )
            self.final.update(
                {
                    k: data[k]
                    for k in ("new_pairs", "reused_pairs", "fixed_fallback_reason")
                    if k in data
                }
            )
        if event == "sdk_start":
            if self.sdk_started:
                self.sdk_started.set()
            hints = {}
            packet = runtime.observation.get("delivered_packet") if runtime else None
            if packet:
                for prepared in packet.source_evidence:
                    key = (str(prepared.source.document_id), prepared.source.chunk_id)
                    hints[key] = [(s.start, s.end) for s in prepared.spans]
                self.rank_fallback = (
                    packet.selection.score_status == "rank_fallback"
                    if packet.selection
                    else None
                )
                self.coverage = packet.retrieval_status
            self.delivered, self.unknown = map_payload(
                data["payload"], self.sources, hints
            )
            self.sdk_payloads.append(data["payload"])
        if event == "turn_complete":
            self.final.update(data)
            self.timings["completion"] = (data["at"] - self.started) * 1000
            if runtime:
                self.timings.update(
                    {
                        k.removesuffix("_ms"): v
                        for k, v in runtime.observation.get("timings", {}).items()
                    }
                )
                acquired = runtime.observation.get("acquisition")
                if acquired:
                    self.final["stop_reason"] = acquired.stop_reason
            if self.done:
                self.done.set()

    def result(self, answer):
        sdk = [e for e in self.events if e["event"] == "sdk_start"]
        http = [e for e in self.events if e["event"] == "rerank_http"]
        cache = [e for e in self.events if e["event"] == "cache_operation"]
        charged = sum(
            e.get("event") == "ledger"
            and e.get("operation") == "start_pair"
            and e.get("result") is True
            for e in self.events
        )
        accounted = len(sdk) == sum(e["event"] == "sdk_end" for e in self.events)
        if charged:
            accounted = accounted and charged == sum(e["pairs"] for e in http)
        return normalize_observation(
            {
                "dispatch_accounting_complete": accounted,
                "answer": answer,
                "source_bindings": [
                    {
                        "document_id": doc,
                        "chunk_id": chunk,
                        "source_id": source["source_id"],
                        "revision": source["revision"],
                    }
                    for (doc, chunk), source in self.sources.items()
                ],
                "events": self.events,
                "delivered": self.delivered,
                "upstream": self.upstream,
                "citations": parse_citations(answer, self.sources),
                "provenance_complete": bool(sdk) and not self.unknown,
                "unknown_provenance": self.unknown,
                "timings_ms": self.timings,
                "dispatches": [
                    {"kind": e["kind"], "provider": e["provider"]} for e in sdk
                ],
                "sdk_payloads": self.sdk_payloads,
                "inference_pairs": sum(e["pairs"] for e in http),
                "rerank_http": http,
                "cache_operations": cache,
                "rank_fallback": self.rank_fallback,
                "window_scores": self.window_scores,
                "coverage": self.coverage,
                **{k: v for k, v in self.final.items() if k != "at"},
            }
        )
