#!/usr/bin/env python3
"""Explicit-endpoint KV benchmark. HTTP overlap is not scheduler/paging proof.

Only committed server usage counts outputs. Prompts are supplied token IDs;
there is no tokenizer download, text-length fallback, or model loading.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import re
import socket
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

IDENTITY_KEYS = ("model", "revision", "runtime", "quantization", "mtp", "hardware", "genesis", "weights")
PROFILE_KEYS = ("active_sequences", "context_tokens", "retained_contexts", "kv_dtype", "storage_mode",
                "execution_mode", "tier_budgets", "full_attention_payload_bytes", "validation_status")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def positive(value, name, limit):
    if type(value) is not int or not 1 <= value <= limit:
        raise ValueError(f"{name} must be an integer in [1, {limit}]")
    return value


def has_identity(profile):
    identity = profile.get("identity", {})
    return (isinstance(identity, dict) and all(isinstance(identity.get(key), str) and identity[key]
                                             for key in IDENTITY_KEYS)
            and identity.get("weights") == "gpu-resident" and identity.get("mtp") == "depth-4"
            and identity.get("quantization") == "awq")


def validate(profile, fixture, url, model, output_tokens, concurrency, timeout):
    n = positive(profile.get("active_sequences"), "active_sequences", 256)
    t = positive(profile.get("context_tokens"), "context_tokens", 262144)
    positive(output_tokens, "output_tokens", t)
    positive(concurrency if concurrency is not None else n, "concurrency", n)
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 86400:
        raise ValueError("timeout must be finite, positive and <=86400 seconds")
    endpoint = urlsplit(url)
    if (endpoint.scheme not in {"http", "https"} or not endpoint.hostname or endpoint.username
            or endpoint.password or endpoint.query or endpoint.fragment):
        raise ValueError("url must be an explicit HTTP(S) completion endpoint without credentials/query/fragment")
    endpoint.port  # Validate before any worker makes a request.
    if profile.get("kv_dtype") != "turboquant_k8v4":
        raise ValueError("profile must retain turboquant_k8v4")
    if fixture.get("model") != model or not isinstance(fixture.get("revision"), str) or not fixture["revision"]:
        raise ValueError("token fixture must identify the selected model and revision")
    identity = profile.get("identity", {})
    if not isinstance(identity, dict):
        raise ValueError("profile identity must be an object")
    for key in ("model", "revision"):
        if key in identity and identity[key] != fixture[key]:
            raise ValueError("profile and token fixture identity mismatch")
    prompts = fixture.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != n:
        raise ValueError("fixture must contain exactly active_sequences prompts")
    fingerprints = []
    for prompt in prompts:
        if (not isinstance(prompt, list) or not prompt or len(prompt) + output_tokens > t
                or any(type(token) is not int or not 0 <= token <= 2147483647 for token in prompt)):
            raise ValueError("each prompt must contain valid token IDs and reserve output inside context_tokens")
        fingerprints.append(digest(prompt))
    if len(set(fingerprints)) != n:
        raise ValueError("each request must have a distinct token-ID prompt")
    return n, t, fingerprints


def stream_request(url, payload, request_id, origin, timeout, api_key=None):
    start = time.perf_counter()
    first = previous = None
    gaps, chunks, usage, finish, error, done = [], 0, None, None, None, False
    endpoint = urlsplit(url)
    connection_type = http.client.HTTPSConnection if endpoint.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(endpoint.hostname, endpoint.port, timeout=timeout)
    expired = threading.Event()
    socket_ref = [None]
    def abort():
        expired.set()
        active_socket = socket_ref[0] or connection.sock
        if active_socket is not None:
            try:
                active_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
    deadline = threading.Timer(timeout, abort)
    deadline.daemon = True
    deadline.start()
    try:
        headers = {"Content-Type": "application/json", "X-Request-ID": request_id}
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
        connection.request("POST", endpoint.path or "/", json.dumps(payload).encode(), headers)
        if expired.is_set():
            raise TimeoutError()
        sock = connection.sock
        socket_ref[0] = sock
        response = connection.getresponse()
        if response.status != 200:
            error = f"http_status_{response.status}"
        elif "text/event-stream" not in response.getheader("Content-Type", "").lower():
            error = "invalid_content_type"
        else:
            buffer, data_lines, event_type = b"", [], ""
            while not done and error is None:
                remaining = timeout - (time.perf_counter() - start)
                if remaining <= 0:
                    raise TimeoutError()
                sock.settimeout(remaining)
                raw = response.read1(65536)
                if not raw:
                    break
                buffer += raw
                if len(buffer) > 1048576:
                    raise ValueError("oversized SSE event")
                while b"\n" in buffer and not done:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.rstrip(b"\r")
                    if not line:
                        if event_type == "error":
                            error = "server_error"
                        if data_lines:
                            data = b"\n".join(data_lines).decode("utf-8")
                            if data == "[DONE]":
                                done = True
                            else:
                                item = json.loads(data)
                                if item.get("error") is not None:
                                    error = "server_error"
                                if item.get("usage") is not None:
                                    if usage is not None and usage != item["usage"]:
                                        error = "conflicting_usage"
                                    usage = item["usage"]
                                for choice in item.get("choices", []):
                                    if choice.get("index", 0) != 0:
                                        error = "multiple_choices"
                                    if choice.get("finish_reason") is not None:
                                        finish = choice["finish_reason"]
                                    if choice.get("text") or choice.get("delta", {}).get("content"):
                                        now = time.perf_counter()
                                        if first is None:
                                            first = now
                                        if previous is not None:
                                            gaps.append(now - previous)
                                        previous = now
                                        chunks += 1
                        data_lines, event_type = [], ""
                    elif line.startswith(b"data:"):
                        data_lines.append(line[5:].lstrip(b" "))
                        if sum(map(len, data_lines)) > 1048576:
                            raise ValueError("oversized SSE event")
                    elif line.startswith(b"event:"):
                        event_type = line[6:].strip().decode("utf-8")
    except (OSError, TimeoutError, http.client.HTTPException):
        error = "transport_or_timeout_error"
    except (ValueError, TypeError, KeyError, AttributeError):
        error = "invalid_stream"
    finally:
        deadline.cancel()
        connection.close()
    if expired.is_set():
        error = "transport_or_timeout_error"
    end = time.perf_counter()
    count = usage.get("completion_tokens") if isinstance(usage, dict) else None
    prompt_count = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    if error is None:
        if not done:
            error = "missing_done"
        elif finish != "length" or first is None:
            error = "incomplete_output"
        elif count is not None and (type(count) is not int or count != payload["max_tokens"]):
            error = "invalid_completion_tokens"
        elif prompt_count is not None and (type(prompt_count) is not int or prompt_count != len(payload["prompt"])):
            error = "prompt_usage_mismatch"
    valid_count = type(count) is int and count >= 0 and error is None
    elapsed = end - start
    return dict(request_id=request_id, started_seconds=start - origin, ended_seconds=end - origin,
                elapsed_seconds=elapsed, first_token_latency_seconds=None if first is None else first - start,
                accepted_output_tokens=count if valid_count else None,
                token_accounting_source="server_usage.completion_tokens" if valid_count else "unavailable",
                accepted_tokens_per_second=count / elapsed if valid_count else None,
                decode_only_tokens_per_second=None, decode_only_status="unavailable-first-chunk-token-count",
                nonempty_chunks=chunks, streaming_gaps_seconds=gaps,
                max_stream_gap_seconds=max(gaps) if gaps else None,
                terminal_wait_seconds=None if previous is None else end - previous,
                observed_prompt_tokens=prompt_count if type(prompt_count) is int else None,
                stream_done=done, error=error)


def maximum_overlap(rows):
    events = [(row[edge], change) for row in rows for edge, change in
              (("started_seconds", 1), ("ended_seconds", -1))]
    count = maximum = 0
    for _, change in sorted(events):
        count += change
        maximum = max(maximum, count)
    return maximum


def finite_number(value, *, positive=False):
    try:
        return (type(value) in (int, float) and math.isfinite(value)
                and (value > 0 if positive else value >= 0))
    except OverflowError:
        return False


def valid_accounting(row, context_tokens):
    """Validate observed accounting; cached verdicts and positive rates are insufficient."""
    if not isinstance(row, dict) or type(context_tokens) is not int:
        return False
    counts = ("supplied_prompt_tokens", "requested_output_tokens", "accepted_output_tokens",
              "reserved_context_tokens", "exercised_context_tokens", "nonempty_chunks")
    if any(type(row.get(key)) is not int or row[key] <= 0 for key in counts):
        return False
    prompt, requested, accepted = (row[key] for key in counts[:3])
    if (row.get("token_accounting_source") != "server_usage.completion_tokens"
            or "error" not in row or row["error"] is not None or row.get("stream_done") is not True
            or accepted != requested or row["reserved_context_tokens"] != prompt + requested
            or row["exercised_context_tokens"] != prompt + accepted or prompt + requested > context_tokens):
        return False
    observed = row.get("observed_prompt_tokens")
    if observed is not None and (type(observed) is not int or observed != prompt):
        return False
    elapsed, rate, ttft = (row.get(key) for key in
                           ("elapsed_seconds", "accepted_tokens_per_second", "first_token_latency_seconds"))
    start, end = row.get("started_seconds"), row.get("ended_seconds")
    if (not all(finite_number(value, positive=True) for value in (elapsed, rate, ttft, end))
            or not finite_number(start) or end <= start or ttft > elapsed
            or not math.isclose(elapsed, end - start, rel_tol=1e-6, abs_tol=1e-8)
            or not math.isclose(rate, accepted / elapsed, rel_tol=1e-9, abs_tol=1e-9)):
        return False
    gaps, terminal = row.get("streaming_gaps_seconds"), row.get("terminal_wait_seconds")
    if (not isinstance(gaps, list) or len(gaps) != row["nonempty_chunks"] - 1
            or not all(finite_number(gap) for gap in gaps) or not finite_number(terminal)
            or not math.isclose(ttft + sum(gaps) + terminal, elapsed, rel_tol=1e-6, abs_tol=1e-8)):
        return False
    maximum = row.get("max_stream_gap_seconds")
    return (maximum is None if not gaps else finite_number(maximum)
            and math.isclose(maximum, max(gaps), rel_tol=1e-9, abs_tol=1e-9))


def valid_run_records(result):
    try:
        n = positive(result["configured_n"], "N", 256)
        t = positive(result["configured_t"], "T", 262144)
        profile, rows = result["profile"], result["requests"]
        if (result.get("schema_version") != 1 or result.get("status") != "complete"
                or profile.get("active_sequences") != n or profile.get("context_tokens") != t
                or profile.get("kv_dtype") != "turboquant_k8v4" or not isinstance(rows, list)
                or len(rows) != n or not all(valid_accounting(row, t) for row in rows)):
            return False
        if not isinstance(result.get("run_id"), str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", result["run_id"]):
            return False
        fingerprints = [row.get("prompt_sha256") for row in rows]
        if (any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in fingerprints)
                or len(set(fingerprints)) != n or any(row.get("request_id") != f"{result['run_id']}:{index}"
                for index, row in enumerate(rows))):
            return False
        reservation = rows[0]["requested_output_tokens"]
        return (all(row["requested_output_tokens"] == reservation for row in rows)
                and result.get("workload_sha256") == digest(dict(prompts=fingerprints,
                    output_tokens=reservation, temperature=0, seed=17, ignore_eos=True))
                and result.get("client_overlap", {}).get("max_requests") == maximum_overlap(rows))
    except (ValueError, TypeError, KeyError, AttributeError):
        return False


def refresh_targets(result):
    """Recompute every performance annotation from validated records, never cached verdicts."""
    profile = result.get("profile", {})
    identity_ok = has_identity(profile) and profile["identity"].get("model") == result.get("model")
    records_ok = valid_run_records(result)
    t = result.get("configured_t")
    statuses = []
    for row in result["requests"]:
        valid = valid_accounting(row, t)
        if not valid:
            row["accepted_tokens_per_second"] = None
            row["token_accounting_source"] = "unavailable"
        row["accounting_validation"] = "consistent" if valid else "unavailable"
        rate = row.get("accepted_tokens_per_second")
        row["rate_target_status"] = ("unavailable" if not valid or not records_ok or not identity_ok
                                     else "pass" if rate >= 55 else "fail")
        row["within_desired_55_75_band"] = None if not valid else 55 <= rate <= 75
        statuses.append(row["rate_target_status"])
    result["rate_target"] = dict(status="unavailable" if not statuses or "unavailable" in statuses else
                                 "pass" if all(s == "pass" for s in statuses) else "fail",
                                 lower_tokens_per_second=55, preferred_band=[55, 75])
    result["capacity_target"] = dict(status="unavailable" if not identity_ok or not records_ok else
        "pass" if all(row["exercised_context_tokens"] == t for row in result["requests"]) else "fail",
        basis="supplied token IDs plus accepted output; configured ceiling alone is insufficient")


def run_benchmark(profile, fixture, url, model, *, output_tokens=256, concurrency=None,
                  timeout=1800, dry_run=False, run_id=None, metrics=None, baseline=None, api_key=None,
                  served_model=None):
    n, t, fingerprints = validate(profile, fixture, url, model, output_tokens, concurrency, timeout)
    if run_id is not None and (not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", run_id)):
        raise ValueError("run_id must be 1-128 ASCII letters, digits, dots, underscores or hyphens")
    if served_model is not None and (not isinstance(served_model, str) or not served_model.strip()):
        raise ValueError("served_model must be a nonempty string")
    safe_profile = {key: profile[key] for key in PROFILE_KEYS if key in profile}
    safe_profile["identity"] = {key: profile.get("identity", {})[key] for key in IDENTITY_KEYS
                                if key in profile.get("identity", {})}
    safe_profile["tier_budgets"] = {key: profile.get("tier_budgets", {}).get(key) for key in ("vram", "ram", "ssd")}
    result = dict(schema_version=1, run_id=run_id or str(uuid.uuid4()),
                  captured_at=datetime.now(timezone.utc).isoformat(),
                  status="dry-run" if dry_run else "complete", profile=safe_profile,
                  metadata_provenance="operator-supplied-unattested", model=model, served_model=served_model or model,
                  workload_sha256=digest(dict(prompts=fingerprints, output_tokens=output_tokens,
                                               temperature=0, seed=17, ignore_eos=True)),
                  configured_n=n, configured_t=t, requests=[])
    if dry_run:
        result["validated_prompt_lengths"] = [len(prompt) for prompt in fixture["prompts"]]
        return result
    origin = time.perf_counter()
    def worker(index):
        prompt = fixture["prompts"][index]
        payload = dict(model=served_model or model, prompt=prompt, max_tokens=output_tokens, temperature=0, seed=17,
                       ignore_eos=True, stream=True, stream_options={"include_usage": True})
        row = stream_request(url, payload, f"{result['run_id']}:{index}", origin, timeout, api_key)
        row.update(prompt_sha256=fingerprints[index], supplied_prompt_tokens=len(prompt),
                   requested_output_tokens=output_tokens, reserved_context_tokens=len(prompt) + output_tokens,
                   exercised_context_tokens=None if row["accepted_output_tokens"] is None else
                   len(prompt) + row["accepted_output_tokens"])
        return row
    with ThreadPoolExecutor(max_workers=concurrency or n) as executor:
        result["requests"] = list(executor.map(worker, range(n)))
    result["client_overlap"] = dict(max_requests=maximum_overlap(result["requests"]), configured_requests=n)
    refresh_targets(result)
    result["evidence"] = evidence_summary(result, metrics)
    result["baseline_comparison"] = compare_baseline(result, baseline)
    return result


def evidence_summary(result, metrics):
    evidence = {"active_decoders": {"status": "unverified"}, "active_paging": {"status": "unverified"},
                "provenance": "operator-supplied-unattested"}
    if not valid_run_records(result) or not isinstance(metrics, dict) or metrics.get("run_id") != result["run_id"]:
        return evidence
    rows = {row["request_id"]: row for row in result["requests"] if row["error"] is None}
    def within(request_id, start, end):
        row = rows.get(request_id)
        return (row is not None and type(start) in (int, float) and type(end) in (int, float)
                and math.isfinite(start) and math.isfinite(end)
                and row["started_seconds"] <= start <= end <= row["ended_seconds"])
    intervals = metrics.get("scheduler_intervals", [])
    for interval in intervals if isinstance(intervals, list) else []:
        if not isinstance(interval, dict):
            continue
        ids = interval.get("request_ids", [])
        start, end = interval.get("start_seconds"), interval.get("end_seconds")
        if (isinstance(ids, list) and all(isinstance(item, str) for item in ids)
                and len(set(ids)) == result["configured_n"] and interval.get("phase") == "decode"
                and all(within(item, start, end) for item in ids) and end > start):
            evidence["active_decoders"] = dict(status="supported-by-supplied-trace", active_requests=len(set(ids)))
    total, count = 0, 0
    transfers = metrics.get("transfers", [])
    for transfer in transfers if isinstance(transfers, list) else []:
        if not isinstance(transfer, dict):
            continue
        at = transfer.get("at_seconds")
        if (transfer.get("purpose") == "active_kv" and transfer.get("direction") in {"d2h", "h2d"}
                and type(transfer.get("bytes")) is int and transfer["bytes"] > 0
                and isinstance(transfer.get("request_id"), str)
                and within(transfer["request_id"], at, at)):
            total += transfer["bytes"]
            count += 1
    if count:
        evidence["active_paging"] = dict(status="supported-by-supplied-trace", transfer_bytes=total,
                                          transfer_events=count)
    return evidence


def compare_baseline(result, baseline):
    unavailable = dict(status="unavailable", reason="compatible complete resident baseline required")
    if not valid_run_records(result) or not valid_run_records(baseline):
        return unavailable
    try:
        profiles = [item["profile"] for item in (result, baseline)]
        if not all(has_identity(profile) for profile in profiles):
            return unavailable
        for item, profile in zip((result, baseline), profiles):
            if (profile["identity"]["model"] != item["model"]
                    or profile["active_sequences"] != item["configured_n"]
                    or profile["context_tokens"] != item["configured_t"]):
                return unavailable
        if profiles[1].get("storage_mode") != "off" or profiles[1].get("execution_mode") != "resident":
            return unavailable
        if any(profiles[0]["identity"][key] != profiles[1]["identity"][key] for key in IDENTITY_KEYS):
            return unavailable
        for key in ("configured_n", "configured_t", "model", "served_model", "workload_sha256"):
            if not result.get(key) or result[key] != baseline.get(key):
                return unavailable
        if profiles[0].get("kv_dtype") != "turboquant_k8v4" or profiles[0].get("kv_dtype") != profiles[1].get("kv_dtype"):
            return unavailable
        n = result["configured_n"]
        if any(len(item["requests"]) != n or item["client_overlap"]["max_requests"] != n
               for item in (result, baseline)):
            return unavailable
        comparisons = []
        def finite_positive(value):
            return type(value) in (int, float) and math.isfinite(value) and value > 0
        for current, resident in zip(result["requests"], baseline["requests"]):
            for key in ("prompt_sha256", "supplied_prompt_tokens", "requested_output_tokens", "accepted_output_tokens"):
                if current.get(key) is None or current[key] != resident.get(key):
                    return unavailable
            if any(row.get("error") is not None or not row.get("stream_done")
                   or not finite_positive(row.get("accepted_tokens_per_second")) for row in (current, resident)):
                return unavailable
            ratios = {}
            for source, target in (("elapsed_seconds", "total_latency_ratio"),
                                   ("first_token_latency_seconds", "first_token_latency_ratio")):
                if not all(finite_positive(row.get(source)) for row in (current, resident)):
                    return unavailable
                ratios[target] = current[source] / resident[source]
            a, b = current.get("max_stream_gap_seconds"), resident.get("max_stream_gap_seconds")
            ratios["max_stream_gap_ratio"] = a / b if finite_positive(a) and finite_positive(b) else None
            ratios.update(prompt_sha256=current["prompt_sha256"],
                          status="pass" if max(ratios["total_latency_ratio"], ratios["first_token_latency_ratio"]) <= 1.10 else "fail")
            comparisons.append(ratios)
        return dict(status="pass" if all(row["status"] == "pass" for row in comparisons) else "fail",
                    ceiling_percent=10, preferred_5_percent=all(max(row["total_latency_ratio"],
                    row["first_token_latency_ratio"]) <= 1.05 for row in comparisons), requests=comparisons,
                    provenance="comparison of operator-supplied identities; no server attestation")
    except (KeyError, TypeError, AttributeError):
        return unavailable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("profile", "prompts", "output", "analyze"):
        parser.add_argument("--" + name, type=Path, required=name == "output")
    parser.add_argument("--url", help="Exact selected /v1/completions URL")
    parser.add_argument("--model")
    parser.add_argument("--served-model", help="Explicit endpoint alias; --model remains tokenizer/model identity")
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--api-key-env", help="Optional environment variable; key never enters reports")
    args = parser.parse_args()
    def read(path):
        return json.loads(path.read_text(encoding="utf-8-sig")) if path else None
    try:
        if args.analyze:
            if args.url or args.profile or args.prompts or args.model or args.served_model or args.api_key_env or args.dry_run:
                raise ValueError("analysis cannot send requests")
            result = read(args.analyze)
            if result.get("schema_version") != 1 or result.get("status") != "complete":
                raise ValueError("analysis requires a completed schema 1 run")
            if not isinstance(result.get("run_id"), str) or not isinstance(result.get("requests"), list):
                raise ValueError("analysis requires a run ID and request records")
            refresh_targets(result)
            result["evidence"] = evidence_summary(result, read(args.metrics))
            result["baseline_comparison"] = compare_baseline(result, read(args.baseline))
        else:
            if not all((args.profile, args.prompts, args.url, args.model)):
                raise ValueError("run requires profile, prompts, url and model")
            result = run_benchmark(read(args.profile), read(args.prompts), args.url, args.model,
                               output_tokens=args.output_tokens, concurrency=args.concurrency,
                               timeout=args.timeout, dry_run=args.dry_run, run_id=args.run_id,
                               metrics=read(args.metrics), baseline=read(args.baseline),
                               api_key=os.environ.get(args.api_key_env) if args.api_key_env else None,
                               served_model=args.served_model)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": result["status"], "run_id": result["run_id"],
                          "request_errors": sum(row["error"] is not None for row in result["requests"])}))
        return 2 if any(row["error"] for row in result["requests"]) else 0
    except (ValueError, OSError, TypeError, AttributeError, KeyError):
        print("ERROR: invalid benchmark inputs or inaccessible files; validate profile/token fixture", file=sys.stderr)
        return 64


if __name__ == "__main__":
    raise SystemExit(main())
