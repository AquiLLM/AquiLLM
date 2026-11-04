"""Bounded concurrent disconnect/recovery screen; no speed or slot-reuse claim.

Requires isolated server traffic and sequence capacity >=2. Metrics are logger
samples: failing to observe two running requests is an ambiguous failed screen.
The long survivor exercises prefill alongside an already streaming decode, but
actual chunk scheduling and physical recurrent-state reuse are not observable.
API credentials remain in memory and are never included in evidence.
"""
import argparse
import copy
import hashlib
import http.client
import json
import os
from pathlib import Path
import socket
import threading
import time
from urllib.parse import urlsplit
import urllib.request

from long_quality_bench import long_cases
from quality_bench import cases, evaluate, identity
from throughput_bench import captured_at, fetch_metrics, finite_positive, settled_metrics


def payloads(model, index, repetitions):
    survivor_case = copy.deepcopy(long_cases()[0])
    old, key = survivor_case["expected"], str(972047 + index * 137)
    prompt = survivor_case["messages"][0]["content"].replace(old, key)
    prompt = prompt.replace("Reply with the number only.",
        f"Reply with exactly {repetitions} copies of the archive key, separated by single spaces, and nothing else.")
    survivor_case.update(id=f"lifecycle-survivor-{index}", expected=" ".join([key] * repetitions),
                         messages=[dict(role="user", content=prompt)])
    survivor = dict(model=model, messages=survivor_case["messages"], temperature=0, seed=17,
        max_tokens=512, stream=True, stream_options={"include_usage": True},
        chat_template_kwargs={"enable_thinking": False})
    cancel = dict(model=model, prompt="Write a very long numbered list of fictional neutral projects. Explain each project in detail. ",
        max_tokens=8192, ignore_eos=True, temperature=0, seed=17, stream=True,
        stream_options={"include_usage": True})
    recovery = [copy.deepcopy(cases()[i]) for i in (0, 16, 1, 17, 2, 18)]
    for position, case in enumerate(recovery):
        old = case["expected"]
        new = str(981001 + index * 100 + position) if old.isdigit() else f"violet-{991001 + index * 100 + position}"
        for message in case["messages"]:
            message["content"] = message["content"].replace(old, new)
        case.update(id=f"lifecycle-recovery-{index}-{position}", expected=new)
    return dict(cancel=cancel, survivor=survivor, survivor_case=survivor_case, recovery_cases=recovery)


def headers():
    result = {"Content-Type": "application/json"}
    key = os.environ.get("VLLM_API_KEY", "")
    if key:
        result["Authorization"] = "Bearer " + key
    return result


def reject_constant(value):
    raise ValueError("Nonfinite JSON number")


class Stream:
    def __init__(self, base, path, payload, timeout):
        self.base, self.path, self.payload, self.timeout = base, path, payload, timeout
        self.first = threading.Event()
        self.done = threading.Event()
        self.sock = self.connection = None
        self.row = dict(input_sha256=identity(payload), started_at=captured_at(), events=[],
            output_text="", nonempty_chunks=0, stream_done=False, finish_reason=None,
            usage=None, request_id=None, cancel_requested=False, error=None)
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        if not self.row["cancel_requested"]:
            self.row["cancel_requested"] = True
            self.row["cancelled_at"] = captured_at()
        stream_socket = self.sock or getattr(self.connection, "sock", None)
        if stream_socket is not None:
            try:
                stream_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if self.connection is not None:
            self.connection.close()

    def _run(self):
        deadline = time.monotonic() + self.timeout
        try:
            target = urlsplit(self.base)
            if target.scheme not in ("http", "https") or target.username or target.password:
                raise ValueError("unsupported API URL")
            connection_type = http.client.HTTPSConnection if target.scheme == "https" else http.client.HTTPConnection
            self.connection = connection_type(target.hostname, target.port, timeout=min(30., self.timeout))
            self.connection.request("POST", target.path.rstrip("/") + self.path,
                                    body=json.dumps(self.payload).encode(), headers=headers())
            with self.connection.getresponse() as response:
                if response.status != 200:
                    raise ValueError("non-success HTTP response")
                # HTTPConnection clears sock for Connection:close responses.
                # Retain the response's underlying socket so shutdown interrupts
                # blocked readline before closing its buffered response file.
                self.sock = self.connection.sock or getattr(getattr(response.fp, "raw", None), "_sock", None)
                if self.sock is None:
                    raise ValueError("stream cancellation socket unavailable")
                while time.monotonic() < deadline and not self.row["cancel_requested"]:
                    raw = response.readline()
                    if not raw:
                        break
                    if raw.startswith(b"event:") and raw[6:].strip() == b"error":
                        raise ValueError("server error event")
                    if not raw.startswith(b"data:"):
                        continue
                    data = raw[5:].strip()
                    if data == b"[DONE]":
                        self.row["stream_done"] = True
                        break
                    event = json.loads(data, parse_constant=reject_constant)
                    self.row["events"].append(dict(captured_at=captured_at(), data=event))
                    if not isinstance(event, dict) or event.get("error") is not None:
                        raise ValueError("server error or invalid event")
                    event_id = event.get("id")
                    if event_id is not None:
                        if self.row["request_id"] not in (None, event_id):
                            raise ValueError("changing stream request identity")
                        self.row["request_id"] = event_id
                    if event.get("usage") is not None:
                        self.row["usage"] = event["usage"]
                    choices = event.get("choices", [])
                    if len(choices) > 1:
                        raise ValueError("multiple stream choices")
                    for choice in choices:
                        if choice.get("finish_reason") is not None:
                            self.row["finish_reason"] = choice["finish_reason"]
                        text = choice.get("text") or choice.get("delta", {}).get("content") or ""
                        if text:
                            if not isinstance(text, str):
                                raise ValueError("invalid stream text")
                            self.row["output_text"] += text
                            self.row["nonempty_chunks"] += 1
                            self.first.set()
                if time.monotonic() >= deadline and not self.row["stream_done"]:
                    self.row["error"] = "stream_deadline"
        except (OSError, ValueError, TypeError, AttributeError, http.client.HTTPException) as error:
            if not self.row["cancel_requested"]:
                self.row["error"] = "stream_error:" + type(error).__name__
        finally:
            if self.connection is not None:
                self.connection.close()
            self.row["captured_at"] = captured_at()
            self.row["output_sha256"] = hashlib.sha256(self.row["output_text"].encode()).hexdigest()
            self.done.set()


def wait_idle(base, settle, budget):
    time.sleep(settle)
    deadline, samples, consecutive = time.monotonic() + budget, [], 0
    while time.monotonic() < deadline:
        sample = fetch_metrics(base)
        samples.append(sample)
        consecutive = consecutive + 1 if sample["values"]["running"] == sample["values"]["waiting"] == 0 else 0
        if consecutive >= 2:
            return dict(sample, idle_samples=samples, settle_seconds=settle)
        time.sleep(0.2)
    raise ValueError("server did not drain")


def recovery_check(base, model, case, timeout):
    payload = dict(model=model, messages=case["messages"], temperature=0, seed=17,
        max_tokens=256, chat_template_kwargs={"enable_thinking": False})
    started = captured_at()
    try:
        req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                     json.dumps(payload).encode(), headers())
        with urllib.request.urlopen(req, timeout=min(30., timeout)) as response:
            raw = json.load(response, parse_constant=reject_constant)
        result = evaluate(case, raw)
    except (OSError, ValueError, TypeError, http.client.HTTPException) as error:
        raw = None
        result = dict(passed=False, complete=False, error="recovery_error:" + type(error).__name__)
    return dict(result, id=case["id"], input_sha256=identity(payload), expected=case["expected"],
                raw_result=raw, started_at=started, captured_at=captured_at())


def run_round(args, index):
    work = payloads(args.model, index, args.survivor_repetitions)
    row = dict(label=args.label, round=index, started_at=captured_at(), passed=False, complete=False,
        errors=[], overlap_sample=None, metric_samples=[], recovery=[],
        frozen_identity={key: identity(value) for key, value in work.items()},
        frozen_inputs=work,
        server_abort_confirmation="not_observed", physical_state_slot_reuse="not_observed",
        scope="Isolated traffic required; sampled scheduler overlap and client disconnect/recovery only",
        prompt_mix="Already streaming decode plus long survivor prefill; individual chunk steps not observed",
        settings=dict(timeout_seconds=args.timeout_seconds, overlap_seconds=args.overlap_seconds,
            drain_seconds=args.drain_seconds, metrics_settle_seconds=args.metrics_settle_seconds,
            cancel_after_chunks=args.cancel_after_chunks, survivor_repetitions=args.survivor_repetitions))
    cancel = Stream(args.base_url, "/v1/completions", work["cancel"], args.timeout_seconds)
    survivor = Stream(args.base_url, "/v1/chat/completions", work["survivor"], args.timeout_seconds)
    try:
        row["before"] = settled_metrics(args.base_url, args.metrics_settle_seconds)
        cancel.start()
        if not cancel.first.wait(args.overlap_seconds) or cancel.done.is_set():
            row["errors"].append("cancel_stream_not_active")
        else:
            survivor.start()
            row["survivor_launched_at"] = captured_at()
            deadline = time.monotonic() + args.overlap_seconds
            while time.monotonic() < deadline:
                sample = fetch_metrics(args.base_url)
                row["metric_samples"].append(sample)
                if (sample["values"]["running"] >= 2 and not survivor.done.is_set()
                        and not cancel.done.is_set() and cancel.row["nonempty_chunks"] >= args.cancel_after_chunks):
                    row["overlap_sample"] = sample
                    break
                if survivor.done.is_set() or cancel.done.is_set():
                    break
                time.sleep(0.2)
            if row["overlap_sample"] is None:
                row["errors"].append("active_overlap_not_proved")
            cancel.close()
            if not survivor.done.wait(args.timeout_seconds):
                row["errors"].append("survivor_deadline")
                survivor.close()
        cancel.close()
        cancel.thread.join(3)
        if cancel.thread.is_alive():
            row["errors"].append("cancel_reader_did_not_stop")
        if survivor.thread.ident is not None:
            survivor.thread.join(3)
        if survivor.thread.is_alive():
            row["errors"].append("survivor_reader_did_not_stop")
        row["cancelled"], row["survivor"] = dict(cancel.row), dict(survivor.row)
        row["cancelled"]["close_method"] = "socket_shutdown_then_connection_close"
        row["survivor"]["expected"] = work["survivor_case"]["expected"]
        row["cancelled"]["passed"] = (cancel.row["nonempty_chunks"] >= args.cancel_after_chunks
            and cancel.row["cancel_requested"] and not cancel.row["stream_done"]
            and cancel.row["finish_reason"] is None and bool(cancel.row["request_id"]) and cancel.row["error"] is None)
        usage = survivor.row["usage"] if isinstance(survivor.row["usage"], dict) else {}
        row["survivor"]["passed"] = (survivor.row["stream_done"] and survivor.row["error"] is None
            and survivor.row["finish_reason"] == "stop" and bool(survivor.row["request_id"])
            and survivor.row["request_id"] != cancel.row["request_id"]
            and cancel.row.get("cancelled_at", "z") < survivor.row.get("captured_at", "")
            and survivor.row["output_text"].strip() == work["survivor_case"]["expected"]
            and type(usage.get("prompt_tokens")) is int and 36864 <= usage["prompt_tokens"] <= 69632
            and type(usage.get("completion_tokens")) is int and 0 < usage["completion_tokens"] < 512)
        if not row["cancelled"]["passed"]:
            row["errors"].append("cancel_close_failed")
        if not row["survivor"]["passed"]:
            row["errors"].append("survivor_oracle_failed")
        row["drain"] = wait_idle(args.base_url, args.metrics_settle_seconds, args.drain_seconds)
        for case in work["recovery_cases"]:
            row["recovery"].append(recovery_check(args.base_url, args.model, case, args.timeout_seconds))
        if not all(item["passed"] for item in row["recovery"]):
            row["errors"].append("recovery_oracle_failed")
        row["after"] = wait_idle(args.base_url, args.metrics_settle_seconds, args.drain_seconds)
    except (OSError, ValueError, TypeError, http.client.HTTPException) as error:
        row["errors"].append("instrumentation_or_transport:" + type(error).__name__)
    finally:
        for stream in (cancel, survivor):
            if stream.thread.ident is not None and stream.thread.is_alive():
                stream.close()
                stream.thread.join(3)
        row.setdefault("cancelled", dict(cancel.row))
        row.setdefault("survivor", dict(survivor.row))
    row.update(captured_at=captured_at(), complete=not row["errors"], passed=not row["errors"],
               qualification="bounded_screen_pass" if not row["errors"] else "failed_or_ambiguous_screen")
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default=os.environ.get("VLLM_SERVED_MODEL_NAME", "qwen3.6:27b-mtp-awq"))
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--cancel-after-chunks", type=int, default=3)
    parser.add_argument("--survivor-repetitions", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=float, default=90)
    parser.add_argument("--overlap-seconds", type=float, default=20)
    parser.add_argument("--drain-seconds", type=float, default=30)
    parser.add_argument("--metrics-settle-seconds", type=float, default=6)
    args = parser.parse_args(argv)
    if (not 1 <= args.rounds <= 4 or not 3 <= args.cancel_after_chunks <= 32
            or not 8 <= args.survivor_repetitions <= 64
            or any(not finite_positive(value) for value in (args.timeout_seconds,
                args.overlap_seconds, args.drain_seconds, args.metrics_settle_seconds))
            or args.metrics_settle_seconds < 1 or args.timeout_seconds > 180
            or args.overlap_seconds > 60 or args.drain_seconds > 60
            or args.metrics_settle_seconds > 30):
        parser.error("Probe settings must stay within finite bounded limits")
    target = urlsplit(args.base_url)
    if target.scheme not in ("http", "https") or target.username or target.password or not target.hostname:
        parser.error("An API URL without embedded credentials is required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        for index in range(args.rounds):
            row = run_round(args, index)
            output.write(json.dumps(row, allow_nan=False) + "\n")
            output.flush()
            print(json.dumps({key: row[key] for key in ("label", "round", "passed", "errors")}), flush=True)
            if not row["passed"]:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
