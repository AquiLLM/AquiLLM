"""Synthetic direct-serving latency probe. MTP chunks are not token timestamps."""
import argparse
import hashlib
import http.client
import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def request(base, path, payload=None):
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("VLLM_API_KEY", "")
    if key:
        headers["Authorization"] = "Bearer " + key
    data = json.dumps(payload).encode() if payload is not None else None
    return urllib.request.urlopen(urllib.request.Request(base.rstrip("/") + path, data, headers), timeout=1800)


def stream_request(base, payload):
    started = time.perf_counter()
    first = None
    chunks = 0
    output = []
    usage = None
    headers_at = None
    done = False
    finish = None
    error = None
    try:
        with request(base, "/v1/completions", payload) as response:
            headers_at = time.perf_counter()
            for raw in response:
                if raw.startswith(b"event:") and raw[6:].strip() == b"error":
                    error = "server_error"
                if not raw.startswith(b"data:"):
                    continue
                data = raw[5:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                item = json.loads(data)
                if item.get("error") is not None:
                    error = "server_error"
                if item.get("usage") is not None:
                    usage = item["usage"]
                for choice in item.get("choices", []):
                    if choice.get("finish_reason") is not None:
                        finish = choice["finish_reason"]
                    text = choice.get("text") or choice.get("delta", {}).get("content") or ""
                    if text:
                        if first is None:
                            first = time.perf_counter()
                        chunks += 1
                        output.append(text)
    except (OSError, TimeoutError, http.client.HTTPException) as exc:
        error = "transport_error:" + type(exc).__name__
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        error = "invalid_stream:" + type(exc).__name__
    ended = time.perf_counter()
    count = usage.get("completion_tokens") if isinstance(usage, dict) else None
    if error is None:
        if not done:
            error = "missing_done"
        elif finish != ("length" if payload.get("ignore_eos") else "stop"):
            error = "invalid_finish_reason"
        elif type(count) is not int or count != payload["max_tokens"]:
            error = "invalid_completion_tokens"
        elif first is None:
            error = "missing_output"
    complete = error is None
    text = "".join(output)
    return dict(complete=complete, error=error, finish_reason=finish, stream_done=done,
                headers_seconds=headers_at - started if headers_at is not None else None,
                ttft_seconds=first - started if first is not None else None,
                total_seconds=ended - started, output_tokens=count, nonempty_chunks=chunks,
                aggregate_decode_seconds_per_token=(ended - first) / (count - 1)
                if complete and first is not None and count > 1 else None,
                output_text=text, output_sha256=hashlib.sha256(text.encode()).hexdigest(), usage=usage)


def speculation_metrics(base):
    with request(base, "/metrics") as response:
        return [line for line in response.read().decode().splitlines()
                if not line.startswith("#") and "spec_decode" in line]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default=os.environ.get("VLLM_SERVED_MODEL_NAME"))
    parser.add_argument("--prompt-tokens", default="512,2048,8192,32768")
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seed_text = "A synthetic performance note: the project has twelve numbered stages. Summarize the stages and explain their order. "
    with request(args.base_url, "/tokenize", {"model": args.model, "prompt": seed_text, "add_special_tokens": False}) as response:
        seed = json.load(response)["tokens"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    before = speculation_metrics(args.base_url)
    shapes = []
    with args.output.open("a", encoding="utf-8") as out:
        for length in map(int, args.prompt_tokens.split(",")):
            ids = (seed * (length // len(seed) + 1))[:length]
            payload = dict(model=args.model, prompt=ids, max_tokens=args.output_tokens,
                           temperature=0, seed=17, ignore_eos=True, stream=True,
                           stream_options={"include_usage": True})
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            shape_before = speculation_metrics(args.base_url)
            for index in range(-args.warmup, args.repeats):
                result = stream_request(args.base_url, payload)
                result.update(label=args.label, prompt_tokens=length, requested_output_tokens=args.output_tokens,
                              repeat=index, warmup=index < 0, input_sha256=digest,
                              captured_at=datetime.now(timezone.utc).isoformat())
                out.write(json.dumps(result) + "\n")
                out.flush()
                print(json.dumps(result), flush=True)
            shapes.append(dict(prompt_tokens=length, requested_output_tokens=args.output_tokens,
                               input_sha256=digest, before=shape_before,
                               after=speculation_metrics(args.base_url),
                               interval_includes_warmups=True))
    metrics_path = args.output.with_name(args.output.stem + "-" + args.label + "-metrics.json")
    metrics_path.write_text(json.dumps(dict(label=args.label, before=before,
                                           after=speculation_metrics(args.base_url),
                                           shapes=shapes), indent=2) + "\n")


if __name__ == "__main__":
    main()
