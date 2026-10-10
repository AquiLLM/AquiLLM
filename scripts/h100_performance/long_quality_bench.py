"""Supplemental synthetic recall/tool checks that exercise long continuation prefill."""
import argparse
import json
import time
from pathlib import Path

from quality_bench import cases, evaluate, identity
from serve_bench import request


def long_cases():
    filler = " Neutral record: all parameters remain unchanged." * 5200
    short = cases()
    result = []
    for index, fraction in enumerate((0.05, 0.5, 0.95)):
        case = short[index].copy()
        position = int(len(filler) * fraction)
        marker = f"\nThe exact archive key for project Cedar-{index} is {case['expected']}.\n"
        prompt = filler[:position] + marker + filler[position:]
        prompt += f"\nWhat is Cedar-{index}'s exact archive key? Reply with the number only."
        case.update(id=f"long-number-{index}", messages=[dict(role="user", content=prompt)])
        result.append(case)
    for index, source in enumerate(short[8:10]):
        case = source.copy()
        expectation = case["expected"]
        marker = f"\nMeasurement to remember: label {expectation['label']}, integer value {expectation['value']}.\n"
        prompt = marker + filler if index == 0 else filler[:len(filler)//2] + marker + filler[len(filler)//2:]
        prompt += "\nCall record_measurement exactly once with the measurement I told you to remember."
        case.update(id=f"long-tool-{index}", messages=[dict(role="user", content=prompt)])
        result.append(case)
    case = short[16].copy()
    messages = [dict(message) for message in case["messages"]]
    messages[-1]["content"] = filler + "\n" + messages[-1]["content"]
    case.update(id="long-multiturn-0", messages=messages)
    result.append(case)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.6:27b-mtp-awq")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        for case in long_cases():
            payload = dict(model=args.model, messages=case["messages"], temperature=0, seed=17,
                           max_tokens=256, chat_template_kwargs={"enable_thinking": False})
            if "tools" in case:
                payload.update(tools=case["tools"], tool_choice="auto")
            started = time.perf_counter()
            try:
                with request(args.base_url, "/v1/chat/completions", payload) as response:
                    result = json.load(response)
                row = evaluate(case, result)
            except (OSError, ValueError) as error:
                row = evaluate(case, {"error": {}})
                row["error"] = "request_error:" + type(error).__name__
            row.update(id=case["id"], label=args.label, input_sha256=identity(payload),
                       seconds=time.perf_counter() - started)
            tokens = (row.get("usage") or {}).get("prompt_tokens", 0)
            row["long_context_exercised"] = 36864 <= tokens <= 69632
            row["passed"] = row["passed"] and row["long_context_exercised"]
            output.write(json.dumps(row) + "\n")
            output.flush()
            print(json.dumps({key: row[key] for key in ("id", "passed", "seconds", "usage")}), flush=True)


if __name__ == "__main__":
    main()
