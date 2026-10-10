"""Frozen synthetic generation checks, independent of performance tuning."""
import argparse
import json
import re
import time
from pathlib import Path

from serve_bench import request


def cases():
    result = []
    for i in range(8):
        value = str(730041 + 137 * i)
        notes = "\n".join(f"Record {j}: category neutral, score {j % 17}." for j in range(128))
        result.append(dict(id=f"number-{i}", expected=value,
                           messages=[dict(role="user", content=f"{notes}\nThe exact archive key for project Cedar-{i} is {value}.\nWhat is Cedar-{i}'s archive key? Reply with the number only.")]))
    tool = dict(type="function", function=dict(name="record_measurement", description="Record the requested measurement.",
                parameters=dict(type="object", properties={"label": {"type": "string"}, "value": {"type": "integer"}},
                                required=["label", "value"], additionalProperties=False)))
    for i in range(8):
        result.append(dict(id=f"tool-{i}", expected={"label": f"sample-{i}", "value": i + 17}, tools=[tool],
                           messages=[dict(role="user", content=f"Use record_measurement exactly once to record label sample-{i} and integer value {i + 17}.")]))
    for i in range(8):
        value = f"violet-{827 + i}"
        result.append(dict(id=f"multiturn-{i}", expected=value, messages=[
            dict(role="user", content=f"Remember: the deployment password in this fictional exercise is {value}."),
            dict(role="assistant", content="I will remember the fictional value for this exercise."),
            dict(role="user", content="Repeat the exact fictional value I gave you, without explanation.")]))
    reasoning = [("Compute 17 + 26. Reply with only the number.", "43"),
                 ("Compute 12 times 8. Reply with only the number.", "96"),
                 ("What is the capital of France? Reply with one word.", "Paris"),
                 ("Which is larger, 19 or 23? Reply with the larger number only.", "23"),
                 ("A box has 40 balls. Remove 13, then add 8. How many? Reply with only the number.", "35"),
                 ("Translate the English word cat into Spanish. Reply with one word.", "gato"),
                 ("How many sides does a hexagon have? Reply with only the number.", "6"),
                 ("If all robins are birds and Pip is a robin, is Pip a bird? Reply yes or no.", "yes")]
    for i, (prompt, expected) in enumerate(reasoning):
        result.append(dict(id=f"reasoning-{i}", expected=expected, messages=[dict(role="user", content=prompt)]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.6:27b-mtp-awq")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        for case in cases():
            payload = dict(model=args.model, messages=case["messages"], temperature=0, seed=17,
                           max_tokens=256, chat_template_kwargs={"enable_thinking": False})
            if "tools" in case:
                payload.update(tools=case["tools"], tool_choice="auto")
            started = time.perf_counter()
            with request(args.base_url, "/v1/chat/completions", payload) as response:
                result = json.load(response)
            message = result["choices"][0]["message"]
            if "tools" in case:
                calls = message.get("tool_calls") or []
                passed = len(calls) == 1 and calls[0]["function"]["name"] == "record_measurement"
                try:
                    passed = passed and json.loads(calls[0]["function"]["arguments"]) == case["expected"]
                except (IndexError, KeyError, ValueError):
                    passed = False
            else:
                passed = bool(re.search(r"(?<!\w)" + re.escape(case["expected"]) + r"(?!\w)",
                                        message.get("content") or "", re.IGNORECASE))
            row = dict(id=case["id"], label=args.label, passed=passed, message=message,
                       usage=result.get("usage"), seconds=time.perf_counter() - started)
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(json.dumps({key: row[key] for key in ("id", "label", "passed", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
