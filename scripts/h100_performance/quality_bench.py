"""Frozen synthetic generation checks, independent of performance tuning."""
import argparse
import hashlib
import json
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


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def evaluate(case, result):
    """Versioned exact oracle; never qualify partial or error responses."""
    message = {}
    finish = None
    error = None
    passed = False
    try:
        if result.get("error") is not None:
            error = "server_error"
        choices = result.get("choices", [])
        if len(choices) != 1:
            error = error or "invalid_choices"
        else:
            message = choices[0]["message"]
            finish = choices[0].get("finish_reason")
            expected_finish = "tool_calls" if "tools" in case else "stop"
            if finish != expected_finish:
                error = error or "invalid_finish_reason"
            if "tools" in case:
                calls = message.get("tool_calls") or []
                if len(calls) == 1 and not message.get("content"):
                    call = calls[0]
                    function = call.get("function", {})
                    arguments = json.loads(function.get("arguments", ""))
                    passed = (call.get("type") == "function" and function.get("name") == "record_measurement"
                              and isinstance(arguments, dict) and set(arguments) == set(case["expected"])
                              and all(type(arguments[key]) is type(value) and arguments[key] == value
                                      for key, value in case["expected"].items()))
            else:
                content = message.get("content")
                if isinstance(content, str) and not message.get("tool_calls"):
                    expected = case["expected"]
                    answer = content.strip()
                    # Only the one-word reasoning cases permit capitalization differences.
                    if case["id"].startswith("reasoning-") and expected.isalpha():
                        answer, expected = answer.casefold(), expected.casefold()
                    passed = answer == expected
    except (ValueError, TypeError, KeyError, AttributeError):
        error = error or "invalid_response"
    return dict(oracle="exact-v1", complete=error is None, error=error, finish_reason=finish,
                passed=passed and error is None, message=message, usage=result.get("usage") if isinstance(result, dict) else None,
                case_sha256=identity(case))


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
            try:
                with request(args.base_url, "/v1/chat/completions", payload) as response:
                    result = json.load(response)
                row = evaluate(case, result)
            except (OSError, ValueError) as exc:
                row = evaluate(case, {"error": {}})
                row["error"] = "request_error:" + type(exc).__name__
            row.update(id=case["id"], label=args.label, input_sha256=identity(payload),
                       seconds=time.perf_counter() - started)
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(json.dumps({key: row[key] for key in ("id", "label", "passed", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
