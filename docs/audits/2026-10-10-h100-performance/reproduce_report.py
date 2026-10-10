"""Recompute the frozen development comparison locally; no model/server access."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'scripts/h100_performance'))
from report import build_report


def read_json(name):
    return json.loads((HERE / name).read_text(encoding='utf-8'))


def read_rows(names):
    return [json.loads(line) for name in names
            for line in (HERE / name).read_text(encoding='utf-8').splitlines() if line.strip()]


def main():
    blocks = (1, 2, 3)
    def rows(role):
        return read_rows([f'serving/h100-prefill-ab-{role}-block{block}.jsonl' for block in blocks])
    def metrics(role):
        return [read_json(f'serving/h100-prefill-ab-{role}-block{block}-{role}-block{block}-metrics.json')
                for block in blocks]
    result = build_report(
        rows('baseline'), rows('prefill'), target_contexts=[36864],
        baseline_metrics=metrics('baseline'), candidate_metrics=metrics('prefill'),
        baseline_quality=read_rows(['serving/h100-quality-baseline-strict.jsonl']),
        candidate_quality=read_rows(['serving/h100-quality-prefill-strict.jsonl']),
        evidence=read_json('serving-evidence.json'),
    )
    (HERE / 'serving-report.json').write_text(
        json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'status': result['status'], 'gates': result['gates']}, indent=2))


if __name__ == '__main__':
    main()
