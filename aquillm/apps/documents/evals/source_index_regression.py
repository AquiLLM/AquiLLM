"""Synthetic fidelity only: real extraction and two chunkers on identical text.

No provider, database write, serving switch, held-out quality or rollout claim.
Latency is one in-process observation; tracemalloc excludes native allocations.
"""

import json
import time
import tracemalloc
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

from lib.parsers.source_chunking import chunk_extraction, chunker_key_for
from lib.parsers.source_contracts import occurrence_key, source_artifact_ref
from lib.parsers.structural import extract_structure

from .source_index_fixtures import build_source
from .source_index_metrics import SyntheticByteTokenizer, measure_extraction, measure_windows

FIXTURE = Path(__file__).parent / 'fixtures/source_index_cases.json'
STRUCTURAL_OPTIONS = {'tokenizer': SyntheticByteTokenizer(), 'max_tokens': 2048, 'overlap_tokens': 384}


@dataclass(frozen=True)
class Window:
    start: int
    end: int
    source_text: str
    embedding_text: str


def load_cases():
    return json.loads(FIXTURE.read_text(encoding='utf-8'))['cases']


def extract_source(filename, data):
    return extract_structure(filename, data, artifact=source_artifact_ref(
        data, member_path=None, container_sha256=None))


def structural_windows(extraction):
    return chunk_extraction(extraction, chunker_key=chunker_key_for(**STRUCTURAL_OPTIONS),
                            **STRUCTURAL_OPTIONS)


def fixed_windows(extraction):
    from apps.documents.tasks import chunking
    # This text-only carrier creates unsaved chunks; it is never owner authority.
    document = SimpleNamespace(full_text=extraction.canonical_text, id=UUID(int=1))
    chunks, image = chunking._prepared_chunks(document)
    assert image is None
    return tuple(Window(c.start_position, c.end_position, c.content, c.content) for c in chunks)


def _observe(function, *args):
    if tracemalloc.is_tracing():
        raise RuntimeError('fidelity_measurement_requires_exclusive_tracemalloc')
    tracemalloc.start()
    started = time.perf_counter_ns()
    try:
        result = function(*args)
        elapsed = (time.perf_counter_ns() - started) / 1_000_000
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return result, {'elapsed_ms': elapsed, 'peak_python_bytes': peak}


def _extract_outputs(case, data):
    if case['source']['format'] != 'zip':
        return [(case['id'], extract_source(case['source']['filename'], data), None)]
    from aquillm.ingestion.parsers import extract_text_payloads
    from apps.ingestion.services.source_artifact_transient import validate_transient_source
    outputs = []
    for payload in extract_text_payloads(case['source']['filename'], data):
        # Real I1 byte/ordinal validation checks the actual ZIP entry, not names alone.
        validate_transient_source(payload)
        leaf = payload.archive_lineage[-1]
        identity = {
            'occurrence_key': occurrence_key(payload.source_artifact,
                original_filename=payload.source_filename,
                content_type=payload.source_content_type, archive_lineage=payload.archive_lineage),
            'archive_lineage': [asdict(step) for step in payload.archive_lineage],
        }
        outputs.append((f"{case['id']}:{leaf.archive_ordinal}", payload.extraction, identity))
    return outputs


def evaluate_case(case):
    data = build_source(case['source'])
    extracted, extraction_observation = _observe(_extract_outputs, case, data)
    expected = {output['id']: output for output in case['outputs']}
    actual_ids = [output_id for output_id, _, _ in extracted]
    rows = []
    for output_id, extraction, identity in extracted:
        wanted = expected.get(output_id)
        variants = {}
        for name, function in [('fixed_character', fixed_windows), ('structural', structural_windows)]:
            windows, observation = _observe(function, extraction)
            variants[name] = dict(measure_windows(extraction, windows, wanted['facts'] if wanted else []),
                                  observation=observation)
        fidelity = measure_extraction(extraction, wanted) if wanted else {'passed': False}
        occurrence_matches = (identity is None and 'ordinal' not in (wanted or {})) or (
            identity is not None and wanted is not None
            and identity['archive_lineage'][-1]['archive_ordinal'] == wanted.get('ordinal')
            and identity['archive_lineage'][-1]['member_path'] == wanted.get('member_path'))
        rows.append({
            'output_id': output_id, 'artifact_sha256': extraction.artifact.sha256,
            'source_bytes': extraction.artifact.size_bytes,
            'canonical_sha256': sha256(extraction.canonical_text.encode('utf-8')).hexdigest(),
            'canonical_codepoints': len(extraction.canonical_text),
            'parser_key': extraction.parser_key, 'quality': extraction.quality,
            'issue_codes': list(extraction.issue_codes), 'extraction': fidelity,
            'occurrence_preserved': occurrence_matches, **(identity or {}),
            'extraction_observation': dict(extraction_observation, scope='case_all_outputs_shared'),
            'variants': variants,
        })
    missing = sorted(set(expected) - set(actual_ids))
    unexpected = sorted(set(actual_ids) - set(expected))
    passed = (not missing and not unexpected and len(actual_ids) == len(set(actual_ids))
              and all(row['extraction']['passed'] and row['occurrence_preserved']
                      and all(v['invalid_windows'] == 0 and v['source_coverage'] in (None, 1.0)
                              for v in row['variants'].values()) for row in rows))
    return {'case_id': case['id'], 'input_sha256': sha256(data).hexdigest(),
            'missing_outputs': missing, 'unexpected_outputs': unexpected,
            'outputs': rows, 'passed': passed}


def run_suite():
    from django.apps import apps
    config = apps.get_app_config('aquillm')
    cases = [evaluate_case(case) for case in load_cases()]
    return {
        'version': 'synthetic-source-index-measurements-v1',
        'fixture_sha256': sha256(FIXTURE.read_bytes()).hexdigest(),
        'fixture_fidelity_passed': all(case['passed'] for case in cases),
        'quality_gate': 'not_evaluated',
        'comparison': 'both_chunkers_receive_same_real_extracted_canonical_text',
        'token_unit': SyntheticByteTokenizer.revision,
        'fixed_character_config': {'chunk_size': config.chunk_size, 'chunk_overlap': config.chunk_overlap},
        'structural_config': {'max_tokens': 2048, 'overlap_tokens': 384,
                              'chunker_key': chunker_key_for(**STRUCTURAL_OPTIONS)},
        'measurement_scope': 'single_process_single_sample_python_allocations_only_no_provider',
        'cases': [{key: value for key, value in case.items() if key != 'outputs'} for case in cases],
        'outputs': [row for case in cases for row in case['outputs']],
    }
