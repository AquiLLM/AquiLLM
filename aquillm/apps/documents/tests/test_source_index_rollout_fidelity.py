"""Synthetic fidelity measurements must expose omissions, never award fixture truth."""

import importlib
import json
from dataclasses import replace

import pytest


def evaluator():
    return importlib.import_module('apps.documents.evals.source_index_regression')


def test_real_sources_preserve_each_fact_cell_unknown_and_occurrence(capsys):
    report = evaluator().run_suite()
    assert report['fixture_fidelity_passed'] is True
    assert report['quality_gate'] == 'not_evaluated'
    outputs = {row['output_id']: row for row in report['outputs']}
    assert set(outputs) == {
        'signed', 'columns', 'formula', 'ordered',
        'archive:0', 'archive:1', 'archive:2', 'archive:3',
    }
    assert outputs['formula']['quality'] == 'partial'
    assert outputs['formula']['issue_codes'] == ['formula_cache_unavailable']
    assert outputs['archive:2']['issue_codes'] == ['empty_extraction']
    assert outputs['archive:3']['issue_codes'] == ['unsupported_member']
    assert outputs['archive:0']['artifact_sha256'] == outputs['archive:1']['artifact_sha256']
    assert outputs['archive:0']['occurrence_key'] != outputs['archive:1']['occurrence_key']
    for row in outputs.values():
        assert row['extraction']['passed'] is True
        assert row['extraction_observation']['elapsed_ms'] >= 0
        assert row['extraction_observation']['peak_python_bytes'] > 0
        for name in ('fixed_character', 'structural'):
            measured = row['variants'][name]
            assert measured['source_coverage'] == (1.0 if row['canonical_codepoints'] else None)
            assert measured['invalid_windows'] == 0
            assert measured['observation']['peak_python_bytes'] > 0
    assert outputs['signed']['variants']['fixed_character']['redundant_source_tokens'] > 0
    assert outputs['signed']['variants']['structural']['redundant_source_tokens'] > 0
    assert outputs['columns']['variants']['structural']['source_region_precision'] == 1.0
    assert outputs['columns']['variants']['fixed_character']['source_region_precision'] < 1.0
    print('I6_FIDELITY_MEASUREMENTS=' + json.dumps(report, sort_keys=True))
    assert 'I6_FIDELITY_MEASUREMENTS=' in capsys.readouterr().out


def test_measurement_output_can_be_retained_without_private_data():
    # -s retains actual observations in the sanctioned wrapper log.
    report = evaluator().run_suite()
    print('I6_FIDELITY_MEASUREMENTS=' + json.dumps(report, sort_keys=True))
    assert len(report['outputs']) == 8


def test_changed_signed_fact_is_not_awarded_from_expected_answers():
    module = evaluator()
    case = next(c for c in module.load_cases() if c['id'] == 'signed')
    case['source']['text'] = case['source']['text'].replace('-12.50', '+12.50')
    report = module.evaluate_case(case)
    assert report['outputs'][0]['extraction']['facts']['matched'] == 1
    assert report['outputs'][0]['extraction']['facts']['expected'] == 2
    assert report['passed'] is False


def test_swapping_table_values_cannot_pass_on_bag_of_numbers():
    module = evaluator()
    case = next(c for c in module.load_cases() if c['id'] == 'columns')
    case['source']['text'] = case['source']['text'].replace('Widget,,7,-4', 'Widget,7,,-4')
    report = module.evaluate_case(case)
    measured = report['outputs'][0]['extraction']
    assert measured['cells']['matched'] == 1
    assert measured['cells']['expected'] == 3
    assert measured['passed'] is False


def test_formula_literal_cannot_impersonate_uncached_formula():
    module = evaluator()
    case = next(c for c in module.load_cases() if c['id'] == 'formula')
    case['source']['rows'][1][1] = '70'
    report = module.evaluate_case(case)
    measured = report['outputs'][0]['extraction']
    assert measured['quality_and_issues_preserved'] is False
    assert measured['cells']['matched'] == 1
    assert measured['passed'] is False


def test_deleted_archive_member_is_reported_separately():
    module = evaluator()
    case = next(c for c in module.load_cases() if c['id'] == 'archive')
    case['source']['members'].pop()
    report = module.evaluate_case(case)
    assert report['missing_outputs'] == ['archive:3']
    assert report['passed'] is False
    assert len(report['outputs']) == 3


def test_wrong_source_region_fails_despite_present_fact():
    module = evaluator()
    case = next(c for c in module.load_cases() if c['id'] == 'formula')
    case['outputs'][0]['facts'][0]['region']['sheet'] = 'WrongSheet'
    report = module.evaluate_case(case)
    assert report['outputs'][0]['extraction']['facts']['matched'] == 0
    assert report['passed'] is False


def test_embedding_context_does_not_replace_missing_source_support():
    module = evaluator()
    source = module.extract_source('proof.txt', b'Loss -4 kg')
    windows = module.structural_windows(source)
    expected = [{'text': 'Loss -4 kg', 'region': {'kind': 'paragraph'}}]
    truncated = replace(windows[0], end=4, source_text='Loss', embedding_text='Loss -4 kg')
    metrics = module.measure_windows(source, (truncated,), expected)
    assert metrics['support_spans']['matched'] == 0
    assert metrics['source_coverage'] == 0.4


def test_false_source_coordinates_are_invalid_even_with_exact_text():
    module = evaluator()
    source = module.extract_source('proof.txt', b'Loss -4 kg')
    windows = module.structural_windows(source)
    shifted = replace(windows[0], start=1)
    metrics = module.measure_windows(source, (shifted,), [
        {'text': 'Loss -4 kg', 'region': {'kind': 'paragraph'}},
    ])
    assert metrics['invalid_windows'] == 1
    assert metrics['support_spans']['matched'] == 0


def test_omitted_chunk_is_measured_as_missing_coverage():
    module = evaluator()
    source = module.extract_source('proof.txt', b'Loss -4 kg')
    metrics = module.measure_windows(source, (), [
        {'text': 'Loss -4 kg', 'region': {'kind': 'paragraph'}},
    ])
    assert metrics['source_coverage'] == 0.0
    assert metrics['support_spans'] == {'matched': 0, 'expected': 1, 'recall': 0.0}


@pytest.mark.parametrize('repeat', [True, -1, 1001])
def test_source_generation_has_a_finite_strict_repeat_limit(repeat):
    with pytest.raises(ValueError, match='fixture_source_limit'):
        evaluator().build_source({
            'format': 'text', 'filename': 'huge.txt', 'text': '',
            'prefix': 'x', 'repeat': repeat,
        })


def test_fixed_windows_use_existing_production_preparation(monkeypatch):
    from apps.documents.tasks import chunking
    module = evaluator()
    source = module.extract_source('proof.txt', b'0123456789')
    original = chunking._prepared_chunks
    calls = []

    def observed(document):
        calls.append(document.full_text)
        return original(document)

    monkeypatch.setattr(chunking, '_prepared_chunks', observed)
    windows = module.fixed_windows(source)
    assert calls == ['0123456789']
    assert [(w.start, w.end, w.source_text) for w in windows] == [(0, 10, '0123456789')]


@pytest.mark.parametrize('case_id', ['formula', 'ordered', 'archive'])
def test_generated_source_bytes_are_repeatable(case_id):
    module = evaluator()
    case = next(c for c in module.load_cases() if c['id'] == case_id)
    assert module.build_source(case['source']) == module.build_source(case['source'])
