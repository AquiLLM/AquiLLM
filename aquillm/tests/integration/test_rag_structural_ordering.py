"""UNEXECUTED: native persisted metadata feeds a separately labelled offline policy."""
from copy import deepcopy
import json

import pytest

from apps.chat.evals.retrieval_treatment_manifest import treatment_manifest
from apps.chat.evals.structural_order_observation import section_order
from apps.ingestion.tests.output_support import upload as upload
from lib.embeddings.tests.test_document_provider import transport as transport
from .structural_native_files import BODY, REFERENCE
from .structural_native_support import native_source as native_source, detached

pytestmark = pytest.mark.django_db(transaction=True)


def test_current_native_reference_ordering_and_exact_lookup_control(native_source):
    source = native_source('body')
    assert source.extraction.parser_key == 'native-docx-ordered-v2'
    assert source.extraction.quality == 'complete'
    rows, sources = detached(source)
    reference = next(row for row in rows if REFERENCE in row['text'])
    body = next(row for row in rows if BODY in row['text'])
    assert reference['section_path'] == ['References'] and reference['kind'] == 'body'
    assert body['section_path'] == ['Results'] and body['kind'] == 'body'
    # Deliberate offline order-control inputs, not claimed production rank output.
    control = [reference, body]
    original = deepcopy(control)
    ordered = section_order(control, sources)
    assert ordered['original_order'] == [reference['chunk_id'], body['chunk_id']]
    assert ordered['experimental_order'] == [body['chunk_id'], reference['chunk_id']]
    assert ordered['moved_ids'] == [reference['chunk_id']]
    assert control == original
    lookup = section_order(control, sources, reference_lookup=True)
    assert lookup['experimental_order'] == lookup['original_order'] and lookup['moved_ids'] == []
    assert ordered['classifications'][body['chunk_id']] == 'ordinary'
    for literal in ('2028', '2029', '2030', 'Morrow (2027)', 'Chen (2026)',
                    '7 ug/L only at pH 6', 'measurements above pH 6 were excluded'):
        assert literal in body['text']
    raw = json.dumps(ordered, sort_keys=True).encode()
    manifest = treatment_manifest(configured={'structure': {'policy': ordered['policy']}},
        reports={'offline': raw}, associations={'structure': ['offline']})
    assert manifest['treatments']['structure']['result'] == 'inconclusive'
    assert manifest['reports']['offline']['runtime_verified'] is False
    assert manifest['reports']['offline']['activation_eligible'] is False
    assert manifest['activation_decision'] == 'not_issued_by_catalog'


@pytest.mark.parametrize('kind', ['scrambled', 'unknown'])
def test_uncertain_native_boundary_and_unknown_snapshot_receive_no_penalty(native_source, kind):
    source = native_source(kind)
    if kind == 'scrambled':
        assert source.extraction.quality == 'partial'
        assert 'layout_unavailable' in source.extraction.issue_codes
        assert all(e.kind != 'heading' for e in source.extraction.elements)
        for literal in ('References', 'Results', 'Synthetic comet calibration',
                        '7 ug/L only at pH 6', 'measurements above pH 6 were excluded'):
            assert literal in ' '.join(source.doc.full_text.split())
    else:
        assert source.extraction is None
        assert source.generation.source_snapshot.source_artifact_id is None
        assert source.doc.full_text == BODY
    rows, sources = detached(source)
    result = section_order(rows, sources)
    assert result['experimental_order'] == result['original_order']
    assert result['moved_ids'] == []
    assert set(result['classifications'].values()) <= {'uncertain', 'unknown'}


@pytest.mark.parametrize('mutation,reason', [
    ('revision', 'structural_revision_mismatch'), ('duplicate', 'structural_candidate_identity'),
    ('unknown_owner', 'unknown_structural_source'), ('slice', 'structural_chunk_membership_mismatch'),
    ('section', 'structural_chunk_membership_mismatch'), ('generation', 'structural_chunk_membership_mismatch'),
    ('unknown_chunk', 'structural_chunk_membership_mismatch'), ('other_real_chunk', 'structural_chunk_membership_mismatch'),
])
def test_offline_policy_rejects_changed_or_unbound_observations(native_source, mutation, reason):
    source = native_source('body')
    rows, sources = detached(source)
    assert section_order(rows, sources)['original_order'] == [row['chunk_id'] for row in rows]
    if mutation == 'revision':
        sources[str(source.doc.id)]['revision_sha256'] = '0' * 64
    elif mutation == 'duplicate':
        rows.append(deepcopy(rows[0]))
    elif mutation == 'unknown_owner':
        rows[0]['doc_id'] = 'missing-owner'
    elif mutation == 'slice':
        rows[0]['text'] += ' invented'
    elif mutation == 'section':
        rows[0]['section_path'] = ['References']
    elif mutation == 'unknown_chunk':
        rows[0]['chunk_id'] = max(row['chunk_id'] for row in rows) + 1
    elif mutation == 'other_real_chunk':
        replacement = rows[1]['chunk_id']
        rows = rows[:1]  # Avoid duplicate-ID rejection: a real ID still cannot identify another row.
        rows[0]['chunk_id'] = replacement
    else:
        rows[0]['generation'] = 'unobserved-generation'
    with pytest.raises(ValueError, match='^' + reason + '$'):
        section_order(rows, sources)
