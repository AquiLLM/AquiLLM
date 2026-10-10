"""Incomplete archives expose failed members without losing successful siblings."""
import pytest
from aquillm.task_ingest_uploaded import run_ingest_uploaded_file
from .output_support import upload

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize('members,code', [
    ((('good.txt', b'good'), ('empty.txt', b'')), 'empty_extraction'),
    ((('good.txt', b'good'), ('unknown.bin', b'opaque')), 'unsupported_member'),
])
def test_archive_dropped_member_is_visible(upload, members, code):
    item = upload(members)
    run_ingest_uploaded_file(item.pk)
    item.refresh_from_db()
    assert item.status == 'partial'
    outputs = item.parser_metadata['outputs']
    assert len(outputs) == 2
    assert outputs[0]['status'] == 'success'
    assert outputs[1]['error_code'] == code


def test_archive_cap_emits_one_bounded_remainder_failure(upload, monkeypatch):
    monkeypatch.setenv('INGEST_ARCHIVE_MAX_FILES', '1')
    item = upload(tuple((f'{i}.txt', str(i).encode()) for i in range(10)))
    run_ingest_uploaded_file(item.pk)
    item.refresh_from_db()
    assert item.status == 'partial'
    assert len(item.parser_metadata['outputs']) == 2
    assert item.parser_metadata['outputs'][1]['error_code'] == 'archive_member_limit'
