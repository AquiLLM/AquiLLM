import importlib
from dataclasses import replace
from hashlib import sha256

import pytest
from lib.embeddings.provenance import input_digest
from lib.parsers.source_contracts import SourceContractError
from .test_structural_extraction import extract


class LocalTokenizer:
    revision = 'synthetic-codepoints-v1'
    def __call__(self, text):
        return list(text)


def chunks(extraction, max_tokens=120, overlap_tokens=8, tokenizer=None):
    module = importlib.import_module('lib.parsers.source_chunking')
    tokenizer = tokenizer or LocalTokenizer()
    key = module.chunker_key_for(tokenizer=tokenizer, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    return module.chunk_extraction(extraction, chunker_key=key, tokenizer=tokenizer, max_tokens=max_tokens, overlap_tokens=overlap_tokens)


def test_contiguous_table_rows_preserve_source_offsets_and_schema():
    result = extract('rows.csv', b'Product,Domestic,Foreign\nWidget,,7\n')
    output = chunks(result)
    assert output
    for chunk in output:
        assert chunk.source_text == result.canonical_text[chunk.start:chunk.end]
        assert len(LocalTokenizer()(chunk.embedding_text)) <= 120
        assert chunk.source_sha256 == sha256(chunk.source_text.encode()).hexdigest()
        assert chunk.input_sha256 == input_digest(chunk.embedding_text)
    assert 'Domestic' in output[-1].embedding_text
    assert all(a.start <= b.start for a, b in zip(output, output[1:]))
    assert set(range(len(result.canonical_text))) == {i for chunk in output for i in range(chunk.start, chunk.end)}


def test_oversized_unicode_element_is_lossless_with_overlap():
    result = extract('notes.txt', ('−4 kg α\n' * 40).encode())
    output = chunks(result, max_tokens=30, overlap_tokens=5)
    assert len(output) > 2
    assert all(len(chunk.embedding_text) <= 30 for chunk in output)
    assert any(a.end > b.start for a, b in zip(output, output[1:]))
    assert set(range(len(result.canonical_text))) == {i for chunk in output for i in range(chunk.start, chunk.end)}


def test_budget_includes_heading_context_and_rejects_unrepresentable_header():
    result = extract('data.csv', b'A very long column name,value\nexample,7\n')
    with pytest.raises(SourceContractError, match='unrepresentable_source_element'):
        chunks(result, max_tokens=5, overlap_tokens=0)


def test_chunker_fingerprint_binds_budget_overlap_and_tokenizer_revision():
    module = importlib.import_module('lib.parsers.source_chunking')
    args = dict(tokenizer=LocalTokenizer(), max_tokens=120, overlap_tokens=8)
    first = module.chunker_key_for(**args)
    assert first != module.chunker_key_for(**dict(args, max_tokens=121))
    assert first != module.chunker_key_for(**dict(args, overlap_tokens=9))
    tokenizer = LocalTokenizer()
    tokenizer.revision = 'synthetic-codepoints-v2'
    assert first != module.chunker_key_for(**dict(args, tokenizer=tokenizer))
    result = extract('notes.txt', b'hello')
    with pytest.raises(SourceContractError, match='chunker_key_mismatch'):
        module.chunk_extraction(result, chunker_key='unknown', **args)


def test_malformed_partition_rejected_before_chunking():
    result = extract('notes.txt', b'hello')
    broken = replace(result, elements=(replace(result.elements[0], start=1),))
    with pytest.raises(SourceContractError, match='invalid_extraction_structure'):
        chunks(broken)


def test_heading_context_formula_qualifiers_and_exact_digest_vectors():
    from docx import Document
    import io
    doc = Document()
    doc.add_heading('Metrics', level=1)
    doc.add_paragraph(''.join(f'−4 kg\nα{number} ' for number in range(20)))
    stream = io.BytesIO()
    doc.save(stream)
    result = extract('metrics.docx', stream.getvalue())
    output = chunks(result, max_tokens=40, overlap_tokens=3)
    assert all(c.embedding_text.startswith('heading=Metrics\n') for c in output[1:])
    assert output[1].source_sha256 != output[2].source_sha256
    assert all(c.source_sha256 != sha256(result.canonical_text.encode()).hexdigest() for c in output)
    # Literal non-ASCII/newline vectors distinguish raw source hashing from
    # established input_digest JSON-string hashing, including escape handling.
    assert sha256('−4 kg\nα'.encode()).hexdigest() == '06dfe8406ad9219197231016bda1adb0913b4e2a4e12b8b55e15e78198d329de'
    assert input_digest('−4 kg\nα') == '1a54d3eb274ea95888db5c2f4c36064fd3b7e4fd0173fafe2986f59564485ae3'
    from .test_structural_extraction import workbook_bytes
    formula = extract('formula.xlsx', workbook_bytes([['Profit', '=50+20']]))
    formula_chunks = chunks(formula)
    assert any('formula==50+20 value=<unknown>' in c.embedding_text for c in formula_chunks)


def test_oversized_formula_row_repeats_unknown_value_qualifiers_in_embedding_context():
    from .test_structural_extraction import workbook_bytes
    result = extract('formulas.xlsx', workbook_bytes([['x' * 150, '=50+20']]))
    output = chunks(result, max_tokens=100, overlap_tokens=4)
    assert len(output) > 1
    assert all('B1 formula==50+20 value=<unknown>' in chunk.embedding_text for chunk in output)
    assert all(chunk.source_text == result.canonical_text[chunk.start:chunk.end] for chunk in output)
