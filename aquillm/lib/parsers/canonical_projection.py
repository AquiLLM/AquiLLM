"""Faithful canonical row projection and strict D3/D5 serialization.

Offsets are Unicode codepoints. Primary elements partition the text; figures
are overlays. These manifests establish source integrity, never authorization.
"""
from dataclasses import asdict
from hashlib import sha256
import json
import math

from .source_contracts import CellV1, ExtractionV1, SourceContractError, SourceElementV1, validate_artifact_ref

PROJECTION_POLICY = 'ordered-coordinate-projection-v1'
KINDS = {'heading', 'paragraph', 'table', 'formula', 'figure', 'separator'}


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('utf-8')


def column_name(column):
    result = ''
    while column:
        column, remainder = divmod(column - 1, 26)
        result = chr(65 + remainder) + result
    return result


def literal_text(value):
    if value is None or value == '':
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def visible(value):
    """Escape literal marker collisions and whitespace without dropping meaning."""
    if value is None:
        return '<blank>'
    if value in ('<blank>', '<unknown>') or any(c.isspace() or c in '"\\' for c in value):
        return json.dumps(value, ensure_ascii=False)
    return value


def project_row(cells, *, sheet=None, slide=None):
    prefix = f'sheet={visible(sheet)} ' if sheet is not None else ''
    if slide is not None:
        prefix += f'slide={slide} '
    prefix += f'row={cells[0].row}' if cells else 'row=<unknown>'
    parts = [prefix]
    for cell in cells:
        if cell.formula is not None:
            value = '<unknown>' if cell.value_status == 'uncached' else visible(cell.text)
            text = f'{cell.address} formula={visible(cell.formula)} value={value}'
        else:
            label = '|'.join(cell.header_keys)
            text = f'{cell.address} ' + (f'{visible(label)}=' if label else 'value=') + visible(cell.text)
        if cell.row_span != 1 or cell.column_span != 1:
            text += f' span={cell.row_span}x{cell.column_span}'
        parts.append(text)
    return ' '.join(parts)


def _invalid():
    raise SourceContractError('invalid_extraction_structure')


def _text(value, *, nullable=False, nonempty=False):
    if value is None and nullable:
        return
    if type(value) is not str or (nonempty and not value) or '\x00' in value:
        _invalid()
    try:
        value.encode('utf-8')
    except UnicodeError:
        _invalid()


def validate_element(element):
    if type(element) is not SourceElementV1 or type(element.kind) is not str or element.kind not in KINDS:
        _invalid()
    _text(element.element_key, nonempty=True)
    for value in (element.start, element.end):
        if type(value) is not int or value < 0:
            _invalid()
    if element.end < element.start:
        _invalid()
    for value in (element.page, element.slide):
        if value is not None and (type(value) is not int or value <= 0):
            _invalid()
    _text(element.sheet, nullable=True)
    _text(element.parent_key, nullable=True, nonempty=True)
    if type(element.cells) is not tuple:
        _invalid()
    positions = set()
    for cell in element.cells:
        if type(cell) is not CellV1:
            _invalid()
        for value in (cell.row, cell.column, cell.row_span, cell.column_span):
            if type(value) is not int or value <= 0:
                _invalid()
        if cell.address != f'{column_name(cell.column)}{cell.row}' or (cell.row, cell.column) in positions:
            _invalid()
        positions.add((cell.row, cell.column))
        _text(cell.text, nullable=True)
        _text(cell.formula, nullable=True, nonempty=True)
        if type(cell.header_keys) is not tuple:
            _invalid()
        for key in cell.header_keys:
            _text(key, nonempty=True)
        if cell.value_status not in ('literal', 'cached', 'uncached', 'blank'):
            _invalid()
        if ((cell.value_status in ('cached', 'uncached')) != (cell.formula is not None)
            or (cell.value_status in ('blank', 'uncached') and cell.text is not None)
            or (cell.value_status == 'literal' and cell.text is None)):
            _invalid()
    if element.bbox is not None:
        if type(element.bbox) is not tuple or len(element.bbox) != 4 or any(type(v) is not float or not math.isfinite(v) for v in element.bbox):
            _invalid()


def validate_extraction(extraction):
    if type(extraction) is not ExtractionV1:
        _invalid()
    validate_artifact_ref(extraction.artifact)
    _text(extraction.parser_key, nonempty=True)
    if len(extraction.parser_key) > 200:
        _invalid()
    _text(extraction.canonical_text)
    if extraction.quality not in ('complete', 'partial', 'unsupported') or type(extraction.issue_codes) is not tuple or type(extraction.elements) is not tuple:
        _invalid()
    for code in extraction.issue_codes:
        _text(code, nonempty=True)
    if len(set(extraction.issue_codes)) != len(extraction.issue_codes):
        _invalid()
    if extraction.quality != 'complete' and not extraction.issue_codes:
        _invalid()
    if extraction.quality == 'unsupported' and (extraction.canonical_text or extraction.elements):
        _invalid()
    position, keys = 0, set()
    for element in extraction.elements:
        validate_element(element)
        if element.element_key in keys or element.end > len(extraction.canonical_text):
            _invalid()
        if element.parent_key is not None and element.parent_key not in keys:
            _invalid()
        keys.add(element.element_key)
        if element.kind != 'figure':
            if element.start != position or (element.end == element.start and element.kind != 'separator'):
                _invalid()
            position = element.end
    if position != len(extraction.canonical_text):
        _invalid()


def elements_to_json(elements):
    if type(elements) is not tuple:
        _invalid()
    for element in elements:
        validate_element(element)
    return json.loads(canonical_json([asdict(element) for element in elements]))


def elements_from_json(value):
    if type(value) is not list:
        _invalid()
    result = []
    element_fields = set(SourceElementV1.__dataclass_fields__)
    cell_fields = set(CellV1.__dataclass_fields__)
    for item in value:
        if type(item) is not dict or set(item) != element_fields or type(item['cells']) is not list:
            _invalid()
        cells = []
        for cell in item['cells']:
            if type(cell) is not dict or set(cell) != cell_fields or type(cell['header_keys']) is not list:
                _invalid()
            cells.append(CellV1(**dict(cell, header_keys=tuple(cell['header_keys']))))
        if item['bbox'] is not None and type(item['bbox']) is not list:
            _invalid()
        element = SourceElementV1(**dict(item, cells=tuple(cells), bbox=tuple(item['bbox']) if item['bbox'] is not None else None))
        validate_element(element)
        result.append(element)
    return tuple(result)


def extraction_manifest(extraction):
    validate_extraction(extraction)
    return {'version': 'source-extraction-v1', 'parser_key': extraction.parser_key,
            'artifact': asdict(extraction.artifact),
            'canonical_sha256': sha256(extraction.canonical_text.encode('utf-8')).hexdigest(),
            'ordered_elements': elements_to_json(extraction.elements),
            'quality': extraction.quality, 'issue_codes': list(extraction.issue_codes)}
