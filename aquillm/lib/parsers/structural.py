"""Local ordered source extraction. No model, tokenization or persistence I/O."""
from dataclasses import replace
import csv
import io
from pathlib import PurePath

from .canonical_projection import column_name, literal_text, project_row, validate_extraction
from .source_contracts import CellV1, ExtractionV1, SourceContractError, SourceElementV1, validate_artifact_ref

# Expand compact repeated-cell structures only within a finite local ceiling.
MAX_CELLS = 100_000
NATIVE_FORMATS = {'.docx', '.pdf', '.xlsx', '.xls', '.ods', '.csv', '.tsv', '.pptx', '.txt', '.md'}


class _Builder:
    def __init__(self):
        self.parts, self.elements, self.offset, self.issues = [], [], 0, []

    def add(self, text, kind='paragraph', *, cells=(), page=None, sheet=None, slide=None, parent_key=None, bbox=None):
        # Delimiters belong to the preceding primary element, so every codepoint
        # remains addressable without manufacturing embedding-only source text.
        if not text:
            text = '<blank>'
        text += '\n'
        key = f'element-{len(self.elements) + 1}'
        element = SourceElementV1(key, kind, self.offset, self.offset + len(text), page, sheet, slide, parent_key, cells, bbox)
        self.elements.append(element)
        self.parts.append(text)
        self.offset += len(text)
        return key

    def row(self, cells, **location):
        return self.add(project_row(cells, sheet=location.get('sheet'), slide=location.get('slide')), 'table', cells=cells, **location)

    def figure(self, *, page=None, sheet=None, slide=None, parent_key=None, bbox=None):
        key = f'element-{len(self.elements) + 1}'
        self.elements.append(SourceElementV1(key, 'figure', self.offset, self.offset, page, sheet, slide, parent_key, (), bbox))
        return key

    def issue(self, code):
        if code not in self.issues:
            self.issues.append(code)

    def finish(self, parser_key, artifact):
        # Preserve traditional single-paragraph wrapper output without a trailing
        # generated newline, keeping a strict partition in the returned snapshot.
        text = ''.join(self.parts)
        if text:
            text = text[:-1]
            for index, element in enumerate(self.elements):
                if element.end > len(text):
                    self.elements[index] = replace(element, start=min(element.start, len(text)), end=len(text))
        result = ExtractionV1(parser_key, artifact, text, tuple(self.elements), 'partial' if self.issues else 'complete', tuple(self.issues))
        validate_extraction(result)
        return result


def _cell(row, column, value=None, *, formula=None, status=None, row_span=1, column_span=1, headers=()):
    text = literal_text(value)
    return CellV1(row, column, f'{column_name(column)}{row}', text, formula,
                  status or ('literal' if text is not None else 'blank'), row_span, column_span, headers)


def _headers(values):
    return tuple((literal_text(value),) if literal_text(value) is not None else () for value in values)


def _csv(builder, data, delimiter):
    rows = list(csv.reader(io.StringIO(data.decode('utf-8-sig'), newline=''), delimiter=delimiter, strict=True))
    width = max((len(row) for row in rows), default=0)
    if len(rows) * width > MAX_CELLS:
        raise SourceContractError('source_structure_limit_exceeded')
    headers = _headers(rows[0] + [''] * (width - len(rows[0]))) if rows else ()
    for row_number, values in enumerate(rows, 1):
        values += [''] * (width - len(values))
        builder.row(tuple(_cell(row_number, column, value, headers=headers[column - 1] if row_number > 1 else ()) for column, value in enumerate(values, 1)))


def _xlsx(builder, data):
    from openpyxl import load_workbook
    formulas = cached = None
    try:
        formulas = load_workbook(io.BytesIO(data), read_only=False, data_only=False)
        cached = load_workbook(io.BytesIO(data), read_only=False, data_only=True)
        total = 0
        for sheet in formulas.worksheets:
            values = cached[sheet.title]
            total += sheet.max_row * sheet.max_column
            if total > MAX_CELLS:
                raise SourceContractError('source_structure_limit_exceeded')
            headers = _headers([c.value if c.data_type != 'f' else None for c in sheet[1]])
            spans = {(merged.min_row, merged.min_col): (merged.max_row - merged.min_row + 1, merged.max_col - merged.min_col + 1) for merged in sheet.merged_cells.ranges}
            for row in sheet.iter_rows():
                cells = []
                for original in row:
                    formula = original.value if original.data_type == 'f' else None
                    value = values.cell(original.row, original.column).value if formula is not None else original.value
                    status = ('uncached' if value is None else 'cached') if formula is not None else None
                    if status == 'uncached':
                        builder.issue('formula_cache_unavailable')
                    row_span, column_span = spans.get((original.row, original.column), (1, 1))
                    cells.append(_cell(original.row, original.column, value, formula=formula, status=status, row_span=row_span, column_span=column_span,
                                       headers=headers[original.column - 1] if original.row > 1 else ()))
                builder.row(tuple(cells), sheet=sheet.title)
            for image in sheet._images:
                builder.figure(sheet=sheet.title)
                builder.issue('figure_layout_unavailable')
            if sheet._charts:
                # Cell coordinates cannot account for chart titles, axis labels
                # and drawing-only facts. Keep the supported rows faithful and
                # explicitly classify the unavailable chart structure.
                builder.issue('spreadsheet_chart_structure_unavailable')
    finally:
        if formulas is not None:
            formulas.close()
        if cached is not None:
            cached.close()


def _xls(builder, data):
    import xlrd
    book = xlrd.open_workbook(file_contents=data, formatting_info=True)
    try:
        total = 0
        for sheet in book.sheets():
            total += sheet.nrows * sheet.ncols
            if total > MAX_CELLS:
                raise SourceContractError('source_structure_limit_exceeded')
            headers = _headers(sheet.row_values(0)) if sheet.nrows else ()
            spans = {(r0 + 1, c0 + 1): (r1 - r0, c1 - c0) for r0, r1, c0, c1 in sheet.merged_cells}
            for row in range(sheet.nrows):
                cells = []
                for column in range(sheet.ncols):
                    value = sheet.cell_value(row, column)
                    rspan, cspan = spans.get((row + 1, column + 1), (1, 1))
                    cells.append(_cell(row + 1, column + 1, value, row_span=rspan, column_span=cspan, headers=headers[column] if row else ()))
                builder.row(tuple(cells), sheet=sheet.name)
        # xlrd exposes values, not formula expressions/cache provenance. Do not
        # advertise complete extraction where those facts cannot be established.
        builder.issue('xls_formula_identity_unavailable')
    finally:
        book.release_resources()


def _ods(builder, data):
    from odf.opendocument import load
    from odf import teletype
    from odf.namespaces import TABLENS
    from odf.table import Table
    from odf.draw import Image
    doc = load(io.BytesIO(data))
    total = 0
    for table in doc.spreadsheet.getElementsByType(Table):
        row_number, headers = 1, ()
        # Descendant row/cell queries conflate nested structures. Read direct
        # children and supported row wrappers in their document order.
        def rows(node):
            for child in node.childNodes:
                if getattr(child, 'qname', None) == (TABLENS, 'table-row'):
                    yield child
                elif getattr(child, 'qname', None) in {(TABLENS, 'table-header-rows'), (TABLENS, 'table-row-group'), (TABLENS, 'table-rows')}:
                    yield from rows(child)
        for row in rows(table):
            images_in_row = row.getElementsByType(Image)
            repeat_rows = int(row.getAttribute('numberrowsrepeated') or 1)
            if repeat_rows < 1 or repeat_rows > MAX_CELLS:
                raise SourceContractError('source_structure_limit_exceeded')
            templates = []
            for cell in row.childNodes:
                if getattr(cell, 'qname', None) not in {(TABLENS, 'table-cell'), (TABLENS, 'covered-table-cell')}:
                    continue
                repeat_cells = int(cell.getAttribute('numbercolumnsrepeated') or 1)
                if repeat_cells < 1 or len(templates) + repeat_cells > MAX_CELLS:
                    raise SourceContractError('source_structure_limit_exceeded')
                covered = cell.qname == (TABLENS, 'covered-table-cell')
                text = None if covered else literal_text(teletype.extractText(cell))
                formula = None if covered else cell.getAttribute('formula')
                if text is None and not covered:
                    value_type = cell.getAttribute('valuetype')
                    attribute = {'float':'value', 'currency':'value', 'percentage':'value', 'boolean':'booleanvalue', 'date':'datevalue', 'time':'timevalue', 'string':'stringvalue'}.get(value_type)
                    if attribute:
                        text = literal_text(cell.getAttribute(attribute))
                status = ('uncached' if text is None else 'cached') if formula else None
                if status == 'uncached':
                    builder.issue('formula_cache_unavailable')
                rspan = int(cell.getAttribute('numberrowsspanned') or 1) if not covered else 1
                cspan = int(cell.getAttribute('numbercolumnsspanned') or 1) if not covered else 1
                templates.extend([(text, formula, status, rspan, cspan)] * repeat_cells)
            total += repeat_rows * len(templates)
            if total > MAX_CELLS:
                raise SourceContractError('source_structure_limit_exceeded')
            for _ in range(repeat_rows):
                cells = tuple(_cell(row_number, column, value, formula=formula, status=status, row_span=rspan, column_span=cspan, headers=headers[column - 1] if column <= len(headers) else ())
                              for column, (value, formula, status, rspan, cspan) in enumerate(templates, 1))
                if cells:
                    key = builder.row(cells, sheet=table.getAttribute('name') or 'Sheet')
                    for image in images_in_row:
                        builder.figure(sheet=table.getAttribute('name') or 'Sheet', parent_key=key)
                        builder.issue('figure_layout_unavailable')
                if row_number == 1:
                    headers = _headers([t[0] for t in templates])
                row_number += 1


def _docx(builder, data):
    from docx import Document
    from docx.table import Table, _Cell
    from docx.text.paragraph import Paragraph
    from docx.oxml.ns import qn
    doc = Document(io.BytesIO(data))
    def unavailable(node):
        if (node.tag == qn('w:altChunk') or any(next(node.iter(qn(tag)), None) is not None
                for tag in ('w:t', 'w:delText', 'w:instrText', 'a:t', 'a:blip'))):
            builder.issue('document_wrapper_structure_unavailable')

    # python-docx omits these wrappers from Paragraph.text/Table.rows. Native
    # body/cell content controls are supported below; run/row/cell controls and
    # revision/custom XML wrappers must not silently advertise completeness.
    block_parents = {qn('w:body'), qn('w:tc'), qn('w:sdtContent')}
    unsupported_wrappers = {qn(f'w:{name}') for name in
                            ('customXml', 'ins', 'del', 'moveFrom', 'moveTo', 'altChunk')}
    for node in doc.element.body.iter():
        if node.tag in unsupported_wrappers or (node.tag == qn('w:sdt') and node.getparent().tag not in block_parents):
            unavailable(node)

    def blocks(owner):
        # Unwrap controls in place, without descendant queries that conflate
        # table/cell order. An explicit stack also handles deeply nested controls.
        stack = [iter(owner)]
        while stack:
            child = next(stack[-1], None)
            if child is None:
                stack.pop()
            elif child.tag in (qn('w:p'), qn('w:tbl')):
                yield child
            elif child.tag == qn('w:sdt'):
                contents = child.findall(qn('w:sdtContent'))
                if len(contents) != 1:
                    builder.issue('document_wrapper_structure_unavailable')
                else:
                    stack.append(iter(contents[0]))
            else:
                unavailable(child)

    for relation in doc.part.rels.values():
        if relation.reltype.rsplit('/', 1)[-1] in {'header', 'footer', 'footnotes', 'endnotes', 'comments'}:
            element = getattr(relation.target_part, 'element', None)
            if element is None or any(node.text for node in element.iter(qn('w:t'))):
                builder.issue('auxiliary_document_content_unavailable')
    def paragraph(node, owner, parent=None):
        p = Paragraph(node, owner)
        for child in node:
            if child.tag not in (qn('w:r'), qn('w:hyperlink'), qn('w:pPr')):
                unavailable(child)
        key = parent
        if p.text:
            kind = 'heading' if p.style and p.style.name.startswith('Heading') else 'paragraph'
            key = builder.add(p.text, kind, parent_key=parent)
        for blip in node.iter(qn('a:blip')):
            builder.figure(parent_key=key)
            builder.issue('figure_layout_unavailable')
        return key
    def table(node, owner, parent=None):
        table_obj = Table(node, owner)
        for child in node:
            if child.tag not in (qn('w:tr'), qn('w:tblPr'), qn('w:tblGrid')):
                unavailable(child)
        headers = ()
        for row_number, row in enumerate(table_obj.rows, 1):
            for child in row._tr:
                if child.tag not in (qn('w:tc'), qn('w:trPr')):
                    unavailable(child)
            cells, tails, column = [], [], 1
            # Read original tc nodes instead of row.cells, which aliases merged
            # cells and repeats the merged origin's text in covered coordinates.
            before = row._tr.grid_before
            for col in range(1, before + 1):
                cells.append(_cell(row_number, col))
            column += before
            for tc in row._tr.tc_lst:
                cell = _Cell(tc, table_obj)
                span = tc.grid_span
                leading, tail, seen_nested = [], [], False
                for child in blocks(tc):
                    if child.tag == qn('w:tbl'):
                        seen_nested = True
                    if child.tag in (qn('w:p'), qn('w:tbl')):
                        if seen_nested:
                            tail.append(child)
                        else:
                            leading.append(Paragraph(child, cell).text)
                value = '\n'.join(leading)
                # A vertical continuation is an explicit blank coordinate.
                if tc.vMerge == 'continue':
                    value = None
                row_span = 1
                if tc.vMerge == 'restart':
                    for following in table_obj.rows[row_number:]:
                        next_column = following._tr.grid_before + 1
                        continuation = None
                        for following_tc in following._tr.tc_lst:
                            if next_column == column:
                                continuation = following_tc
                                break
                            next_column += following_tc.grid_span
                        if continuation is None or continuation.vMerge != 'continue' or continuation.grid_span != span:
                            break
                        row_span += 1
                cells.append(_cell(row_number, column, value, row_span=row_span, column_span=span,
                                   headers=headers[column - 1] if row_number > 1 and column <= len(headers) else ()))
                for covered in range(1, span):
                    cells.append(_cell(row_number, column + covered))
                tails.append((cell, tail, column, span))
                column += span
            for _ in range(row._tr.grid_after):
                cells.append(_cell(row_number, column))
                column += 1
            if row_number == 1:
                headers = _headers([cell.text for cell in cells])
            if not any(tail for _, tail, _, _ in tails):
                key = builder.row(tuple(cells), parent_key=parent)
                for cell, _, _, _ in tails:
                    for child in blocks(cell._tc):
                        if child.tag == qn('w:p'):
                            for blip in child.iter(qn('a:blip')):
                                builder.figure(parent_key=key)
                                builder.issue('figure_layout_unavailable')
            else:
                # Nested blocks belong at their exact position within each cell.
                # Emit per-cell row segments only where nesting requires it;
                # an ordinary row remains one full coordinate-bearing element.
                emitted_columns = set()
                for cell, tail, cell_column, span in tails:
                    segment = tuple(c for c in cells if cell_column <= c.column < cell_column + span)
                    emitted_columns.update(c.column for c in segment)
                    key = builder.row(segment, parent_key=parent)
                    for child in blocks(cell._tc):
                        if child in tail:
                            break
                        if child.tag == qn('w:p'):
                            for blip in child.iter(qn('a:blip')):
                                builder.figure(parent_key=key)
                                builder.issue('figure_layout_unavailable')
                    for child in tail:
                        if child.tag == qn('w:tbl'):
                            table(child, cell, key)
                        else:
                            paragraph(child, cell, key)
                omitted = tuple(c for c in cells if c.column not in emitted_columns)
                if omitted:
                    builder.row(omitted, parent_key=parent)
    for node in blocks(doc.element.body):
        if node.tag == qn('w:p'):
            paragraph(node, doc)
        elif node.tag == qn('w:tbl'):
            table(node, doc)


def _pptx(builder, data):
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE
    deck = Presentation(io.BytesIO(data))
    def shapes(items, slide_number, parent=None):
        for shape in items:
            box = tuple(float(v) for v in (shape.left, shape.top, shape.left + shape.width, shape.top + shape.height))
            if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                key = builder.add(f'slide={slide_number} group={shape.shape_id}', 'separator', slide=slide_number, parent_key=parent, bbox=box)
                shapes(shape.shapes, slide_number, key)
            elif shape.has_table:
                table = shape.table
                headers = _headers([cell.text for cell in table.rows[0].cells])
                for row_number, row in enumerate(table.rows, 1):
                    cells = tuple(_cell(row_number, col, None if cell.is_spanned else cell.text,
                                        row_span=cell.span_height if cell.is_merge_origin else 1,
                                        column_span=cell.span_width if cell.is_merge_origin else 1,
                                        headers=headers[col - 1] if row_number > 1 else ())
                                  for col, cell in enumerate(row.cells, 1))
                    builder.row(cells, slide=slide_number, parent_key=parent, bbox=box)
            elif shape.has_text_frame and shape.text:
                kind = 'heading' if shape.is_placeholder and shape.placeholder_format.idx == 0 else 'paragraph'
                builder.add(shape.text, kind, slide=slide_number, parent_key=parent, bbox=box)
            elif shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                builder.figure(slide=slide_number, parent_key=parent, bbox=box)
            elif not shape.has_text_frame:
                builder.issue('slide_shape_structure_unavailable')
    for number, slide in enumerate(deck.slides, 1):
        shapes(slide.shapes, number)


def _pdf(builder, data):
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    for number, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ''
        if text:
            builder.add(text, page=number)
        else:
            key = f'element-{len(builder.elements) + 1}'
            builder.elements.append(SourceElementV1(key, 'separator', builder.offset, builder.offset, number, None, None, None))
    builder.issue('layout_unavailable')


def extract_structure(filename: str, data: bytes, *, artifact) -> ExtractionV1:
    validate_artifact_ref(artifact, data=data)
    extension = PurePath(filename).suffix.lower()
    policy_version = 2 if extension in ('.docx', '.xlsx') else 1
    parser_key = f'native-{extension.lstrip(".")}-ordered-v{policy_version}'
    if extension not in NATIVE_FORMATS:
        return ExtractionV1(parser_key, artifact, '', (), 'unsupported', ('source_format_unsupported',))
    builder = _Builder()
    try:
        if extension in ('.csv', '.tsv'):
            _csv(builder, data, '\t' if extension == '.tsv' else ',')
        elif extension in ('.txt', '.md'):
            text = data.decode('utf-8-sig')
            if text:
                builder.add(text)
        else:
            {'.docx': _docx, '.xlsx': _xlsx, '.xls': _xls, '.ods': _ods, '.pptx': _pptx, '.pdf': _pdf}[extension](builder, data)
        return builder.finish(parser_key, artifact)
    except ImportError:
        return ExtractionV1(parser_key, artifact, '', (), 'unsupported', ('local_parser_unavailable',))
    except SourceContractError:
        raise
    except Exception as exc:
        raise SourceContractError('malformed_source_structure') from exc
